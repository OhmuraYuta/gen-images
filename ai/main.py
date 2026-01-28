from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import torch
import shutil
import os
import subprocess
import threading
from diffusers import StableDiffusionXLPipeline
from google import genai
from google.genai import types
import io
import re
import time
from PIL import Image
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

app = FastAPI()

# モデルのロード
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
trans_model_id = "Helsinki-NLP/opus-mt-ja-en"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

try:
    # Gemini Client の初期化
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        genai_client = genai.Client(api_key=gemini_api_key)
        print("Gemini API client initialized")
    else:
        genai_client = None
        print("Warning: GEMINI_API_KEY not found. Gemini-based translation will be unavailable.")

    # SDXL Pipeline をロード (安全フィルターを無効化)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        variant="fp16", 
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to(device)
except Exception as e:
    print(f"Error loading model: {e}")
    pipe = None
    genai_client = None

def contains_japanese(text: str) -> bool:
    # 日本語文字が含まれているかチェック
    return bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text))

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = "ugly, deformed, disfigured, poor quality, blurry, low res, bad anatomy, bad hands, text, error, missing fingers, bad legs,"
    guidance_scale: float = 7.5
    num_inference_steps: int = 40
    width: int = 1024
    height: int = 1024
    seed: Optional[int] = -1

@app.get("/health")
def health():
    return {"status": "ok", "device": device, "model": model_id}

@app.post("/generate")
async def generate(request: GenerateRequest):
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # 日本語が含まれている場合は翻訳
        final_prompt = request.prompt
        translated = False
        
        if genai_client is not None and contains_japanese(request.prompt):
            # Gemini にプロンプト職人になってもらう
            sys_instruct = (
                "You are an expert anime and photography prompt engineer for Stable Diffusion SDXL. "
                "Your task: Translate the user's Japanese input into highly effective English image generation tags. "
                "Optimization rules:\n"
                "1. Convert natural language into comma-separated tags.\n"
                "2. Automatically add cinematic lighting, masterpiece, high quality, and detailed background tags.\n"
                "3. If the user mentions a style (e.g., 'anime style'), emphasize relevant tags.\n"
                "4. Output ONLY the final English tags, no explanations."
            )
            
            # レート制限エラーなどの場合にリトライする関数
            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=2, max=10),
                retry=retry_if_exception(lambda e: "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e))
            )
            def generate_with_retry():
                return genai_client.models.generate_content(
                    model="gemini-2.0-flash",
                    config=types.GenerateContentConfig(system_instruction=sys_instruct),
                    contents=request.prompt
                )
            
            try:
                response = generate_with_retry()
                if response and response.text:
                    final_prompt = response.text.strip()
                    translated = True
                    print(f"Gemini Refined: {request.prompt} -> {final_prompt}")
            except Exception as e:
                print(f"Gemini API error after retries: {e}")
                # リトライに失敗した場合は、最悪日本語のままか、直前のプロンプトで試行
                pass

        # シード値の設定
        used_seed = request.seed
        if used_seed is None or used_seed < 0:
            used_seed = torch.seed() % (2**32) 
        
        generator = torch.Generator(device=device).manual_seed(used_seed)

        # パラメータを指定して生成
        image = pipe(
            prompt=final_prompt,
            negative_prompt=request.negative_prompt,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            width=request.width,
            height=request.height,
            generator=generator
        ).images[0]
        
        # 画像をメモリ上のバイト列に変換
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        headers = {
            "X-Translated": "true" if translated else "false",
            "X-Used-Seed": str(used_seed)
        }
        if translated:
            headers["X-Translated-Prompt"] = final_prompt

        return Response(content=byte_im, media_type="image/png", headers=headers)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 学習関連の機能 ---

TRAIN_DATA_DIR = "outputs/train/images"
TRAIN_OUTPUT_DIR = "outputs/train/output"
TRAIN_LOG_FILE = "outputs/train/training.log"

class TrainRequest(BaseModel):
    instance_prompt: str
    max_train_steps: int = 500
    learning_rate: float = 1e-4

from fastapi import UploadFile, File

@app.post("/train/upload")
async def upload_images(files: list[UploadFile] = File(...)):
    # 既存の画像を削除
    if os.path.exists(TRAIN_DATA_DIR):
        shutil.rmtree(TRAIN_DATA_DIR)
    os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
    
    for file in files:
        file_path = os.path.join(TRAIN_DATA_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
    return {"message": f"{len(files)} images uploaded successfully"}

@app.post("/train/start")
async def start_training(request: TrainRequest):
    # 非同期で学習を開始
    def run_training():
        # 推論用モデルを一旦メモリから逃がす（VRAM節約）
        global pipe
        temp_pipe = pipe
        pipe = None
        torch.cuda.empty_cache()
        
        try:
            with open(TRAIN_LOG_FILE, "w") as log_file:
                cmd = [
                    "poetry", "run", "python", "train.py",
                    "--instance_data_dir", TRAIN_DATA_DIR,
                    "--output_dir", TRAIN_OUTPUT_DIR,
                    "--instance_prompt", request.instance_prompt,
                    "--max_train_steps", str(request.max_train_steps),
                    "--learning_rate", str(request.learning_rate),
                    "--gradient_checkpointing"
                ]
                process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
                process.wait()
        finally:
            # 推論用モデルを再ロード（または戻す）
            pipe = temp_pipe
            if pipe:
                pipe.to(device)

    thread = threading.Thread(target=run_training)
    thread.start()
    
    return {"message": "Training started in background"}

@app.get("/train/status")
async def training_status():
    if not os.path.exists(TRAIN_LOG_FILE):
        return {"status": "idle"}
    
    with open(TRAIN_LOG_FILE, "r") as f:
        log_content = f.readlines()
        last_lines = log_content[-5:] if log_content else []
        
    return {
        "status": "running" if "Steps" in "".join(last_lines) else "finished/error",
        "last_log": last_lines
    }
