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
from typing import Optional, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
import base64
import cv2
import numpy as np
import traceback
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from huggingface_hub import hf_hub_download
from transformers import CLIPVisionModelWithProjection
from PIL import Image

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
    # ここでの to(device) や enable_model_cpu_offload は削除し、
    # すべてのコンポーネント（IP-Adapter等）をロードした後に設定します
except Exception as e:
    print(f"Error loading model: {e}")
    pipe = None
    genai_client = None

# InsightFace の初期化
try:
    print(f"Initializing InsightFace (cwd: {os.getcwd()})...")
    # root='./' の場合、./models/antelopev2/*.onnx を見に行く
    face_app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    print("InsightFace initialized successfully")
except Exception as e:
    print(f"Warning: Failed to initialize InsightFace: {e}")
    traceback.print_exc()
    face_app = None

# IP-Adapter weights のロード
try:
    if pipe is not None:
        # 1. Load CLIP Image Encoder (ViT-H - 1024 dim)
        # BigG (1280 dim) は入手困難なため、より汎用的な ViT-H を使用し、
        # 対応する weight の IP-Adapter をロードします
        print("Loading CLIP Image Encoder (ViT-H)...")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", 
            torch_dtype=torch.float16
        ).to(device)
        
        # Image Encoder をパイプラインに登録し、CPU Offload の管理対象にする
        pipe.register_modules(image_encoder=image_encoder)

        # 2. Load IP-Adapter for SDXL (ViT-H version)
        # ViT-H エンコーダーに対応した重みを使用します
        print("Loading IP-Adapter SDXL (ViT-H)...")
        pipe.load_ip_adapter(
            "h94/IP-Adapter", 
            subfolder="sdxl_models", 
            weight_name="ip-adapter_sdxl_vit-h.bin",
            image_encoder=image_encoder
        )
        pipe.set_ip_adapter_scale(0.7)
        print("IP-Adapter Standard SDXL loaded successfully")

        # すべてのモジュールロード完了後に CPU Offload を有効化
        if device == "cuda":
            print("Enabling model CPU offload for memory optimization (Finalizing)...")
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)

except Exception as e:
    print(f"Warning: Failed to load IP-Adapter: {e}")
    traceback.print_exc()

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
    face_image: Optional[str] = None # Base64 encoded image

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

        # 顔画像が提供されている場合の処理
        ip_adapter_image = None
        id_embeds = None
        if request.face_image and face_app is not None:
            try:
                # Base64デコード
                header, encoded = request.face_image.split(",", 1) if "," in request.face_image else (None, request.face_image)
                image_data = base64.b64decode(encoded)
                nparr = np.frombuffer(image_data, np.uint8)
                face_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # 顔認識
                faces = face_app.get(face_img)
                if len(faces) > 0:
                    # 最初の顔の特徴を使用 (最も大きい顔を選択)
                    face_info = sorted(faces, key=lambda x:(x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
                    # InsightFace の埋め込みベクトル (ID embeds)
                    id_embeds = torch.from_numpy(face_info.normed_embedding).unsqueeze(0).to(device, dtype=torch.float16)
                    
                    # IP-Adapter (PlusV2) が必要とするリファレンス画像（切り抜き・リサイズ）
                    # InsightFace の align を使って正規化された顔画像を取得
                    aligned_face = face_align.norm_crop(face_img, landmark=face_info.kps, image_size=224)
                    ip_adapter_image = Image.fromarray(cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB))
                    
                    print("Face identified and embeddings extracted for FaceID")
                else:
                    print("No face detected in the provided image")
            except Exception as e:
                print(f"Error processing face image: {e}")
                traceback.print_exc()

        # パラメータを指定して生成
        generate_kwargs = {
            "prompt": final_prompt,
            "negative_prompt": request.negative_prompt,
            "guidance_scale": request.guidance_scale,
            "num_inference_steps": request.num_inference_steps,
            "width": request.width,
            "height": request.height,
            "generator": generator
        }
        
        if ip_adapter_image:
            # Standard IP-Adapter は画像をリスト形式で受け取ります
            pipe.set_ip_adapter_scale(0.7)
            generate_kwargs["ip_adapter_image"] = [ip_adapter_image]
        else:
            # 顔指定がない場合はスケールを0にして無効化
            # 制約を満たすためダミー画像（黒）を渡す
            pipe.set_ip_adapter_scale(0.0)
            generate_kwargs["ip_adapter_image"] = [Image.new("RGB", (224, 224), (0, 0, 0))]

        image = pipe(**generate_kwargs).images[0]
        
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
        traceback.print_exc()
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
