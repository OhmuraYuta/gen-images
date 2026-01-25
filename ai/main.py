from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import torch
from diffusers import StableDiffusionPipeline
import io
from PIL import Image

app = FastAPI()

# モデルのロード
model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

try:
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
except Exception as e:
    print(f"Error loading model: {e}")
    pipe = None

class GenerateRequest(BaseModel):
    prompt: str

@app.get("/health")
def health():
    return {"status": "ok", "device": device}

@app.post("/generate")
async def generate(request: GenerateRequest):
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        image = pipe(request.prompt).images[0]
        
        # 画像をメモリ上のバイト列に変換
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        return Response(content=byte_im, media_type="image/png")
    
except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
