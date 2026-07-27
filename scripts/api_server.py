"""API local de edición con el checkpoint Grafito (FastAPI).

Microservicio para integrar el modelo en una web: carga el pipeline una vez
al arrancar y expone un endpoint HTTP. La web (cualquier stack) solo necesita
hacer un POST con la imagen y el prompt.

Uso:
    python scripts/api_server.py [--port 8000] [--host 127.0.0.1]

Endpoints:
    GET  /health  -> estado del servicio y del modelo.
    POST /edit    -> form-data: image (archivo), prompt (texto), seed (opcional).
                     Devuelve la imagen editada como PNG.
"""

import argparse
import io
import os
import sys
import tempfile
import threading
from pathlib import Path

# Sin límite artificial de memoria MPS (igual que en scripts/demo.py).
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

from grafito.config import EditConfig
from grafito.editor import edit_image, load_pipeline
from grafito.memory import get_device

CHECKPOINT = os.environ.get("GRAFITO_CHECKPOINT", "models/checkpoints/grafito-magicbrush-v2")
BASE_MODEL = "timbrooks/instruct-pix2pix"
RESOLUTION = 512
BORDER_CROP_PX = 8  # recorte anti-manchas de esquina (plan v3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="API local de edición Grafito (FastAPI).")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host de escucha (por defecto: 127.0.0.1)."
    )
    parser.add_argument("--port", type=int, default=8000, help="Puerto (por defecto: 8000).")
    return parser.parse_args()


def crop_borders(image: Image.Image, px: int) -> Image.Image:
    """Recorta px por lado para eliminar manchas de esquina esporádicas."""
    w, h = image.size
    if w > 2 * px and h > 2 * px:
        return image.crop((px, px, w - px, h - px))
    return image


device = get_device()
model_id = CHECKPOINT if Path(CHECKPOINT).exists() else BASE_MODEL
if model_id == BASE_MODEL:
    print(f"[aviso] Checkpoint no encontrado en {CHECKPOINT}; se usará el modelo base.")
print(f"Cargando {model_id} en {device}...")
pipe = load_pipeline(model_id=model_id, device=device)
if device != "cuda":
    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
pipe.set_progress_bar_config(disable=True)

app = FastAPI(title="Grafito API", version="0.2.0")
# CORS abierto para pruebas locales; restringir a tu dominio en producción.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
edit_lock = threading.Lock()  # una GPU: serializar ediciones para no hacer OOM


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "device": device, "checkpoint": model_id, "resolution": RESOLUTION}


@app.post("/edit")
def edit(image: UploadFile = File(...), prompt: str = Form(...), seed: int = Form(-1)) -> Response:
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="El prompt no puede estar vacío.")
    try:
        pil_image = Image.open(io.BytesIO(image.file.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Imagen inválida: {exc}") from exc

    config = EditConfig(
        resolution=RESOLUTION,
        num_inference_steps=20,
        guidance_scale=7.0,
        image_guidance_scale=1.5,
        seed=None if seed < 0 else seed,
    )
    with edit_lock:
        edited = edit_image(pipe, pil_image, prompt.strip(), config)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    edited = crop_borders(edited, BORDER_CROP_PX)
    buffer = io.BytesIO()
    edited.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")


if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
