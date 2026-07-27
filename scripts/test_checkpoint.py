"""Script de prueba rápida para un checkpoint entrenado de InstructPix2Pix.

Uso simple (un solo prompt):
    python scripts/test_checkpoint.py \
        --checkpoint models/checkpoints/grafito-magicbrush \
        --image assets/example.jpg \
        --prompt "make it look like a painting" \
        --output outputs/test_entrenado.jpg \
        --resolution 256

Uso batch (varios prompts desde JSON):
    python scripts/test_checkpoint.py \
        --checkpoint models/checkpoints/grafito-magicbrush \
        --image assets/example.jpg \
        --prompts-file assets/test_prompts.json \
        --output outputs/batch_test \
        --resolution 512
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prueba un checkpoint entrenado de Grafito.")
    parser.add_argument("--checkpoint", required=True, help="Ruta al checkpoint entrenado.")
    parser.add_argument("--image", required=True, help="Imagen de entrada.")
    parser.add_argument("--prompt", default=None, help="Instrucción de edición.")
    parser.add_argument(
        "--prompts-file",
        default=None,
        help="JSON con lista de prompts (ver assets/test_prompts.json).",
    )
    parser.add_argument("--output", default="outputs/test_checkpoint.jpg", help="Ruta de salida.")
    parser.add_argument("--resolution", type=int, default=256, help="Tamaño máximo del lado mayor.")
    parser.add_argument("--steps", type=int, default=20, help="Pasos de inferencia.")
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--image-guidance-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42, help="Seed para reproducibilidad.")
    return parser.parse_args()


def load_and_resize(image_path: str, max_size: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    return image


def safe_filename(text: str, max_len: int = 40) -> str:
    """Devuelve un nombre de archivo seguro a partir de un prompt."""
    cleaned = re.sub(r"[^\w\s-]", "", text).strip().lower()
    cleaned = re.sub(r"[-\s]+", "_", cleaned)
    return cleaned[:max_len]


def load_prompts(prompts_file: str) -> Iterable[dict]:
    """Carga prompts desde un JSON con el formato de assets/test_prompts.json."""
    data = json.loads(Path(prompts_file).read_text(encoding="utf-8"))
    entries = data if isinstance(data, list) else data.get("prompts", [])
    for entry in entries:
        if isinstance(entry, str):
            yield {"id": safe_filename(entry), "prompt": entry}
        else:
            yield entry


def run_inference(
    pipe,
    image: Image.Image,
    prompt: str,
    args: argparse.Namespace,
    seed_offset: int = 0,
) -> Image.Image:
    device = next(pipe.unet.parameters()).device
    generator = torch.Generator(device).manual_seed(args.seed + seed_offset)
    return pipe(
        prompt,
        image=image,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        generator=generator,
    ).images[0]


def main() -> None:
    args = parse_args()

    if not args.prompt and not args.prompts_file:
        raise SystemExit("Error: debes indicar --prompt o --prompts-file.")

    checkpoint_path = Path(args.checkpoint)
    checkpoint_id = str(checkpoint_path) if checkpoint_path.exists() else args.checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Cargando checkpoint desde {checkpoint_id}...")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        checkpoint_id,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe = pipe.to(device)

    print(f"Cargando imagen {args.image}...")
    image = load_and_resize(args.image, args.resolution)

    if args.prompts_file:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for entry in load_prompts(args.prompts_file):
            prompt_id = entry.get("id") or safe_filename(entry["prompt"])
            prompt = entry["prompt"]
            safe_prompt = safe_filename(prompt)
            out_name = f"{timestamp}_{prompt_id}_{safe_prompt}_seed{args.seed}.png"
            out_path = output_dir / out_name

            print(f"Generando imagen con prompt: '{prompt}'...")
            result = run_inference(pipe, image, prompt, args)
            result.save(out_path)
            print(f"Imagen guardada en: {out_path}")
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Generando imagen con prompt: '{args.prompt}'...")
        result = run_inference(pipe, image, args.prompt, args)
        result.save(output_path)
        print(f"Imagen guardada en: {output_path}")


if __name__ == "__main__":
    main()
