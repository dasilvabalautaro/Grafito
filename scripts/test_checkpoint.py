"""Script de prueba rápida para un checkpoint entrenado de InstructPix2Pix.

Uso:
    python scripts/test_checkpoint.py \
        --checkpoint models/checkpoints/grafito-magicbrush \
        --image assets/example.jpg \
        --prompt "make it look like a painting" \
        --output outputs/test_entrenado.jpg \
        --resolution 256
"""

import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prueba un checkpoint entrenado de Grafito.")
    parser.add_argument("--checkpoint", required=True, help="Ruta al checkpoint entrenado.")
    parser.add_argument("--image", required=True, help="Imagen de entrada.")
    parser.add_argument("--prompt", required=True, help="Instrucción de edición.")
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


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint_id = str(checkpoint_path) if checkpoint_path.exists() else args.checkpoint

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    generator = torch.Generator(device).manual_seed(args.seed)

    print(f"Generando imagen con prompt: '{args.prompt}'...")
    result = pipe(
        args.prompt,
        image=image,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        generator=generator,
    ).images[0]

    result.save(output_path)
    print(f"Imagen guardada en: {output_path}")


if __name__ == "__main__":
    main()
