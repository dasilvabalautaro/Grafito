"""Evaluación cuantitativa de un checkpoint de InstructPix2Pix.

Calcula LPIPS y CLIP score sobre un split del dataset procesado, comparando el
checkpoint entrenado con la línea base `timbrooks/instruct-pix2pix`.

Uso:
    python src/scripts/evaluate.py \
        --checkpoint models/checkpoints/grafito-magicbrush \
        --dataset data/processed/magicbrush \
        --split validation \
        --num_samples 100 \
        --output outputs/eval_report.json
"""

import argparse
import json
from pathlib import Path

import clip
import lpips
import torch
from datasets import load_from_disk
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalúa un checkpoint de Grafito.")
    parser.add_argument("--checkpoint", required=True, help="Ruta al checkpoint entrenado.")
    parser.add_argument(
        "--base_checkpoint",
        default="timbrooks/instruct-pix2pix",
        help="Checkpoint de la línea base para comparar.",
    )
    parser.add_argument("--dataset", default="data/processed/magicbrush", help="Dataset procesado.")
    parser.add_argument("--split", default="validation", help="Split a evaluar.")
    parser.add_argument("--num_samples", type=int, default=None, help="Número de ejemplos a evaluar.")
    parser.add_argument("--output", default="outputs/eval_report.json", help="Ruta del reporte JSON.")
    parser.add_argument("--resolution", type=int, default=256, help="Tamaño máximo del lado mayor.")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Pasos de inferencia.")
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42, help="Seed para reproducibilidad.")
    parser.add_argument("--device", default=None, help="Dispositivo (cuda/cpu).")
    return parser.parse_args()


def load_and_resize(image, max_size: int) -> Image.Image:
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")
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


def pil_to_tensor(image: Image.Image, device: str) -> torch.Tensor:
    tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    return tensor * 2 - 1  # LPIPS espera rango [-1, 1]


def main() -> None:
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Cargando dataset desde {args.dataset} (split={args.split})...")
    ds = load_from_disk(args.dataset)[args.split]
    if args.num_samples is not None:
        ds = ds.select(range(min(args.num_samples, len(ds))))

    print(f"Cargando checkpoint entrenado {args.checkpoint}...")
    pipe_trained = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.checkpoint,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    print(f"Cargando línea base {args.base_checkpoint}...")
    pipe_base = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.base_checkpoint,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    print("Cargando modelos de métricas...")
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    results = []

    for idx, example in enumerate(tqdm(ds, desc="Evaluando")):
        original_image = load_and_resize(example["original_image"], args.resolution)
        target_image = load_and_resize(example["edited_image"], args.resolution)
        prompt = example["edit_prompt"]

        generator = torch.Generator(device).manual_seed(args.seed)

        def generate(pipe):
            return pipe(
                prompt,
                image=original_image,
                num_inference_steps=args.num_inference_steps,
                image_guidance_scale=args.image_guidance_scale,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]

        pred_trained = generate(pipe_trained)
        pred_base = generate(pipe_base)

        # LPIPS vs target
        lpips_trained = lpips_fn(
            pil_to_tensor(pred_trained, device), pil_to_tensor(target_image, device)
        ).item()
        lpips_base = lpips_fn(
            pil_to_tensor(pred_base, device), pil_to_tensor(target_image, device)
        ).item()

        # CLIP similarity (prompt vs generated image)
        def clip_score(image, text):
            image_tensor = clip_preprocess(image).unsqueeze(0).to(device)
            text_tokens = clip.tokenize([text]).to(device)
            with torch.no_grad():
                image_features = clip_model.encode_image(image_tensor)
                text_features = clip_model.encode_text(text_tokens)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            return (image_features @ text_features.T).item()

        clip_trained = clip_score(pred_trained, prompt)
        clip_base = clip_score(pred_base, prompt)

        results.append(
            {
                "idx": idx,
                "prompt": prompt,
                "lpips_trained": lpips_trained,
                "lpips_base": lpips_base,
                "clip_trained": clip_trained,
                "clip_base": clip_base,
            }
        )

    # Agregados
    metrics = {
        "num_samples": len(results),
        "lpips_trained_mean": sum(r["lpips_trained"] for r in results) / len(results),
        "lpips_base_mean": sum(r["lpips_base"] for r in results) / len(results),
        "clip_trained_mean": sum(r["clip_trained"] for r in results) / len(results),
        "clip_base_mean": sum(r["clip_base"] for r in results) / len(results),
        "details": results,
    }

    metrics["lpips_improvement"] = metrics["lpips_base_mean"] - metrics["lpips_trained_mean"]
    metrics["clip_improvement"] = metrics["clip_trained_mean"] - metrics["clip_base_mean"]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\nReporte guardado en {output_path}")
    print(f"LPIPS entrenado: {metrics['lpips_trained_mean']:.4f}")
    print(f"LPIPS base:      {metrics['lpips_base_mean']:.4f}")
    print(f"Mejora LPIPS:    {metrics['lpips_improvement']:.4f} (positivo = mejor)")
    print(f"CLIP entrenado:  {metrics['clip_trained_mean']:.4f}")
    print(f"CLIP base:       {metrics['clip_base_mean']:.4f}")
    print(f"Mejora CLIP:     {metrics['clip_improvement']:.4f} (positivo = mejor)")


if __name__ == "__main__":
    main()
