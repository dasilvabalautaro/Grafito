"""Filtra pares ruidosos de MagicBrush antes de construir la mezcla v3.

El filtro ataca tres problemas observados en v2:

1. **Cambio visual casi nulo**: el par original/editado es idéntico perceptualmente
   pero el prompt pide una edición. Se detecta con LPIPS(original, edited) bajo.
2. **Prompt sin relación con la imagen**: el prompt no describe lo que realmente
   aparece en la imagen editada. Se detecta con CLIP similarity bajo.
3. **Prompt genérico sin señal**: frases como ``make it better`` o ``enhance``
   que no guían al modelo.

Requiere ``lpips`` y ``transformers`` (CLIP). Si ``lpips`` no está, el script
puede usar una métrica de diferencia de pixeles como fallback, aunque no es
recomendable.

Ejemplo:
    python src/scripts/filter_noisy_pairs.py \
        --input_dir data/processed/magicbrush \
        --output_dir data/processed/magicbrush_v3_prefilter \
        --lpips_threshold 0.03 \
        --clip_threshold 0.18 \
        --device auto
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from PIL import Image


GENERIC_PROMPTS = [
    "make it better",
    "make this better",
    "enhance",
    "improve quality",
    "improve the quality",
    "make it look better",
    "make it nicer",
    "fix it",
    "correct it",
]

KEEP_KEYWORDS = ["keep", "same", "unchanged", "preserve", "maintain"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filtra pares ruidosos de MagicBrush para v3."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/processed/magicbrush",
        help="DatasetDict procesado por prepare_magicbrush.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/magicbrush_v3_prefilter",
        help="Directorio de salida con el dataset filtrado.",
    )
    parser.add_argument(
        "--lpips_threshold",
        type=float,
        default=0.03,
        help="LPIPS máximo para considerar que el par tiene cambio visual nulo.",
    )
    parser.add_argument(
        "--clip_threshold",
        type=float,
        default=0.18,
        help="CLIP similarity mínima entre prompt e imagen editada.",
    )
    parser.add_argument(
        "--generic_threshold",
        type=int,
        default=3,
        help=(
            "Máximo de palabras permitidas en prompts genéricos antes de "
            "descartarlos (0 = no filtrar por genéricos)."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Dispositivo para CLIP/LPIPS: auto, cpu, cuda, mps.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Si se indica, procesa solo N ejemplos (útil para pruebas).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Tamaño de batch para CLIP.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def is_generic_prompt(prompt: str, max_words: int) -> bool:
    """Devuelve True si el prompt es una frase genérica sin señal útil."""
    lowered = prompt.lower().strip()
    for generic in GENERIC_PROMPTS:
        if generic in lowered:
            return True
    # Prompts cortos que no contienen verbos de acción claros.
    if max_words > 0:
        words = re.findall(r"\b[a-z]+\b", lowered)
        if len(words) <= max_words and not any(
            w in lowered
            for w in ("add", "remove", "replace", "change", "make", "turn", "give")
        ):
            return True
    return False


def is_keep_prompt(prompt: str) -> bool:
    """Devuelve True si el prompt indica explícitamente conservar la imagen."""
    lowered = prompt.lower()
    return any(kw in lowered for kw in KEEP_KEYWORDS)


class MetricsCalculator:
    """Calcula LPIPS y CLIP similarity para un dataset de pares de edición."""

    def __init__(self, device: torch.device, lpips_threshold: float, clip_threshold: float):
        self.device = device
        self.lpips_threshold = lpips_threshold
        self.clip_threshold = clip_threshold
        self.lpips_model = self._load_lpips()
        self.clip_processor, self.clip_model = self._load_clip()

    def _load_lpips(self) -> Any | None:
        try:
            import lpips

            print("Cargando LPIPS (alex)...")
            return lpips.LPIPS(net="alex").to(self.device)
        except Exception as e:
            print(f"No se pudo cargar lpips: {e}. Se usará fallback MSE.")
            return None

    def _load_clip(self) -> tuple[Any, Any]:
        from transformers import CLIPModel, CLIPProcessor

        print("Cargando CLIP (openai/clip-vit-base-patch32)...")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        model.eval()
        return processor, model

    @torch.no_grad()
    def lpips_score(self, img0: Image.Image, img1: Image.Image) -> float:
        """Devuelve la distancia LPIPS (o MSE fallback) entre dos imágenes."""
        # LPIPS requiere que ambas imágenes tengan exactamente el mismo tamaño.
        if img0.size != img1.size:
            target = img0.size
            img1 = img1.resize(target, Image.LANCZOS)

        if self.lpips_model is not None:
            import lpips

            t0 = lpips.im2tensor(np.array(img0)).to(self.device)
            t1 = lpips.im2tensor(np.array(img1)).to(self.device)
            score = self.lpips_model(t0, t1).item()
            return float(score)

        # Fallback: distancia media cuadrática normalizada en [0,1]
        arr0 = np.asarray(img0).astype(np.float32) / 255.0
        arr1 = np.asarray(img1).astype(np.float32) / 255.0
        return float(np.mean((arr0 - arr1) ** 2))

    @torch.no_grad()
    def clip_similarity(self, prompts: list[str], images: list[Image.Image]) -> list[float]:
        """Devuelve la similitud coseno CLIP entre prompts e imágenes."""
        inputs = self.clip_processor(
            text=prompts, images=images, return_tensors="pt", padding=True
        ).to(self.device)
        outputs = self.clip_model(**inputs)
        logits_per_text = outputs.logits_per_text
        # Normalizar por las magnitudes para obtener coseno verdadero.
        text_embeds = outputs.text_embeds
        image_embeds = outputs.image_embeds
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        similarities = (text_embeds @ image_embeds.T).diag().cpu().tolist()
        return [float(s) for s in similarities]


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Dispositivo: {device}")

    print(f"Cargando dataset desde {input_dir}...")
    ds = load_from_disk(str(input_dir))
    train = ds["train"]
    if args.max_samples:
        train = train.select(range(min(args.max_samples, len(train))))
    print(f"Ejemplos a evaluar: {len(train)}")

    calc = MetricsCalculator(device, args.lpips_threshold, args.clip_threshold)

    # Pre-calcular LPIPS secuencialmente (es rápido en GPU/MPS, lento en CPU).
    print("Calculando LPIPS...")
    lpips_scores: list[float] = []
    for i, example in enumerate(train):
        score = calc.lpips_score(example["original_image"], example["edited_image"])
        lpips_scores.append(score)
        if (i + 1) % 500 == 0:
            print(f"  LPIPS: {i + 1}/{len(train)}")

    # Calcular CLIP en batches.
    print("Calculando CLIP similarity...")
    clip_scores: list[float] = []
    batch_prompts: list[str] = []
    batch_images: list[Image.Image] = []
    for i, example in enumerate(train):
        batch_prompts.append(example["edit_prompt"])
        batch_images.append(example["edited_image"])
        if len(batch_prompts) == args.batch_size or i == len(train) - 1:
            clip_scores.extend(calc.clip_similarity(batch_prompts, batch_images))
            batch_prompts.clear()
            batch_images.clear()
            if (i + 1) % 500 == 0:
                print(f"  CLIP: {i + 1}/{len(train)}")

    # Aplicar criterios de descarte.
    drop_reasons: dict[str, int] = {
        "low_lpips_no_change": 0,
        "low_clip_similarity": 0,
        "generic_prompt": 0,
    }
    keep_indices: list[int] = []

    for i, example in enumerate(train):
        prompt = example["edit_prompt"]
        lp = lpips_scores[i]
        cp = clip_scores[i]
        drop = False

        if lp < args.lpips_threshold and not is_keep_prompt(prompt):
            drop_reasons["low_lpips_no_change"] += 1
            drop = True

        if cp < args.clip_threshold:
            drop_reasons["low_clip_similarity"] += 1
            drop = True

        if args.generic_threshold > 0 and is_generic_prompt(prompt, args.generic_threshold):
            drop_reasons["generic_prompt"] += 1
            drop = True

        if not drop:
            keep_indices.append(i)

    filtered_train = train.select(keep_indices)
    validation = ds["validation"] if "validation" in ds else ds["dev"]

    result = DatasetDict({"train": filtered_train, "validation": validation})
    result.save_to_disk(str(output_dir))

    stats = {
        "input_train": len(train),
        "output_train": len(filtered_train),
        "dropped": len(train) - len(filtered_train),
        "dropped_fraction": (len(train) - len(filtered_train)) / len(train)
        if len(train)
        else 0.0,
        "drop_reasons": drop_reasons,
        "lpips_threshold": args.lpips_threshold,
        "clip_threshold": args.clip_threshold,
        "lpips_mean": float(np.mean(lpips_scores)) if lpips_scores else 0.0,
        "clip_mean": float(np.mean(clip_scores)) if clip_scores else 0.0,
    }
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Dataset filtrado guardado en {output_dir}")
    print(f"  Train: {stats['input_train']} -> {stats['output_train']}")
    print(f"  Descartados: {stats['dropped']} ({stats['dropped_fraction']:.2%})")
    print(f"  Razones: {drop_reasons}")
    print(f"  LPIPS medio: {stats['lpips_mean']:.4f}")
    print(f"  CLIP medio: {stats['clip_mean']:.4f}")


if __name__ == "__main__":
    main()
