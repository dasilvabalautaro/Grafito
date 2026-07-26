"""Auditoría de calidad del dataset procesado (puerta 0.2 del plan v2).

Mide tres cosas sobre el split train (ver docs/TRAINING_V2_PLAN.md):

1. Desplazamiento de color: diferencia media por canal RGB entre imagen
   original y editada. Un sesgo sistemático (p. ej. R y B suben, G baja)
   apuntaría a un origen de dataset para el tinte magenta observado en v1.
   Ojo: los cambios de color legítimos ("make it blue") también cuentan;
   lo que se reporta es la tendencia media global.
2. Anomalías de borde: % de imágenes editadas con píxeles brillantes y
   saturados (S>0.8, V>0.5) en los 8 px de borde — posible origen de las
   manchas de color esporádicas vistas en v1.
3. Cobertura de personas: % de prompts que mencionan personas (misma
   función `is_person_prompt` que prepare_v2_mix.py).

Uso:
    python scripts/audit_dataset.py [--dataset_dir data/processed/magicbrush] \
        [--max_examples N] [--output audit.json]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from datasets import load_from_disk

from scripts.prepare_v2_mix import is_person_prompt

BORDER_PX = 8
BORDER_FLAG_FRACTION = 0.01  # >1% de píxeles de borde anómalos -> imagen marcada
STATS_SIZE = 64  # lado para las estadísticas de color (velocidad)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auditoría de calidad del dataset.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/processed/magicbrush",
        help="DatasetDict procesado por prepare_magicbrush.py.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Límite de ejemplos a auditar (por defecto: todos).",
    )
    parser.add_argument("--output", type=str, default=None, help="Ruta JSON de salida.")
    return parser.parse_args()


def rgb_stats(image) -> np.ndarray:
    """Media por canal RGB de una imagen PIL reducida a STATS_SIZE."""
    small = image.convert("RGB").resize((STATS_SIZE, STATS_SIZE))
    return np.asarray(small, dtype=np.float32).mean(axis=(0, 1))


def border_anomaly_fraction(image) -> float:
    """Fracción de píxeles de borde brillantes y saturados (HSV S>0.8, V>0.5)."""
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    h, w = hsv.shape[:2]
    b = min(BORDER_PX, h // 2, w // 2)
    if b < 1:
        return 0.0
    mask = np.zeros((h, w), dtype=bool)
    mask[:b, :] = mask[-b:, :] = mask[:, :b] = mask[:, -b:] = True
    border = hsv[mask]
    anomalous = (border[:, 1] > 0.8) & (border[:, 2] > 0.5)
    return float(anomalous.mean())


def main() -> None:
    args = parse_args()
    print(f"Cargando dataset desde {args.dataset_dir}...")
    train = load_from_disk(args.dataset_dir)["train"]
    n = len(train) if args.max_examples is None else min(args.max_examples, len(train))
    print(f"Auditando {n} ejemplos...")

    deltas = []
    border_flagged = 0
    person_count = 0

    for i in range(n):
        ex = train[i]
        deltas.append(rgb_stats(ex["edited_image"]) - rgb_stats(ex["original_image"]))
        if border_anomaly_fraction(ex["edited_image"]) > BORDER_FLAG_FRACTION:
            border_flagged += 1
        if is_person_prompt(ex["edit_prompt"]):
            person_count += 1
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n}...")

    deltas = np.stack(deltas)
    mean_delta = deltas.mean(axis=0)
    # Pares que siguen el patrón "magenta" (R y B suben, G baja respecto al original)
    magenta_like = ((deltas[:, 0] > 2) & (deltas[:, 2] > 2) & (deltas[:, 1] < 0)).mean()

    report = {
        "examples_audited": n,
        "mean_rgb_delta_edited_minus_original": {
            "R": round(float(mean_delta[0]), 2),
            "G": round(float(mean_delta[1]), 2),
            "B": round(float(mean_delta[2]), 2),
        },
        "magenta_like_pairs_fraction": round(float(magenta_like), 4),
        "border_anomaly_images_fraction": round(border_flagged / n, 4),
        "person_prompts_fraction": round(person_count / n, 4),
        "person_prompts_count": person_count,
    }

    print("\n=== Auditoría del dataset ===")
    print(f"Ejemplos auditados:          {n}")
    print(
        f"ΔRGB medio (editada-original): R{mean_delta[0]:+.2f} "
        f"G{mean_delta[1]:+.2f} B{mean_delta[2]:+.2f} (0-255)"
    )
    print(f"Pares con patrón magenta:      {magenta_like:.2%}")
    print(f"Imágenes con borde anómalo:    {border_flagged / n:.2%} ({border_flagged})")
    print(f"Prompts con personas:          {person_count / n:.2%} ({person_count})")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nInforme guardado en {args.output}")


if __name__ == "__main__":
    main()
