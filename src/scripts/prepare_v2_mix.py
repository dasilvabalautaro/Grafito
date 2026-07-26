"""Construye la mezcla de entrenamiento v2 a partir del MagicBrush procesado.

La mezcla v2 = MagicBrush train completo + sesiones con personas duplicadas
(sobremuestreo x2), para corregir la degradación de caras observada en v1
(ver «Observaciones de calidad» en docs/NEXT_DEMO.md y docs/TRAINING_V2_PLAN.md).

El split validation se copia sin cambios para mantener la comparabilidad de
métricas con v1.

Uso:
    python src/scripts/prepare_v2_mix.py \
        --input_dir data/processed/magicbrush \
        --output_dir data/processed/magicbrush_v2
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

PERSON_KEYWORDS = [
    "man",
    "woman",
    "person",
    "people",
    "face",
    "boy",
    "girl",
    "child",
    "baby",
    "portrait",
    "selfie",
    "he",
    "she",
]

_PERSON_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in PERSON_KEYWORDS) + r")\b", re.IGNORECASE
)


def is_person_prompt(prompt: str) -> bool:
    """Devuelve True si el prompt menciona personas (con límites de palabra).

    Los límites de palabra evitan falsos positivos como "the" (contiene "he")
    o "facet" (contiene "face").
    """
    return bool(_PERSON_PATTERN.search(prompt))


def has_corner_calibration_strip(
    image,
    corner_px: int = 32,
    sat: float = 0.8,
    val: float = 0.5,
    min_pixels: int = 20,
    min_dispersion: float = 0.15,
) -> bool:
    """Detecta tiras de calibración de color en esquinas (auditoría v2).

    Algunas imágenes editadas de MagicBrush (escaneos de libros) traen un
    parche compacto multicolor muy saturado en una esquina; v1 aprendió a
    reproducirlo como manchas esporádicas en los bordes. El detector usa
    umbrales estrictos a propósito: alta precisión, recall parcial. Es un
    filtro de "casos seguros", no exhaustivo.
    """
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    h, w = hsv.shape[:2]
    c = min(corner_px, h // 2, w // 2)
    corners = [hsv[:c, :c], hsv[:c, -c:], hsv[-c:, :c], hsv[-c:, -c:]]
    for box in corners:
        mask = (box[..., 1] > sat) & (box[..., 2] > val)
        if mask.sum() < min_pixels:
            continue
        hues = box[..., 0][mask]
        ang = hues * 2 * np.pi
        dispersion = 1 - np.abs(np.exp(1j * ang).mean())
        if dispersion > min_dispersion:
            return True
    return False


def build_v2_mix(train: Dataset, repeat: int = 2) -> tuple[Dataset, dict]:
    """Duplica los ejemplos con personas dentro del split train.

    Args:
        train: Split de entrenamiento con columna ``edit_prompt``.
        repeat: Número total de veces que aparece cada ejemplo con personas
            (2 = aparece la original más una copia).

    Returns:
        Tupla (dataset mezclado, dict con estadísticas).
    """
    if repeat < 1:
        raise ValueError("repeat debe ser >= 1")

    person_indices = [
        i for i, prompt in enumerate(train["edit_prompt"]) if is_person_prompt(prompt)
    ]
    person_subset = train.select(person_indices)

    parts = [train] + [person_subset] * (repeat - 1)
    mixed = concatenate_datasets(parts)

    stats = {
        "train_original": len(train),
        "person_examples": len(person_subset),
        "person_fraction": len(person_subset) / len(train) if len(train) else 0.0,
        "repeat": repeat,
        "train_mixed": len(mixed),
    }
    return mixed, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construye la mezcla de datos v2.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/processed/magicbrush",
        help="DatasetDict procesado por prepare_magicbrush.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/magicbrush_v2",
        help="Directorio de salida para la mezcla v2.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="Veces que aparece cada ejemplo con personas (por defecto: 2).",
    )
    parser.add_argument(
        "--drop_corner_strips",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Filtra ejemplos con tira de calibración de color en esquinas "
            "(por defecto: activado; desactivar con --no-drop_corner_strips)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print(f"Cargando dataset desde {input_dir}...")
    ds = load_from_disk(str(input_dir))
    train = ds["train"]

    dropped_strips = 0
    if args.drop_corner_strips:
        print("Buscando tiras de calibración en esquinas...")
        strip_indices = [
            i for i in range(len(train)) if has_corner_calibration_strip(train[i]["edited_image"])
        ]
        dropped_strips = len(strip_indices)
        if strip_indices:
            keep = [i for i in range(len(train)) if i not in set(strip_indices)]
            train = train.select(keep)
        print(f"  {dropped_strips} ejemplos filtrados ({dropped_strips / len(ds['train']):.2%})")

    mixed_train, stats = build_v2_mix(train, repeat=args.repeat)
    stats["dropped_corner_strips"] = dropped_strips
    print(
        f"Personas: {stats['person_examples']}/{stats['train_original']} "
        f"({stats['person_fraction']:.1%}) x{stats['repeat']} -> "
        f"train final: {stats['train_mixed']} ejemplos"
    )

    # MagicBrush llama "dev" al split de validación; se normaliza a "validation".
    validation = ds["validation"] if "validation" in ds else ds["dev"]

    result = DatasetDict({"train": mixed_train, "validation": validation})
    output_dir.mkdir(parents=True, exist_ok=True)
    result.save_to_disk(str(output_dir))

    stats["validation"] = len(validation)
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Mezcla v2 guardada en {output_dir}")
    print(f"Estadísticas guardadas en {stats_path}")


if __name__ == "__main__":
    main()
