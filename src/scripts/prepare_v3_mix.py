"""Construye la mezcla de entrenamiento v3.

Combina:

1. MagicBrush filtrado (tiras de esquina + pares ruidosos ya eliminados).
2. Sobremuestreo ×2 de ejemplos con personas.
3. Filtro facial opcional.
4. Instruct-CelebA submuestreado.

El split ``validation`` se copia sin cambios desde MagicBrush para mantener la
comparabilidad con v1 y v2.

Ejemplo:
    python src/scripts/prepare_v3_mix.py \
        --magicbrush_dir data/processed/magicbrush_v3_prefilter \
        --instruct_celeba_dir data/processed/instruct_celeba \
        --output_dir data/processed/magicbrush_v3 \
        --person_repeat 2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

from scripts.prepare_instruct_celeba import has_face, load_face_detector


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construye la mezcla de datos v3.")
    parser.add_argument(
        "--magicbrush_dir",
        type=str,
        default="data/processed/magicbrush_v3_prefilter",
        help="DatasetDict de MagicBrush ya filtrado.",
    )
    parser.add_argument(
        "--instruct_celeba_dir",
        type=str,
        default="data/processed/instruct_celeba",
        help="DatasetDict de Instruct-CelebA procesado.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/magicbrush_v3",
        help="Directorio de salida para la mezcla v3.",
    )
    parser.add_argument(
        "--person_repeat",
        type=int,
        default=2,
        help="Veces que aparece cada ejemplo con personas en MagicBrush.",
    )
    parser.add_argument(
        "--drop_corner_strips",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filtrar tiras de calibración de color en esquinas.",
    )
    parser.add_argument(
        "--corner_px",
        type=int,
        default=32,
        help="Tamaño de la ventana de esquina para el detector de tiras.",
    )
    parser.add_argument(
        "--corner_sat",
        type=float,
        default=0.8,
        help="Umbral de saturación para considerar un píxel como colorido.",
    )
    parser.add_argument(
        "--corner_val",
        type=float,
        default=0.5,
        help="Umbral de valor para descartar esquinas oscuras.",
    )
    parser.add_argument(
        "--corner_min_pixels",
        type=int,
        default=20,
        help="Mínimo de píxeles saturados para considerar una tira.",
    )
    parser.add_argument(
        "--corner_min_dispersion",
        type=float,
        default=0.15,
        help="Mínima dispersión de tonos para considerar una tira multicolor.",
    )
    parser.add_argument(
        "--face_filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filtrar pares faciales deformados (requiere opencv-python).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para el shuffle final.",
    )
    return parser.parse_args()


def is_person_prompt(prompt: str) -> bool:
    """Devuelve True si el prompt menciona personas."""
    return bool(_PERSON_PATTERN.search(prompt))


def has_corner_calibration_strip(
    image,
    corner_px: int = 32,
    sat: float = 0.8,
    val: float = 0.5,
    min_pixels: int = 20,
    min_dispersion: float = 0.15,
) -> bool:
    """Detector de tiras de calibración de color en esquinas (heredado de v2).

    Usa umbrales estrictos a propósito: alta precisión, recall parcial. Es un
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


def apply_face_filter(dataset: Dataset, drop_attr_column: bool = True) -> Dataset:
    """Descarta ejemplos donde la cara desaparece entre original y editada.

    Si la imagen original no tiene cara detectable, se conserva el ejemplo
    (es un caso no facial). Solo se descarta cuando había cara en la original
    y no se detecta en la editada, indicando una deformación o borrado facial.
    """
    detector = load_face_detector()
    if detector is None:
        print("Advertencia: opencv-python no está instalado; se omite filtro facial.")
        return dataset

    keep_indices = []
    dropped = 0
    for i in range(len(dataset)):
        ex = dataset[i]
        orig_has_face = has_face(ex["original_image"], detector)
        edit_has_face = has_face(ex["edited_image"], detector)
        if orig_has_face and not edit_has_face:
            dropped += 1
        else:
            keep_indices.append(i)
    print(f"  Filtrados por calidad facial: {dropped}")
    return dataset.select(keep_indices)


def main() -> None:
    args = parse_args()
    magicbrush_dir = Path(args.magicbrush_dir)
    instruct_celeba_dir = Path(args.instruct_celeba_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Cargar MagicBrush
    print(f"Cargando MagicBrush desde {magicbrush_dir}...")
    mb_ds = load_from_disk(str(magicbrush_dir))
    mb_train = mb_ds["train"]
    validation = mb_ds["validation"] if "validation" in mb_ds else mb_ds["dev"]

    stats = {
        "magicbrush_train_input": len(mb_train),
        "dropped_corner_strips": 0,
        "person_examples": 0,
        "person_repeat": args.person_repeat,
        "face_filtered": 0,
        "instruct_celeba": 0,
    }

    # 1. Filtro de tiras de esquina
    if args.drop_corner_strips:
        print("Buscando tiras de calibración en esquinas...")
        strip_indices = [
            i
            for i in range(len(mb_train))
            if has_corner_calibration_strip(
                mb_train[i]["edited_image"],
                corner_px=args.corner_px,
                sat=args.corner_sat,
                val=args.corner_val,
                min_pixels=args.corner_min_pixels,
                min_dispersion=args.corner_min_dispersion,
            )
        ]
        stats["dropped_corner_strips"] = len(strip_indices)
        if strip_indices:
            keep = [i for i in range(len(mb_train)) if i not in set(strip_indices)]
            mb_train = mb_train.select(keep)
        print(f"  {stats['dropped_corner_strips']} tiras eliminadas")

    # 2. Filtro facial sobre MagicBrush
    if args.face_filter:
        print("Aplicando filtro de calidad facial a MagicBrush...")
        mb_train = apply_face_filter(mb_train)

    # 3. Sobremuestreo de personas
    person_indices = [
        i for i, prompt in enumerate(mb_train["edit_prompt"]) if is_person_prompt(prompt)
    ]
    stats["person_examples"] = len(person_indices)
    person_subset = mb_train.select(person_indices)
    mb_train = concatenate_datasets([mb_train] + [person_subset] * (args.person_repeat - 1))
    print(
        f"MagicBrush: {stats['person_examples']} personas x{args.person_repeat} "
        f"-> {len(mb_train)} ejemplos"
    )

    # 4. Cargar Instruct-CelebA
    print(f"Cargando Instruct-CelebA desde {instruct_celeba_dir}...")
    celeba_ds = load_from_disk(str(instruct_celeba_dir))
    celeba_train = celeba_ds["train"]
    stats["instruct_celeba"] = len(celeba_train)

    # 5. Combinar
    combined_train = concatenate_datasets([mb_train, celeba_train])
    # Mezclar para evitar sesgos de orden en el entrenamiento
    combined_train = combined_train.shuffle(seed=args.seed)

    result = DatasetDict({"train": combined_train, "validation": validation})
    result.save_to_disk(str(output_dir))

    stats["train_final"] = len(combined_train)
    stats["validation"] = len(validation)
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Mezcla v3 guardada en {output_dir}")
    print(f"  Train final: {stats['train_final']}")
    print(f"  Validation: {stats['validation']}")
    print(f"  Estadísticas: {stats_path}")


if __name__ == "__main__":
    main()
