"""Descarga y formatea MagicBrush para entrenamiento con Diffusers.

El script oficial de entrenamiento (`train_instruct_pix2pix.py`) espera un dataset
con las columnas ``original_image``, ``edited_image`` y ``edit_prompt``. Este script
prepara MagicBrush en ese formato, filtra ejemplos corruptos o con prompts vacíos,
redimensiona los lados grandes al tamaño objetivo y guarda los splits en disco en
formato Arrow como un ``DatasetDict`` (compatible con ``load_dataset(path)``
del script oficial de entrenamiento).
"""

import argparse
import json
from pathlib import Path

from datasets import DatasetDict, load_dataset
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara MagicBrush para entrenamiento de InstructPix2Pix."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/magicbrush",
        help="Directorio donde se guardarán los splits procesados.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Tamaño máximo del lado mayor antes del crop/resize del entrenamiento.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directorio de caché para datasets descargados.",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="data/raw/magicbrush",
        help=(
            "Directorio local con archivos parquet de MagicBrush. "
            "Si existe, se usa antes de descargar desde HuggingFace."
        ),
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help=(
            "Número de procesos paralelos para `datasets.map`. "
            "Por defecto usa 1 (útil en macOS para evitar problemas con multiprocessing)."
        ),
    )
    return parser.parse_args()


def resize_image(image: Image.Image, max_size: int) -> Image.Image:
    """Redimensiona una imagen manteniendo aspect ratio si supera ``max_size``.

    Args:
        image: Imagen PIL en cualquier modo.
        max_size: Tamaño máximo permitido para el lado mayor.

    Returns:
        Imagen RGB redimensionada.
    """
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


def preprocess_and_validate(example: dict, max_size: int) -> dict:
    """Valida un ejemplo, redimensiona imágenes y normaliza el prompt.

    Los ejemplos inválidos se marcan con ``_drop=True`` para filtrarlos después.
    """
    result = example.copy()
    prompt = str(result.get("edit_prompt", "")).strip()
    if not prompt:
        result["_drop"] = True
        return result

    for key in ("original_image", "edited_image"):
        image = result.get(key)
        if image is None:
            result["_drop"] = True
            return result
        try:
            image = image.convert("RGB")
            image.load()
            result[key] = resize_image(image, max_size)
        except Exception:
            result["_drop"] = True
            return result

    if "mask_img" in result:
        try:
            result["mask_img"] = resize_image(result["mask_img"], max_size)
        except Exception:
            # La máscara es opcional para entrenamiento; si falla, la dejamos como None.
            result["mask_img"] = None

    result["edit_prompt"] = prompt
    result["_drop"] = False
    return result


def load_magicbrush(local_dir: Path | None, cache_dir: str | None) -> dict:
    """Carga MagicBrush desde HF o desde un directorio local de parquet."""
    if local_dir is not None and (local_dir / "data").exists():
        data_dir = local_dir / "data"
        parquet_files = list(data_dir.glob("*.parquet"))
        if parquet_files:
            print(f"Cargando MagicBrush desde archivos locales en {data_dir}...")
            ds = load_dataset("parquet", data_dir=str(data_dir), cache_dir=cache_dir)
            return ds

    print("Descargando MagicBrush (osunlp/MagicBrush)...")
    return load_dataset("osunlp/MagicBrush", cache_dir=cache_dir)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_dir = Path(args.local_dir) if args.local_dir else None
    ds = load_magicbrush(local_dir, args.cache_dir)

    column_mapping = {
        "source_img": "original_image",
        "target_img": "edited_image",
        "instruction": "edit_prompt",
    }

    def rename_columns(split):
        available = {k: v for k, v in column_mapping.items() if k in split.column_names}
        return split.rename_columns(available)

    ds = {name: rename_columns(split) for name, split in ds.items()}

    processed = {}
    stats = {}
    for name, split in ds.items():
        print(f"\nProcesando split '{name}' ({len(split)} ejemplos crudos)...")
        split = split.map(
            lambda ex: preprocess_and_validate(ex, args.resolution),
            batched=False,
            desc=f"Preprocesando {name}",
            num_proc=args.num_proc,
        )
        split = split.filter(
            lambda ex: not ex["_drop"],
            batched=False,
            desc=f"Filtrando {name}",
            num_proc=args.num_proc,
        )
        split = split.remove_columns(["_drop"])

        processed[name] = split
        stats[name] = {
            "num_examples": len(split),
            "columns": split.column_names,
            "resolution": args.resolution,
        }
        print(f"  {name}: {len(split)} ejemplos válidos")

    dataset_dict = DatasetDict(processed)
    dataset_dict.save_to_disk(output_dir)
    print(f"\nDataset guardado en {output_dir}")

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Estadísticas guardadas en {stats_path}")
    print("Listo.")


if __name__ == "__main__":
    main()
