"""Descarga, empareja y submuestra Instruct-CelebA para el entrenamiento v3.

Instruct-CelebA (CoIE) aporta pares de edición faciales a 512×512. Este script:

1. Localiza los 3 archivos ``Instruct_CelebA_Dataset_2.zip.00[1-3]``.
2. Los une en un zip único y los extrae.
3. Recorre la estructura extraída buscando ``instruct.json``.
4. Empareja cada imagen editada con la imagen original de CelebA-HQ
   **por nombre de archivo** (``<face_id>.jpg``) desde un directorio local
   con ``CelebA-HQ-img/``. El emparejamiento por índice posicional sobre
   datasets de Hugging Face (p. ej. ``v-xchen-v/celebamask_hq``) se eliminó:
   la auditoría de 2026-07-29 demostró que produce pares de personas
   distintas (ver ``docs/TRAINING_V4_PLAN.md`` §0.5).
5. Aplica un filtro facial opcional (requiere ``opencv-python``).
6. Submuestrea estratificado según pesos por atributo.
7. Guarda el resultado como ``DatasetDict`` en ``data/processed/instruct_celeba``.

Ejemplo:
    python src/scripts/prepare_instruct_celeba.py \
        --zip_dir data/raw/instruct_celeba \
        --celebamask_hq_dir data/raw/celebamask_hq/CelebA-HQ-img \
        --output_dir data/processed/instruct_celeba \
        --max_samples 20000 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets
from PIL import Image


# Pesos por defecto para el submuestreo estratificado. Los atributos más
# cercanos a quitar/reemplazar objetos sobre la cara tienen prioridad.
DEFAULT_ATTRIBUTE_WEIGHTS: dict[str, int | None] = {
    "glasses": None,      # None = usar todos los ejemplos disponibles
    "beard": None,
    "eyes": None,
    "hair": 5000,
    "age": 3000,
    "gender": 2000,
    "expression": 2000,
    "skin": 1000,
    "anime": 0,           # dominio muy diferente; omitir por defecto
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara Instruct-CelebA para el entrenamiento v3 de Grafito."
    )
    parser.add_argument(
        "--zip_dir",
        type=str,
        default="data/raw/instruct_celeba",
        help="Directorio con los 3 archivos .zip.001, .zip.002 y .zip.003.",
    )
    parser.add_argument(
        "--celebamask_hq_dir",
        type=str,
        default="data/raw/celebamask_hq/CelebA-HQ-img",
        help=(
            "Directorio local con las imágenes originales de CelebA-HQ "
            "nombradas <face_id>.jpg (contenido de CelebA-HQ-img/). "
            "Obligatorio: el emparejamiento es por nombre de archivo."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/instruct_celeba",
        help="Directorio de salida para el DatasetDict procesado.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=20000,
        help="Número máximo de ejemplos finales (aproximado por estratos).",
    )
    parser.add_argument(
        "--attribute_weights",
        type=str,
        default=None,
        help=(
            "JSON con pesos por atributo para el submuestreo. "
            "None significa 'usar todos'; 0 significa 'omitir'. "
            "Ejemplo: '{\"glasses\":null,\"hair\":5000}'."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resolución objetivo para las imágenes originales de CelebA-HQ.",
    )
    parser.add_argument(
        "--face_filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Filtrar pares donde la cara no se detecte en la imagen editada "
            "(requiere opencv-python; si no está, se omite)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para el submuestreo estratificado.",
    )
    parser.add_argument(
        "--keep_extracted",
        action="store_true",
        help="No borrar el directorio extraído tras terminar (útil para depurar).",
    )
    return parser.parse_args()


def find_zip_parts(zip_dir: Path) -> list[Path]:
    """Localiza los 3 archivos de Instruct-CelebA en orden."""
    parts = []
    for i in ("001", "002", "003"):
        candidate = zip_dir / f"Instruct_CelebA_Dataset_2.zip.{i}"
        if not candidate.exists():
            raise FileNotFoundError(
                f"No se encontró {candidate}. Descarga los 3 archivos desde "
                "https://github.com/Junyi136/Instruct-Edit/releases/tag/vv1.0"
            )
        parts.append(candidate)
    return parts


def unsplit_and_extract(parts: list[Path], extract_dir: Path) -> None:
    """Une los archivos .zip.00x y extrae el contenido en ``extract_dir``."""
    zip_dir = parts[0].parent
    merged_zip = zip_dir / "Instruct_CelebA_Dataset_2_merged.zip"

    if not merged_zip.exists():
        print("Uniendo partes del zip...")
        with open(merged_zip, "wb") as out:
            for part in parts:
                with open(part, "rb") as src:
                    shutil.copyfileobj(src, out)
        print(f"  Zip unido: {merged_zip} ({merged_zip.stat().st_size / 1e9:.2f} GB)")

    print(f"Extrayendo en {extract_dir}...")
    extract_dir.mkdir(parents=True, exist_ok=True)

    # zipfile puede fallar con zips muy grandes o con estructuras especiales;
    # usamos unzip del sistema como primera opción.
    _extract_zip(merged_zip, extract_dir)

    # El zip unido contiene a su vez un zip interior (Instruct_CelebA_Dataset.zip).
    inner_zips = list(extract_dir.glob("*.zip"))
    for inner in inner_zips:
        print(f"Extrayendo zip interior {inner}...")
        _extract_zip(inner, extract_dir)
    print("  Extracción completada.")


def _extract_zip(zip_path: Path, extract_dir: Path) -> None:
    """Extrae un zip con ``unzip`` del sistema o zipfile como fallback."""
    if shutil.which("unzip"):
        subprocess.run(
            ["unzip", "-q", "-o", str(zip_path), "-d", str(extract_dir)],
            check=True,
        )
    else:
        import zipfile

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)


def parse_instruct_json(path: Path) -> list[dict[str, Any]]:
    """Lee un ``instruct.json`` y devuelve una lista de dicts normalizados.

    Acepta varios formatos posibles:
    - Dict {filename: instruction, ...}
    - Dict con claves como "instruction", "input", "output"
    - Lista de dicts con campos de instrucción
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results: list[dict[str, Any]] = []

    if isinstance(data, dict):
        # Formato {filename: instruction}
        for key, value in data.items():
            if isinstance(value, str):
                results.append({"edited_file": key, "prompt": value})
            elif isinstance(value, dict):
                results.append(_normalize_item(value, default_file=key))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                results.append({"edited_file": None, "prompt": item})
            elif isinstance(item, dict):
                results.append(_normalize_item(item))

    return results


def _normalize_item(item: dict, default_file: str | None = None) -> dict[str, Any]:
    """Normaliza un dict de instrucción a {edited_file, prompt}."""
    prompt = (
        item.get("instruction")
        or item.get("prompt")
        or item.get("edit_prompt")
        or item.get("text")
        or item.get("output")
    )
    edited_file = (
        item.get("edited_file")
        or item.get("output_file")
        or item.get("image")
        or item.get("file")
        or default_file
    )
    return {"edited_file": edited_file, "prompt": str(prompt).strip() if prompt else ""}


def _find_instruct_dataset_root(extract_dir: Path) -> Path:
    """Localiza la carpeta raíz que contiene ``train``/``test`` con ``instruct.json``."""
    # Caso común: extract_dir/directamente contiene instruct_dataset
    for candidate in (extract_dir / "instruct_dataset", extract_dir):
        if (candidate / "train").is_dir():
            return candidate
    # Caso con wrapper intermedio (ej. dataset/instruct_dataset)
    found = list(extract_dir.rglob("instruct_dataset/train"))
    if found:
        return found[0].parent
    return extract_dir


def discover_edit_pairs(extract_dir: Path) -> list[dict[str, Any]]:
    """Recorre el dataset extraído y devuelve todos los pares editados encontrados."""
    root = _find_instruct_dataset_root(extract_dir)
    pairs: list[dict[str, Any]] = []
    json_files = sorted(root.rglob("instruct.json"))
    print(f"Encontrados {len(json_files)} archivos instruct.json bajo {root}")

    for json_path in json_files:
        # Estructura: instruct_dataset/<split>/<attribute>/<face_id>/instruct.json
        rel_parts = json_path.relative_to(root).parts
        if len(rel_parts) < 3:
            continue
        attribute = rel_parts[-3]
        face_id = rel_parts[-2]

        instructions = parse_instruct_json(json_path)
        parent = json_path.parent

        for instr in instructions:
            prompt = instr.get("prompt", "")
            if not prompt:
                continue

            edited_file = instr.get("edited_file")
            edited_path = None
            if edited_file:
                # Instruct-CelebA usa <face_id>_<suffix>.jpg
                candidates = [
                    parent / edited_file,
                    parent / f"{edited_file}.jpg",
                    parent / f"{edited_file}.jpeg",
                    parent / f"{edited_file}.png",
                    parent / f"{face_id}_{edited_file}.jpg",
                    parent / f"{face_id}_{edited_file}.jpeg",
                    parent / f"{face_id}_{edited_file}.png",
                ]
                for candidate in candidates:
                    if candidate.exists():
                        edited_path = candidate
                        break
            else:
                # Si no se indica archivo, buscamos la única imagen editada en la carpeta
                images = sorted(
                    p for p in parent.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                )
                if len(images) == 1:
                    edited_path = images[0]
                elif len(images) > 1:
                    # Tomar la imagen cuyo nombre contenga el face_id y no sea "original"
                    for img in images:
                        if face_id in img.name and "original" not in img.name.lower():
                            edited_path = img
                            break
                    if edited_path is None:
                        edited_path = images[0]

            if edited_path is None:
                continue

            pairs.append(
                {
                    "face_id": face_id,
                    "attribute": attribute.lower(),
                    "prompt": prompt,
                    "edited_path": str(edited_path),
                    "json_path": str(json_path),
                }
            )

    print(f"  Pares editados descubiertos: {len(pairs)}")
    return pairs


def load_face_detector() -> Any | None:
    """Carga un detector facial de OpenCV si está disponible."""
    try:
        import cv2
    except ImportError:
        return None

    # Usar el clasificador Haar frontal por defecto (viene con opencv-python)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if Path(cascade_path).exists():
        return cv2.CascadeClassifier(cascade_path)
    return None


def has_face(image: Image.Image, detector: Any | None) -> bool:
    """Devuelve True si se detecta al menos una cara en la imagen."""
    if detector is None:
        return True
    import cv2

    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(32, 32))
    return len(faces) > 0


def _load_and_resize(path: Path, resolution: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if img.size != (resolution, resolution):
        img = img.resize((resolution, resolution), Image.LANCZOS)
    return img



def stratified_sample(
    pairs: list[dict[str, Any]],
    weights: dict[str, int | None],
    max_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Submuestra estratificado por atributo según los pesos configurados."""
    rng = np.random.default_rng(seed)

    by_attribute: dict[str, list[dict[str, Any]]] = {}
    for p in pairs:
        attr = p["attribute"]
        by_attribute.setdefault(attr, []).append(p)

    # Atributos desconocidos se mapean a un peso 0 a menos que aparezcan explícitamente.
    selected: list[dict[str, Any]] = []
    for attr, items in by_attribute.items():
        target = weights.get(attr, 0)
        if target == 0:
            continue
        if target is None or target >= len(items):
            chosen = items
        else:
            indices = rng.choice(len(items), size=target, replace=False)
            chosen = [items[i] for i in indices]
        selected.extend(chosen)

    # Si por los pesos explicitados superamos max_samples, hacemos un shuffle y truncado
    # conservando la proporción relativa entre estratos ya seleccionados.
    if len(selected) > max_samples:
        rng.shuffle(selected)
        selected = selected[:max_samples]

    return selected


def _load_original_from_dir(
    face_id: str,
    img_dir: Path,
    resolution: int,
) -> Image.Image | None:
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = img_dir / f"{face_id}{ext}"
        if candidate.exists():
            return _load_and_resize(candidate, resolution)
    return None


def _process_face_group(
    face_id: str,
    original: Image.Image,
    pairs: list[dict[str, Any]],
    detector: Any | None,
) -> tuple[list[dict[str, Any]], int]:
    """Procesa todos los pares de un ``face_id`` y devuelve (records, dropped_faces)."""
    records: list[dict[str, Any]] = []
    dropped = 0
    for item in pairs:
        edited = Image.open(item["edited_path"]).convert("RGB")
        if edited.size != original.size:
            edited = edited.resize(original.size, Image.LANCZOS)

        if detector is not None:
            if not has_face(original, detector) or not has_face(edited, detector):
                dropped += 1
                continue

        records.append(
            {
                "original_image": original,
                "edited_image": edited,
                "edit_prompt": item["prompt"],
                "face_id": face_id,
                "attribute": item["attribute"],
            }
        )
    return records, dropped


def _record_generator(
    selected: list[dict[str, Any]],
    celebamask_hq_dir: Path,
    resolution: int,
    face_filter: bool,
):
    """Generador que produce registros uno a uno sin mantener todos en memoria.

    Agrupa los pares por ``face_id``, carga el original una vez, produce los
    registros para ese face_id y lo descarta antes de continuar.
    """
    detector = load_face_detector() if face_filter else None
    if face_filter and detector is None:
        print("Advertencia: opencv-python no está instalado; se omite el filtro facial.")

    by_face_id: dict[str, list[dict[str, Any]]] = {}
    for item in selected:
        by_face_id.setdefault(item["face_id"], []).append(item)

    needed_ids = set(by_face_id.keys())
    print(f"Buscando {len(needed_ids)} imágenes originales de CelebA-HQ en {celebamask_hq_dir}...")

    if not celebamask_hq_dir.exists():
        raise FileNotFoundError(f"No existe {celebamask_hq_dir}")

    missing_originals = 0
    dropped_faces = 0
    processed = 0
    attributes: dict[str, int] = {}

    def process_one_face(face_id: str, original: Image.Image) -> None:
        nonlocal dropped_faces
        for item in by_face_id[face_id]:
            edited = Image.open(item["edited_path"]).convert("RGB")
            if edited.size != original.size:
                edited = edited.resize(original.size, Image.LANCZOS)

            if detector is not None:
                if not has_face(original, detector) or not has_face(edited, detector):
                    dropped_faces += 1
                    continue

            attr = item["attribute"]
            attributes[attr] = attributes.get(attr, 0) + 1
            yield {
                "original_image": original,
                "edited_image": edited,
                "edit_prompt": item["prompt"],
                "face_id": face_id,
                "attribute": attr,
            }

    for face_id in sorted(needed_ids, key=int):
        original = _load_original_from_dir(face_id, celebamask_hq_dir, resolution)
        if original is None:
            missing_originals += 1
            continue
        yield from process_one_face(face_id, original)
        processed += 1
        if processed % 500 == 0:
            print(f"  Procesados {processed}/{len(needed_ids)} originales")

    print(f"  Originales procesados: {processed}; faltantes: {missing_originals}")


def build_dataset_batched(
    selected: list[dict[str, Any]],
    celebamask_hq_dir: Path,
    resolution: int,
    face_filter: bool,
    chunk_size: int = 1000,
    max_missing_fraction: float = 0.01,
) -> tuple[Dataset, dict[str, Any]]:
    """Construye el dataset procesando los originales en lotes para evitar OOM.

    En lugar de ``Dataset.from_generator`` (que mostró parones de varios
    minutos al volcar shards), procesamos grupos faciales en lotes, guardamos
    cada lote en disco y concatenamos al final.

    Si falta más de ``max_missing_fraction`` de los originales necesarios se
    aborta: es señal de que el directorio de originales no corresponde a los
    ``face_id`` de Instruct-CelebA (el fallo que envenenó el dataset v3).
    """
    needed_ids = {item["face_id"] for item in selected}

    detector = load_face_detector() if face_filter else None
    if face_filter and detector is None:
        print("Advertencia: opencv-python no está instalado; se omite el filtro facial.")

    by_face_id: dict[str, list[dict[str, Any]]] = {}
    for item in selected:
        by_face_id.setdefault(item["face_id"], []).append(item)

    shard_dir = Path(tempfile.mkdtemp(prefix="instruct_celeba_shards_"))
    shard_paths: list[Path] = []
    chunk: list[dict[str, Any]] = []

    print(f"Procesando {len(needed_ids)} originales en lotes de {chunk_size}...")
    processed = 0
    missing = 0
    for face_id in sorted(needed_ids, key=int):
        original = _load_original_from_dir(face_id, celebamask_hq_dir, resolution)
        if original is None:
            missing += 1
            if missing <= 5:
                print(f"  Original no encontrado para face_id {face_id}")
            continue

        records, _ = _process_face_group(face_id, original, by_face_id[face_id], detector)
        chunk.extend(records)
        processed += 1

        if len(chunk) >= chunk_size:
            shard_path = shard_dir / f"shard_{len(shard_paths)}.arrow"
            Dataset.from_list(chunk).save_to_disk(shard_path)
            shard_paths.append(shard_path)
            print(f"  Guardado shard {len(shard_paths)} con {len(chunk)} ejemplos")
            chunk = []

        if processed % 500 == 0:
            print(f"  Procesados {processed}/{len(needed_ids)} originales")

    if chunk:
        shard_path = shard_dir / f"shard_{len(shard_paths)}.arrow"
        Dataset.from_list(chunk).save_to_disk(shard_path)
        shard_paths.append(shard_path)
        print(f"  Guardado shard final {len(shard_paths)} con {len(chunk)} ejemplos")

    missing_fraction = missing / len(needed_ids)
    print(f"  Originales procesados: {processed}; faltantes: {missing} ({missing_fraction:.2%})")
    if missing_fraction > max_missing_fraction:
        shutil.rmtree(shard_dir, ignore_errors=True)
        raise RuntimeError(
            f"Faltan {missing}/{len(needed_ids)} originales ({missing_fraction:.1%}). "
            "El directorio de CelebA-HQ no corresponde a los face_id de "
            "Instruct-CelebA. Revisa --celebamask_hq_dir."
        )

    if not shard_paths:
        raise RuntimeError("No se generó ningún registro. Revisa los paths de originales.")

    print(f"Concatenando {len(shard_paths)} shards...")
    dataset = concatenate_datasets([Dataset.load_from_disk(p) for p in shard_paths])

    print(f"Limpiando {len(shard_paths)} shards temporales...")
    shutil.rmtree(shard_dir, ignore_errors=True)

    stats = {
        "num_records": len(dataset),
        "num_originals_loaded": len(set(dataset["face_id"])),
        "missing_originals": missing,
        "attributes": dict(Counter(dataset["attribute"])),
    }

    return dataset, stats


def main() -> None:
    args = parse_args()
    zip_dir = Path(args.zip_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attribute_weights = DEFAULT_ATTRIBUTE_WEIGHTS.copy()
    if args.attribute_weights:
        attribute_weights.update(json.loads(args.attribute_weights))

    # 1. Localizar y extraer
    extract_dir = zip_dir / "extracted"
    try:
        root = _find_instruct_dataset_root(extract_dir)
        extraction_exists = (root / "train").is_dir()
    except FileNotFoundError:
        extraction_exists = False

    if not extraction_exists:
        parts = find_zip_parts(zip_dir)
        unsplit_and_extract(parts, extract_dir)
    else:
        print(f"Usando extracción existente en {extract_dir}")

    # 2. Descubrir pares editados
    pairs = discover_edit_pairs(extract_dir)
    if not pairs:
        raise RuntimeError(
            "No se encontraron pares editados. Revisa la estructura extraída."
        )

    # 3. Submuestreo estratificado (antes de cargar originales para ahorrar I/O)
    selected = stratified_sample(
        pairs, attribute_weights, args.max_samples, args.seed
    )
    print(f"Ejemplos seleccionados tras submuestreo: {len(selected)}")

    # 4. Construir dataset cargando originales en lotes (evita OOM)
    celebamask_hq_dir = Path(args.celebamask_hq_dir)
    train_ds, stats = build_dataset_batched(
        selected,
        celebamask_hq_dir,
        args.resolution,
        face_filter=args.face_filter,
    )

    # 5. Guardar
    dataset_dict = DatasetDict({"train": train_ds})
    dataset_dict.save_to_disk(output_dir)

    stats["output_dir"] = str(output_dir)
    stats["num_selected"] = len(selected)
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Dataset guardado en {output_dir}")
    print(f"  Ejemplos finales: {len(train_ds)}")
    print(f"  Estadísticas: {stats_path}")

    if not args.keep_extracted:
        print(f"Borrando extracción intermedia {extract_dir} para ahorrar espacio...")
        shutil.rmtree(extract_dir, ignore_errors=True)
        merged_zip = zip_dir / "Instruct_CelebA_Dataset_2_merged.zip"
        if merged_zip.exists():
            print(f"Borrando zip unido {merged_zip}...")
            merged_zip.unlink()


if __name__ == "__main__":
    main()
