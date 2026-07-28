"""Tests para src/scripts/prepare_v3_mix.py."""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset, DatasetDict
from PIL import Image

from scripts.prepare_v3_mix import (
    is_person_prompt,
    has_corner_calibration_strip,
)


def _make_image(size=(64, 64), color=(128, 128, 128)):
    arr = np.full((*size, 3), color, dtype=np.uint8)
    return Image.fromarray(arr)


def _make_strip_image(size=(64, 64)):
    """Crea una imagen con una tira multicolor saturada en la esquina superior izquierda."""
    arr = np.full((*size, 3), (128, 128, 128), dtype=np.uint8)
    colors = np.array(
        [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)], dtype=np.uint8
    )
    # Tile de 4 colores en la esquina 16x16 para que haya dispersión de tonos.
    for y in range(16):
        for x in range(16):
            arr[y, x] = colors[(y // 4 + x // 4) % 4]
    return Image.fromarray(arr)


def test_is_person_prompt():
    assert is_person_prompt("add a hat to the man") is True
    assert is_person_prompt("a woman smiling") is True
    assert is_person_prompt("make the sky blue") is False
    # "the" no debe activar "he"
    assert is_person_prompt("the sky") is False


def test_has_corner_calibration_strip_positive():
    img = _make_strip_image()
    assert has_corner_calibration_strip(img) is True


def test_has_corner_calibration_strip_negative():
    img = _make_image()
    assert has_corner_calibration_strip(img) is False


def _build_dummy_dataset(tmp_path: Path, n: int = 4, corner_index: int | None = None):
    """Crea un DatasetDict temporal con ``train`` y ``validation``."""
    records = []
    for i in range(n):
        color = (i * 50 + 50, i * 50 + 50, i * 50 + 50)
        img = _make_strip_image() if i == corner_index else _make_image(color=color)
        records.append(
            {
                "original_image": _make_image(color=(0, 0, 0)),
                "edited_image": img,
                "edit_prompt": "add a hat to the man" if i % 2 == 0 else "make the sky blue",
            }
        )

    ds = Dataset.from_list(records)
    ds_dict = DatasetDict({"train": ds, "validation": ds})
    path = tmp_path / "magicbrush"
    ds_dict.save_to_disk(str(path))
    return path


def _build_dummy_celeba(tmp_path: Path, n: int = 4):
    """Crea un DatasetDict temporal de Instruct-CelebA."""
    records = []
    for i in range(n):
        color = (200, 200, 200)
        records.append(
            {
                "original_image": _make_image(color=(0, 0, 0)),
                "edited_image": _make_image(color=color),
                "edit_prompt": "add glasses",
                "face_id": str(i),
                "attribute": "glasses",
            }
        )

    ds = Dataset.from_list(records)
    ds_dict = DatasetDict({"train": ds})
    path = tmp_path / "instruct_celeba"
    ds_dict.save_to_disk(str(path))
    return path


def test_prepare_v3_mix(tmp_path: Path):
    """Ejecuta prepare_v3_mix.py con datasets dummy y verifica la salida."""
    import subprocess

    mb_path = _build_dummy_dataset(tmp_path, n=4, corner_index=3)
    celeba_path = _build_dummy_celeba(tmp_path, n=2)
    output_path = tmp_path / "magicbrush_v3"

    result = subprocess.run(
        [
            "python",
            "src/scripts/prepare_v3_mix.py",
            "--magicbrush_dir",
            str(mb_path),
            "--instruct_celeba_dir",
            str(celeba_path),
            "--output_dir",
            str(output_path),
            "--person_repeat",
            "2",
            "--drop_corner_strips",
            "--no-face_filter",
            "--seed",
            "42",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    assert output_path.exists()

    # 4 ejemplos - 1 tira = 3; 2 personas x2 = 4, 1 no-persona = 1 -> 5; + 2 CelebA = 7
    ds = DatasetDict.load_from_disk(str(output_path))
    assert len(ds["train"]) == 7
    assert len(ds["validation"]) == 4

    stats = json.loads((output_path / "stats.json").read_text())
    assert stats["dropped_corner_strips"] == 1
    assert stats["person_examples"] == 2
    assert stats["person_repeat"] == 2
    assert stats["instruct_celeba"] == 2
    assert stats["train_final"] == 7


def test_prepare_v3_mix_no_corner_filter(tmp_path: Path):
    """Verifica que sin filtro de esquinas se conservan todos los ejemplos."""
    import subprocess

    mb_path = _build_dummy_dataset(tmp_path, n=4, corner_index=3)
    celeba_path = _build_dummy_celeba(tmp_path, n=2)
    output_path = tmp_path / "magicbrush_v3_nocorner"

    subprocess.run(
        [
            "python",
            "src/scripts/prepare_v3_mix.py",
            "--magicbrush_dir",
            str(mb_path),
            "--instruct_celeba_dir",
            str(celeba_path),
            "--output_dir",
            str(output_path),
            "--person_repeat",
            "1",
            "--no-drop_corner_strips",
            "--no-face_filter",
            "--seed",
            "42",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    # 4 MagicBrush + 2 CelebA = 6
    ds = DatasetDict.load_from_disk(str(output_path))
    assert len(ds["train"]) == 6

    stats = json.loads((output_path / "stats.json").read_text())
    assert stats["dropped_corner_strips"] == 0
    assert stats["person_examples"] == 2
