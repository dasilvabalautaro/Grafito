"""Tests para src/scripts/audit_v3_data.py."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.audit_v3_data import make_sheet, summarize


def _make_image(size=(64, 64), color=(128, 128, 128)):
    arr = np.full((*size, 3), color, dtype=np.uint8)
    return Image.fromarray(arr)


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        pair_sim_threshold=0.70,
        lpips_threshold=0.35,
        text_sim_threshold=0.18,
    )


def _record(pair_sim, text_sim, lpips, face_orig=True, face_edited=True):
    return {
        "pair_sim": pair_sim,
        "text_sim": text_sim,
        "lpips": lpips,
        "face_orig": face_orig,
        "face_edited": face_edited,
    }


def test_summarize_counts_flags():
    records = [
        _record(0.95, 0.25, 0.10),                     # bueno
        _record(0.50, 0.25, 0.10),                     # mal emparejamiento
        _record(0.95, 0.10, 0.10),                     # adherencia baja
        _record(0.95, 0.25, 0.50),                     # reemplazo de escena
        _record(0.95, 0.25, 0.10, face_edited=False),  # cara perdida
    ]
    s = summarize(records, _args())
    assert s["n"] == 5
    assert s["flag_bad_pairing"] == 1
    assert s["flag_weak_adherence"] == 1
    assert s["flag_scene_replacement"] == 1
    assert s["flag_face_lost"] == 1
    assert s["flag_bad_pairing_frac"] == 0.2


def test_summarize_no_faces_in_original_not_flagged():
    # Si la original no tiene cara, no cuenta como cara perdida.
    records = [_record(0.9, 0.2, 0.1, face_orig=False, face_edited=False)]
    s = summarize(records, _args())
    assert s["flag_face_lost"] == 0


def test_make_sheet_creates_png(tmp_path: Path):
    examples = [
        {
            "original_image": _make_image(color=(200, 100, 100)),
            "edited_image": _make_image(color=(100, 100, 200)),
            "edit_prompt": "add glasses",
        }
        for _ in range(3)
    ]
    out = tmp_path / "sheet_test.png"
    make_sheet(examples, "título de prueba", out)
    assert out.exists()
    sheet = Image.open(out)
    # 2 imágenes por fila + separación; 3 filas con caption + caption del título
    assert sheet.size[0] == 224 * 2 + 16
    assert sheet.size[1] == 34 + 3 * (224 + 34)
