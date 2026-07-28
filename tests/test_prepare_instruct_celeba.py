"""Tests para src/scripts/prepare_instruct_celeba.py."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.prepare_instruct_celeba import (
    _find_instruct_dataset_root,
    discover_edit_pairs,
    parse_instruct_json,
    stratified_sample,
)


def _make_image(size=(64, 64), color=(128, 128, 128)) -> Image.Image:
    arr = np.full((*size, 3), color, dtype=np.uint8)
    return Image.fromarray(arr)


def _write_fake_extracted(base: Path) -> Path:
    """Crea una estructura Instruct-CelebA mínima para tests."""
    root = base / "dataset" / "instruct_dataset"
    train = root / "train" / "glasses"
    face_dir = train / "42"
    face_dir.mkdir(parents=True)

    _make_image().save(face_dir / "42_addglasses.jpg")
    _make_image().save(face_dir / "42_removeglasses.jpg")
    with open(face_dir / "instruct.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "addglasses": "Make the man wearing glasses.",
                "removeglasses": "Give the man a glasses-free look.",
            },
            f,
        )
    return base


def test_find_instruct_dataset_root_with_wrapper(tmp_path: Path):
    root = tmp_path / "dataset" / "instruct_dataset"
    (root / "train" / "glasses" / "1").mkdir(parents=True)
    assert _find_instruct_dataset_root(tmp_path) == root


def test_find_instruct_dataset_root_direct(tmp_path: Path):
    root = tmp_path / "instruct_dataset"
    (root / "train" / "glasses" / "1").mkdir(parents=True)
    assert _find_instruct_dataset_root(tmp_path) == root


def test_parse_instruct_json(tmp_path: Path):
    data = {"addhat": "Add a hat.", "removehat": "Remove the hat."}
    json_path = tmp_path / "instruct.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    parsed = parse_instruct_json(json_path)
    assert len(parsed) == 2
    assert parsed[0]["edited_file"] == "addhat"
    assert parsed[0]["prompt"] == "Add a hat."


def test_discover_edit_pairs(tmp_path: Path):
    base = _write_fake_extracted(tmp_path)
    pairs = discover_edit_pairs(base)
    assert len(pairs) == 2
    assert all(p["face_id"] == "42" for p in pairs)
    assert all(p["attribute"] == "glasses" for p in pairs)
    assert any("wearing glasses" in p["prompt"] for p in pairs)
    assert all(Path(p["edited_path"]).exists() for p in pairs)


def test_stratified_sample():
    pairs = [
        {"face_id": "1", "attribute": "glasses", "prompt": "p1", "edited_path": "a"},
        {"face_id": "2", "attribute": "glasses", "prompt": "p2", "edited_path": "b"},
        {"face_id": "3", "attribute": "hair", "prompt": "p3", "edited_path": "c"},
        {"face_id": "4", "attribute": "hair", "prompt": "p4", "edited_path": "d"},
        {"face_id": "5", "attribute": "anime", "prompt": "p5", "edited_path": "e"},
    ]
    weights = {"glasses": None, "hair": 1, "anime": 0}
    selected = stratified_sample(pairs, weights, max_samples=10, seed=0)
    assert len(selected) == 3  # 2 glasses + 1 hair
    attrs = {p["attribute"] for p in selected}
    assert attrs == {"glasses", "hair"}
