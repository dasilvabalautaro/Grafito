"""Tests para src/scripts/prepare_v2_mix.py."""

import numpy as np
import pytest
from datasets import Dataset
from PIL import Image

from scripts.prepare_v2_mix import build_v2_mix, has_corner_calibration_strip, is_person_prompt


def test_is_person_prompt_matches_keywords():
    assert is_person_prompt("make the man wear a hat")
    assert is_person_prompt("add glasses to her face")
    assert is_person_prompt("give the baby a toy")


def test_is_person_prompt_avoids_substring_false_positives():
    # "the" contiene "he" y "facet" contiene "face": no deben marcar.
    assert not is_person_prompt("make the sky blue")
    assert not is_person_prompt("highlight the facet of the gem")
    assert not is_person_prompt("add a hat to the statue")


def _fake_train():
    return Dataset.from_dict(
        {
            "edit_prompt": [
                "make the man wear a hat",
                "make the sky blue",
                "give the girl a balloon",
                "turn the car red",
            ],
            "id": [0, 1, 2, 3],
        }
    )


def test_build_v2_mix_duplicates_person_examples():
    mixed, stats = build_v2_mix(_fake_train(), repeat=2)
    assert stats["person_examples"] == 2
    assert stats["train_original"] == 4
    assert stats["train_mixed"] == 6
    prompts = mixed["edit_prompt"]
    assert prompts.count("make the man wear a hat") == 2
    assert prompts.count("make the sky blue") == 1


def test_build_v2_mix_rejects_repeat_below_one():
    with pytest.raises(ValueError):
        build_v2_mix(_fake_train(), repeat=0)


def _image_with_corner_strip() -> Image.Image:
    # Parche multicolor saturado en la esquina inferior derecha (tira de calibración).
    arr = np.full((64, 64, 3), 128, dtype=np.uint8)
    colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0)]
    for k, color in enumerate(colors):
        arr[-12:, -16 + k * 4 : -16 + (k + 1) * 4] = color
    return Image.fromarray(arr)


def test_has_corner_calibration_strip_detects_strip():
    assert has_corner_calibration_strip(_image_with_corner_strip())


def test_has_corner_calibration_strip_ignores_uniform_image():
    assert not has_corner_calibration_strip(Image.new("RGB", (64, 64), (200, 30, 30)))
