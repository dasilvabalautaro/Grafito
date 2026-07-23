"""Tests para src/scripts/prepare_magicbrush.py."""

from PIL import Image

from scripts.prepare_magicbrush import preprocess_and_validate, resize_image


def test_resize_image_keeps_aspect_ratio_and_max_size():
    image = Image.new("RGB", (1024, 512), color=(128, 128, 128))
    resized = resize_image(image, max_size=256)
    assert resized.size == (256, 128)


def test_resize_image_does_not_upsample():
    image = Image.new("RGB", (200, 100), color=(128, 128, 128))
    resized = resize_image(image, max_size=512)
    assert resized.size == (200, 100)


def test_resize_image_converts_rgba_to_rgb():
    image = Image.new("RGBA", (256, 256), color=(255, 0, 0, 128))
    resized = resize_image(image, max_size=128)
    assert resized.mode == "RGB"


def test_preprocess_and_validate_keeps_valid_example():
    original = Image.new("RGB", (256, 256), color=(0, 0, 0))
    edited = Image.new("RGB", (256, 256), color=(255, 255, 255))
    example = {
        "original_image": original,
        "edited_image": edited,
        "edit_prompt": "  Add a Hat  ",
    }
    result = preprocess_and_validate(example, max_size=128)
    assert result["_drop"] is False
    assert result["edit_prompt"] == "Add a Hat"
    assert result["original_image"].size == (128, 128)


def test_preprocess_and_validate_drops_empty_prompt():
    original = Image.new("RGB", (256, 256), color=(0, 0, 0))
    edited = Image.new("RGB", (256, 256), color=(255, 255, 255))
    example = {
        "original_image": original,
        "edited_image": edited,
        "edit_prompt": "   ",
    }
    result = preprocess_and_validate(example, max_size=128)
    assert result["_drop"] is True


def test_preprocess_and_validate_drops_missing_image():
    example = {
        "original_image": Image.new("RGB", (64, 64)),
        "edited_image": None,
        "edit_prompt": "make it red",
    }
    result = preprocess_and_validate(example, max_size=128)
    assert result["_drop"] is True
