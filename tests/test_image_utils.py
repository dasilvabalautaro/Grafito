"""Tests para image_utils."""

from PIL import Image

from grafito.image_utils import resize_for_model, trim_light_border


def test_resize_for_model_keeps_aspect_ratio():
    image = Image.new("RGB", (1024, 512), color=(128, 128, 128))
    resized = resize_for_model(image, max_size=512)
    assert resized.size == (512, 256)


def test_resize_for_model_does_not_upsample():
    image = Image.new("RGB", (400, 300), color=(128, 128, 128))
    resized = resize_for_model(image, max_size=512)
    assert resized.size == (400, 300)


def test_trim_light_border_crops_white_frame():
    image = Image.new("RGB", (200, 200), color=(255, 255, 255))
    # Dibuja un cuadrado oscuro en el centro
    for x in range(50, 150):
        for y in range(50, 150):
            image.putpixel((x, y), (0, 0, 0))

    trimmed = trim_light_border(image, threshold=250)
    assert trimmed.size == (100, 100)
