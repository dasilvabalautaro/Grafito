"""Utilidades de carga y preprocesamiento de imágenes."""

from pathlib import Path

import requests
from PIL import Image, ImageOps


def load_image(path_or_url: str) -> Image.Image:
    """Carga una imagen desde ruta local o URL y la normaliza a RGB."""
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        response = requests.get(path_or_url, stream=True, timeout=30)
        response.raise_for_status()
        image = Image.open(response.raw)
    else:
        image = Image.open(Path(path_or_url))

    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def resize_for_model(image: Image.Image, max_size: int = 512) -> Image.Image:
    """Redimensiona manteniendo el aspect ratio para que el lado mayor sea max_size."""
    width, height = image.size
    scale = max_size / max(width, height)
    if scale >= 1.0:
        return image
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def trim_light_border(image: Image.Image, threshold: int = 250) -> Image.Image:
    """Recorta bordes muy claros (por ejemplo, marcos blancos)."""
    gray = image.convert("L")
    bbox = gray.point(lambda p: 255 if p < threshold else 0).getbbox()
    if bbox is None:
        return image
    return image.crop(bbox)
