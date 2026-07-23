"""Configuración por defecto del editor."""

from dataclasses import dataclass


@dataclass
class EditConfig:
    """Parámetros por defecto para la edición con InstructPix2Pix."""

    model_id: str = "timbrooks/instruct-pix2pix"
    resolution: int = 512
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    image_guidance_scale: float = 1.5
    seed: int | None = None
    dtype: str = "float16"
