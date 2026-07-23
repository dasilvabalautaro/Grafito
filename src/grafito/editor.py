"""Pipeline de edición con InstructPix2Pix."""

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image

from .config import EditConfig
from .image_utils import load_image, resize_for_model
from .memory import get_device, get_offload_strategy, get_torch_dtype


def load_pipeline(
    model_id: str | None = None,
    device: str | None = None,
    dtype=None,
) -> StableDiffusionInstructPix2PixPipeline:
    """Carga el pipeline IP2P en el dispositivo adecuado."""
    config = EditConfig()
    model_id = model_id or config.model_id
    device = device or get_device()
    dtype = dtype or get_torch_dtype(device)

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
    )

    offload = get_offload_strategy(device)
    if offload == "model_cpu_offload":
        pipe.enable_model_cpu_offload()
    elif offload == "sequential_cpu_offload":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = pipe.to(device)

    return pipe


def edit_image(
    pipe: StableDiffusionInstructPix2PixPipeline,
    image: Image.Image | str,
    prompt: str,
    config: EditConfig | None = None,
) -> Image.Image:
    """Edita una imagen siguiendo la instrucción dada."""
    config = config or EditConfig()

    if isinstance(image, str):
        image = load_image(image)

    image = resize_for_model(image, max_size=config.resolution)

    generator = None
    if config.seed is not None:
        device = get_device()
        generator = torch.Generator(device).manual_seed(config.seed)

    result = pipe(
        prompt,
        image=image,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        image_guidance_scale=config.image_guidance_scale,
        generator=generator,
    )
    return result.images[0]
