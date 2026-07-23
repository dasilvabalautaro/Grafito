"""Script temporal para validar la línea base de IP2P en este hardware."""

import time
from pathlib import Path

from diffusers.utils import load_image

from grafito.config import EditConfig
from grafito.editor import edit_image, load_pipeline


def main():
    print("Cargando pipeline...")
    t0 = time.time()
    pipe = load_pipeline()
    print(f"Pipeline cargado en {time.time() - t0:.1f}s")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
    print(f"Descargando imagen de ejemplo...")
    image = load_image(url).convert("RGB")
    print(f"Imagen: {image.size}")

    config = EditConfig(
        resolution=256,
        num_inference_steps=10,
        guidance_scale=7.5,
        image_guidance_scale=1.5,
        seed=42,
    )

    prompt = "turn him into a cyborg"
    print(f"Ejecutando inferencia: '{prompt}'")
    t0 = time.time()
    edited = edit_image(pipe, image, prompt, config=config)
    elapsed = time.time() - t0
    print(f"Inferencia completada en {elapsed:.1f}s")

    output_path = output_dir / "baseline_edit.jpg"
    edited.save(output_path)
    print(f"Guardado en: {output_path}")


if __name__ == "__main__":
    main()
