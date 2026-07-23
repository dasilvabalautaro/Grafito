"""Interfaz de línea de comandos para Grafito."""

import click

from grafito.config import EditConfig
from grafito.editor import edit_image, load_pipeline


@click.command()
@click.option("--image", required=True, type=click.Path(exists=True), help="Imagen de entrada.")
@click.option("--prompt", required=True, help="Instrucción de edición.")
@click.option("--output", required=True, type=click.Path(), help="Ruta de la imagen de salida.")
@click.option("--model-id", default=None, help="ID del modelo en Hugging Face.")
@click.option("--resolution", default=512, type=int, help="Tamaño máximo del lado mayor.")
@click.option("--steps", default=20, type=int, help="Pasos de inferencia.")
@click.option("--guidance-scale", default=7.5, type=float)
@click.option("--image-guidance-scale", default=1.5, type=float)
@click.option("--seed", default=None, type=int)
def main(
    image: str,
    prompt: str,
    output: str,
    model_id: str | None,
    resolution: int,
    steps: int,
    guidance_scale: float,
    image_guidance_scale: float,
    seed: int | None,
):
    """Edita una imagen usando InstructPix2Pix."""
    config = EditConfig(
        model_id=model_id or EditConfig.model_id,
        resolution=resolution,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        seed=seed,
    )

    pipe = load_pipeline(model_id=config.model_id)
    edited = edit_image(pipe, image, prompt, config=config)
    edited.save(output)
    click.echo(f"Imagen guardada en: {output}")


if __name__ == "__main__":
    main()
