"""Evaluación básica de un checkpoint de InstructPix2Pix."""

from pathlib import Path

import click


@click.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--dataset", default="data/processed/magicbrush/dev", type=click.Path(exists=True))
@click.option("--output", default="outputs/eval_report.json", type=click.Path())
def main(checkpoint: str, dataset: str, output: str):
    """Genera métricas de evaluación comparando el checkpoint con la línea base."""
    checkpoint_path = Path(checkpoint)
    dataset_path = Path(dataset)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # TODO: implementar cálculo de métricas (CLIP directional, SSIM, LPIPS).
    raise NotImplementedError("La evaluación cuantitativa se implementará en el hito H4.")


if __name__ == "__main__":
    main()
