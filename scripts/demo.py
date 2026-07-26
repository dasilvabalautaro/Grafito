"""Demo web (Gradio) del editor Grafito.

Uso:
    python scripts/demo.py                        # local en http://127.0.0.1:7860
    python scripts/demo.py --share                # genera un link público temporal
    python scripts/demo.py --checkpoint RUTA      # usa otro checkpoint compatible con IP2P
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import gradio as gr

from grafito.config import EditConfig
from grafito.editor import edit_image, load_pipeline
from grafito.memory import get_device

DEFAULT_CHECKPOINT = "models/checkpoints/grafito-magicbrush"
BASE_MODEL = "timbrooks/instruct-pix2pix"

# Parámetros de inferencia usados en la evaluación (ver docs/NEXT_DEMO.md).
RESOLUTION = 256
DEFAULT_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 7.0
DEFAULT_IMAGE_GUIDANCE = 1.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo web de Grafito (Gradio).")
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"Ruta al checkpoint entrenado (por defecto: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument(
        "--share", action="store_true", help="Genera un link público temporal de Gradio."
    )
    parser.add_argument("--port", type=int, default=7860, help="Puerto local (por defecto: 7860).")
    return parser.parse_args()


def resolve_model_id(checkpoint: str) -> str:
    """Devuelve el checkpoint si existe en disco; si no, el modelo base con un aviso."""
    path = Path(checkpoint)
    if path.exists():
        return str(path)
    print(f"[aviso] Checkpoint no encontrado en {path}; se usará el modelo base {BASE_MODEL}.")
    return BASE_MODEL


def build_demo(pipe) -> gr.Blocks:
    """Construye la interfaz Gradio alrededor del pipeline ya cargado."""

    def edit(image, prompt, steps, guidance_scale, image_guidance_scale, seed):
        if image is None:
            raise gr.Error("Sube una imagen primero.")
        if not prompt or not prompt.strip():
            raise gr.Error("Escribe una instrucción de edición.")

        config = EditConfig(
            resolution=RESOLUTION,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            image_guidance_scale=float(image_guidance_scale),
            seed=None if seed < 0 else int(seed),
        )
        edited = edit_image(pipe, image, prompt.strip(), config)

        out_path = Path(tempfile.mkdtemp(prefix="grafito_")) / "grafito_edit.png"
        edited.save(out_path)
        return image, edited, str(out_path)

    with gr.Blocks(title="Grafito — Demo") as demo:
        gr.Markdown("# Grafito — editor local de imágenes por instrucciones")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Imagen original", type="pil")
                prompt = gr.Textbox(
                    label="Instrucción de edición",
                    placeholder="add a hat, make the background blue, ...",
                )
                with gr.Accordion("Parámetros", open=False):
                    steps = gr.Slider(
                        5, 50, value=DEFAULT_STEPS, step=1, label="Pasos de inferencia"
                    )
                    guidance = gr.Slider(
                        1.0, 15.0, value=DEFAULT_GUIDANCE_SCALE, step=0.5, label="Guidance scale"
                    )
                    image_guidance = gr.Slider(
                        1.0,
                        3.0,
                        value=DEFAULT_IMAGE_GUIDANCE,
                        step=0.1,
                        label="Image guidance scale (fidelidad a la original)",
                    )
                    seed = gr.Number(value=42, precision=0, label="Seed (-1 = aleatoria)")
                run_btn = gr.Button("Editar", variant="primary")
            with gr.Column():
                original_out = gr.Image(label="Original", interactive=False)
                edited_out = gr.Image(label="Editada", interactive=False)
                download = gr.DownloadButton(label="Descargar resultado")

        run_btn.click(
            edit,
            inputs=[input_image, prompt, steps, guidance, image_guidance, seed],
            outputs=[original_out, edited_out, download],
        )

    return demo


def main() -> None:
    args = parse_args()

    device = get_device()
    model_id = resolve_model_id(args.checkpoint)
    print(f"Cargando {model_id} en {device}...")
    pipe = load_pipeline(model_id=model_id, device=device)
    if device != "cuda":
        # En MPS/CPU reduce el pico de memoria (Radeon 5500 XT: 8 GB VRAM).
        pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)

    demo = build_demo(pipe)
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
