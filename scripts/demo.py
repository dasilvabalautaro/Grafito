"""Demo web (Gradio) del editor Grafito.

Incluye las mitigaciones gratuitas del plan v3 (docs/TRAINING_V3_PLAN.md):
recorte de bordes contra manchas de esquina, plantillas de prompt con
atributos concretos y multi-variante para elegir la mejor seed.

Uso:
    python scripts/demo.py                        # local en http://127.0.0.1:7860
    python scripts/demo.py --share                # genera un link público temporal
    python scripts/demo.py --checkpoint RUTA      # usa otro checkpoint compatible con IP2P
"""

import argparse
import os
import random
import sys
import tempfile
from pathlib import Path

# Sin límite artificial de memoria MPS: con el límite por defecto, la segunda
# variante a 512 px hace OOM aunque la primera encaje (ver docs/TRAINING_V3_PLAN.md).
# Debe definirse antes de importar torch.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import gradio as gr
import torch

from grafito.config import EditConfig
from grafito.editor import edit_image, load_pipeline
from grafito.memory import get_device

DEFAULT_CHECKPOINT = "models/checkpoints/grafito-magicbrush-v2"
BASE_MODEL = "timbrooks/instruct-pix2pix"

# Parámetros de inferencia del checkpoint v2 (entrenado a 512 px; ver docs/TRAINING_V2_PLAN.md).
RESOLUTION = 512
DEFAULT_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 7.0
DEFAULT_IMAGE_GUIDANCE = 1.5
BORDER_CROP_PX = 8  # recorte anti-manchas de esquina (plan v3)

# Plantillas con atributos concretos: en v2 funcionan mejor que los prompts vagos.
PROMPT_EXAMPLES = [
    "add a black hat",
    "add a pair of sunglasses",
    "make the background light blue",
    "turn the shirt red",
    "make it look like a watercolor painting",
    "replace the coffee with a cup of tea",
]


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


def crop_borders(image, px: int):
    """Recorta px por lado para eliminar manchas de esquina esporádicas."""
    w, h = image.size
    if w > 2 * px and h > 2 * px:
        return image.crop((px, px, w - px, h - px))
    return image


def build_demo(pipe) -> gr.Blocks:
    """Construye la interfaz Gradio alrededor del pipeline ya cargado."""

    def edit(image, prompt, variants, border_crop, steps, guidance, image_guidance, seed):
        if image is None:
            raise gr.Error("Sube una imagen primero.")
        if not prompt or not prompt.strip():
            raise gr.Error("Escribe una instrucción de edición.")

        base_seed = random.randint(0, 2**31 - 1) if seed < 0 else int(seed)
        paths = []
        for k in range(int(variants)):
            config = EditConfig(
                resolution=RESOLUTION,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                image_guidance_scale=float(image_guidance),
                seed=base_seed + k,
            )
            edited = edit_image(pipe, image, prompt.strip(), config)
            if border_crop:
                edited = crop_borders(edited, BORDER_CROP_PX)
            out_path = Path(tempfile.mkdtemp(prefix="grafito_")) / f"edit_seed{base_seed + k}.png"
            edited.save(out_path)
            paths.append(str(out_path))
            if torch.backends.mps.is_available():
                # Sin esto, la segunda variante hace OOM por caché fragmentado en MPS.
                torch.mps.empty_cache()

        # Descarga por defecto: la primera variante; la selección en la galería la cambia.
        return image, paths, paths[0], paths

    def select_variant(paths, evt: gr.SelectData):
        return paths[evt.index]

    with gr.Blocks(title="Grafito — Demo") as demo:
        gr.Markdown("# Grafito — editor local de imágenes por instrucciones")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Imagen original", type="pil")
                prompt = gr.Textbox(
                    label="Instrucción de edición",
                    placeholder="add a black hat, make the background light blue, ...",
                )
                gr.Examples(
                    examples=[[p] for p in PROMPT_EXAMPLES],
                    inputs=prompt,
                    label="Plantillas (con atributos concretos funcionan mejor)",
                )
                with gr.Accordion("Parámetros", open=False):
                    variants = gr.Slider(1, 3, value=2, step=1, label="Variantes (seeds distintas)")
                    border_crop = gr.Checkbox(
                        value=True,
                        label=f"Recortar bordes ({BORDER_CROP_PX}px, anti-manchas de esquina)",
                    )
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
                    seed = gr.Number(value=-1, precision=0, label="Seed (-1 = aleatoria)")
                run_btn = gr.Button("Editar", variant="primary")
            with gr.Column():
                original_out = gr.Image(label="Original", interactive=False)
                gallery = gr.Gallery(
                    label="Variantes (clic para seleccionar)",
                    columns=3,
                    object_fit="contain",
                    show_download_button=True,
                )
                download = gr.DownloadButton(label="Descargar seleccionada")

        paths_state = gr.State([])
        run_btn.click(
            edit,
            inputs=[
                input_image,
                prompt,
                variants,
                border_crop,
                steps,
                guidance,
                image_guidance,
                seed,
            ],
            outputs=[original_out, gallery, download, paths_state],
        )
        gallery.select(select_variant, inputs=paths_state, outputs=download)

    return demo


def main() -> None:
    args = parse_args()

    device = get_device()
    model_id = resolve_model_id(args.checkpoint)
    print(f"Cargando {model_id} en {device}...")
    pipe = load_pipeline(model_id=model_id, device=device)
    if device != "cuda":
        # En MPS/CPU reduce el pico de memoria (Radeon 5500 XT: 8 GB VRAM).
        # Validado: 512 px en MPS ~31 s con los tres mecanismos activos.
        pipe.enable_attention_slicing()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)

    demo = build_demo(pipe)
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
