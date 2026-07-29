"""Auditoría de calidad del componente Instruct-CelebA dentro de la mezcla v3.

Puerta 0.5 de ``docs/TRAINING_V4_PLAN.md``: antes de gastar GPU en v4 hay que
verificar que los pares de Instruct-CelebA son buenos maestros. La mezcla
``magicbrush_v3`` conserva la procedencia: los ejemplos con ``face_id`` no
nulo provienen de Instruct-CelebA; los demás, de MagicBrush.

Chequeos por par (sobre una muestra estratificada por atributo):

1. **Emparejamiento original↔editada**: similitud coseno CLIP imagen-imagen.
   El pipeline asumió que el índice de ``v-xchen-v/celebamask_hq`` coincide
   con ``face_id``; si no, los pares enseñan a sustituir personas.
2. **Magnitud de la edición**: LPIPS(original, editada). Muy alto = reemplazo
   de escena, no edición.
3. **Adherencia al prompt**: similitud coseno CLIP texto-imagen.
4. **Presencia facial**: detector Haar en original y editada.

Como grupo de control se evalúa también una muestra de MagicBrush con las
mismas métricas, para juzgar los umbrales con referencia.

Además genera láminas de contacto (original | editada + prompt) por atributo
en ``--sheets_dir`` para revisión visual humana.

Uso:
    python src/scripts/audit_v3_data.py \
        --dataset_dir data/processed/magicbrush_v3 \
        --sample_per_attribute 100 \
        --output_json outputs/audit_v3/audit_v3.json
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from datasets import load_from_disk
from PIL import Image, ImageDraw

from scripts.prepare_instruct_celeba import has_face, load_face_detector

METRIC_SIZE = 256  # lado para las métricas (velocidad; suficiente para CLIP/LPIPS)
SHEET_IMAGE_SIZE = 224
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auditoría de calidad de Instruct-CelebA dentro de la mezcla v3."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/processed/magicbrush_v3",
        help="DatasetDict de la mezcla (con columnas face_id/attribute).",
    )
    parser.add_argument(
        "--sample_per_attribute",
        type=int,
        default=100,
        help="Pares de Instruct-CelebA a auditar por atributo.",
    )
    parser.add_argument(
        "--magicbrush_control",
        type=int,
        default=200,
        help="Tamaño de la muestra de control de MagicBrush.",
    )
    parser.add_argument(
        "--sheet_examples",
        type=int,
        default=12,
        help="Ejemplos por lámina de contacto (uno por fila).",
    )
    parser.add_argument(
        "--pair_sim_threshold",
        type=float,
        default=0.70,
        help="Bajo este valor de similitud CLIP img-img se sospecha mal emparejamiento.",
    )
    parser.add_argument(
        "--lpips_threshold",
        type=float,
        default=0.35,
        help="Sobre este valor LPIPS se sospecha reemplazo de escena.",
    )
    parser.add_argument(
        "--text_sim_threshold",
        type=float,
        default=0.18,
        help="Bajo este valor de similitud CLIP texto-img se sospecha mala adherencia.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_json",
        type=str,
        default="outputs/audit_v3/audit_v3.json",
        help="Informe JSON de salida.",
    )
    parser.add_argument(
        "--sheets_dir",
        type=str,
        default="outputs/audit_v3",
        help="Directorio para las láminas de contacto.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cpu"],
        help="Dispositivo para CLIP/LPIPS.",
    )
    return parser.parse_args()


def pick_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    return torch.device(choice)


def load_models(device: torch.device):
    import lpips
    from transformers import CLIPModel, CLIPProcessor

    print(f"Cargando CLIP ({CLIP_MODEL_ID}) y LPIPS en {device}...")
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device)
    return clip_model, clip_processor, lpips_model


def _prep(image: Image.Image) -> Image.Image:
    return image.convert("RGB").resize((METRIC_SIZE, METRIC_SIZE), Image.LANCZOS)


@torch.no_grad()
def pair_metrics(
    original: Image.Image,
    edited: Image.Image,
    prompt: str,
    clip_model,
    clip_processor,
    lpips_model,
    device: torch.device,
) -> dict[str, float]:
    """Métricas de un par: CLIP img-img, CLIP texto-img y LPIPS."""
    original = _prep(original)
    edited = _prep(edited)

    inputs = clip_processor(
        text=[prompt], images=[original, edited], return_tensors="pt", padding=True,
        truncation=True,
    ).to(device)
    text_feat = clip_model.get_text_features(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
    )
    img_feat = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

    pair_sim = float((img_feat[0] * img_feat[1]).sum())
    text_sim = float((text_feat[0] * img_feat[1]).sum())

    def to_lpips_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return (t * 2 - 1).to(device)

    lpips_val = float(lpips_model(to_lpips_tensor(original), to_lpips_tensor(edited)))

    return {"pair_sim": pair_sim, "text_sim": text_sim, "lpips": lpips_val}


def make_sheet(
    examples: list[dict[str, Any]],
    title: str,
    out_path: Path,
) -> None:
    """Lámina de contacto: una fila por ejemplo (original | editada + prompt)."""
    caption_h = 34
    row_h = SHEET_IMAGE_SIZE + caption_h
    width = SHEET_IMAGE_SIZE * 2 + 16
    height = row_h * len(examples) + caption_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill="black")

    for row, ex in enumerate(examples):
        y = caption_h + row * row_h
        orig = ex["original_image"].convert("RGB").resize(
            (SHEET_IMAGE_SIZE, SHEET_IMAGE_SIZE), Image.LANCZOS
        )
        edit = ex["edited_image"].convert("RGB").resize(
            (SHEET_IMAGE_SIZE, SHEET_IMAGE_SIZE), Image.LANCZOS
        )
        sheet.paste(orig, (0, y))
        sheet.paste(edit, (SHEET_IMAGE_SIZE + 16, y))
        prompt = textwrap.shorten(ex["edit_prompt"], width=90, placeholder="…")
        draw.text((8, y + SHEET_IMAGE_SIZE + 4), prompt, fill="black")

    sheet.save(out_path)


def summarize(records: list[dict[str, float]], args: argparse.Namespace) -> dict[str, Any]:
    """Agregados de un grupo de registros de métricas."""
    pair_sims = np.array([r["pair_sim"] for r in records])
    text_sims = np.array([r["text_sim"] for r in records])
    lpips_vals = np.array([r["lpips"] for r in records])
    face_lost = sum(1 for r in records if r["face_orig"] and not r["face_edited"])
    n = len(records)
    return {
        "n": n,
        "pair_sim_mean": round(float(pair_sims.mean()), 4),
        "pair_sim_p10": round(float(np.percentile(pair_sims, 10)), 4),
        "text_sim_mean": round(float(text_sims.mean()), 4),
        "lpips_mean": round(float(lpips_vals.mean()), 4),
        "lpips_p90": round(float(np.percentile(lpips_vals, 90)), 4),
        "flag_bad_pairing": int((pair_sims < args.pair_sim_threshold).sum()),
        "flag_scene_replacement": int((lpips_vals > args.lpips_threshold).sum()),
        "flag_weak_adherence": int((text_sims < args.text_sim_threshold).sum()),
        "flag_face_lost": face_lost,
        "flag_bad_pairing_frac": round(float((pair_sims < args.pair_sim_threshold).mean()), 4),
        "flag_scene_replacement_frac": round(float((lpips_vals > args.lpips_threshold).mean()), 4),
        "flag_weak_adherence_frac": round(float((text_sims < args.text_sim_threshold).mean()), 4),
        "flag_face_lost_frac": round(face_lost / n, 4),
    }


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    sheets_dir = Path(args.sheets_dir)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cargando dataset desde {args.dataset_dir}...")
    train = load_from_disk(args.dataset_dir)["train"]

    # Separar por procedencia: face_id no nulo -> Instruct-CelebA
    face_ids = train["face_id"]
    celeba_indices = [i for i, fid in enumerate(face_ids) if fid is not None]
    magicbrush_indices = [i for i, fid in enumerate(face_ids) if fid is None]
    print(f"Total: {len(train)} | Instruct-CelebA: {len(celeba_indices)} | "
          f"MagicBrush: {len(magicbrush_indices)}")

    rng = np.random.default_rng(args.seed)

    # Muestra estratificada de Instruct-CelebA por atributo
    attributes = train["attribute"]
    by_attribute: dict[str, list[int]] = defaultdict(list)
    for i in celeba_indices:
        by_attribute[attributes[i]].append(i)

    sample_indices: list[int] = []
    for attr, indices in sorted(by_attribute.items()):
        k = min(args.sample_per_attribute, len(indices))
        sample_indices.extend(rng.choice(indices, size=k, replace=False).tolist())

    # Muestra de control de MagicBrush
    k = min(args.magicbrush_control, len(magicbrush_indices))
    control_indices = rng.choice(magicbrush_indices, size=k, replace=False).tolist()

    clip_model, clip_processor, lpips_model = load_models(device)
    detector = load_face_detector()

    report: dict[str, Any] = {
        "dataset_dir": args.dataset_dir,
        "counts": {
            "total": len(train),
            "instruct_celeba": len(celeba_indices),
            "magicbrush": len(magicbrush_indices),
            "attributes": dict(Counter(attributes[i] for i in celeba_indices)),
        },
        "thresholds": {
            "pair_sim": args.pair_sim_threshold,
            "lpips": args.lpips_threshold,
            "text_sim": args.text_sim_threshold,
        },
        "per_attribute": {},
        "magicbrush_control": {},
    }

    def audit_group(indices: list[int], label: str) -> list[dict[str, float]]:
        records: list[dict[str, float]] = []
        for pos, i in enumerate(indices):
            ex = train[i]
            m = pair_metrics(
                ex["original_image"], ex["edited_image"], ex["edit_prompt"],
                clip_model, clip_processor, lpips_model, device,
            )
            m["face_orig"] = has_face(ex["original_image"], detector)
            m["face_edited"] = has_face(ex["edited_image"], detector)
            records.append(m)
            if (pos + 1) % 50 == 0:
                print(f"  {label}: {pos + 1}/{len(indices)}...")
        return records

    for attr, indices in sorted(by_attribute.items()):
        group = [i for i in sample_indices if attributes[i] == attr]
        if not group:
            continue
        print(f"Auditando atributo '{attr}' ({len(group)} pares)...")
        records = audit_group(group, attr)
        report["per_attribute"][attr] = summarize(records, args)

        sheet_examples = [train[i] for i in group[: args.sheet_examples]]
        make_sheet(
            sheet_examples,
            f"Instruct-CelebA — {attr} (izq: original, der: editada)",
            sheets_dir / f"sheet_{attr}.png",
        )

    print(f"Auditando control MagicBrush ({len(control_indices)} pares)...")
    control_records = audit_group(control_indices, "control")
    report["magicbrush_control"] = summarize(control_records, args)
    make_sheet(
        [train[i] for i in control_indices[: args.sheet_examples]],
        "Control MagicBrush (izq: original, der: editada)",
        sheets_dir / "sheet_control_magicbrush.png",
    )

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== Resumen por atributo ===")
    header = f"{'atributo':<12} {'n':>4} {'pairSim':>8} {'p10':>6} {'txtSim':>7} {'LPIPS':>6} {'malPar':>7} {'reemp':>6} {'adhBaja':>8}"
    print(header)
    for attr, s in sorted(report["per_attribute"].items()):
        print(
            f"{attr:<12} {s['n']:>4} {s['pair_sim_mean']:>8} {s['pair_sim_p10']:>6} "
            f"{s['text_sim_mean']:>7} {s['lpips_mean']:>6} "
            f"{s['flag_bad_pairing_frac']:>6.1%} {s['flag_scene_replacement_frac']:>6.1%} "
            f"{s['flag_weak_adherence_frac']:>7.1%}"
        )
    c = report["magicbrush_control"]
    print(
        f"{'CONTROL_MB':<12} {c['n']:>4} {c['pair_sim_mean']:>8} {c['pair_sim_p10']:>6} "
        f"{c['text_sim_mean']:>7} {c['lpips_mean']:>6} "
        f"{c['flag_bad_pairing_frac']:>6.1%} {c['flag_scene_replacement_frac']:>6.1%} "
        f"{c['flag_weak_adherence_frac']:>7.1%}"
    )
    print(f"\nInforme: {out_json}")
    print(f"Láminas: {sheets_dir}/sheet_*.png")


if __name__ == "__main__":
    main()
