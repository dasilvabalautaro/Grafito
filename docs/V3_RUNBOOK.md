# Runbook v3 — del dataset v3 al modelo descargado

Procedimiento operativo del entrenamiento v3, tarea por tarea, con comandos
exactos y criterio de éxito. El porqué de cada decisión está en
`docs/TRAINING_V3_PLAN.md`; este documento es solo el paso a paso.

Leyenda: `[x]` hecha · `[ ]` pendiente. `<IP>`/`<PUERTO>` = los que asigne
Vast.ai a la instancia. `<REPO>` = URL del repo (GitHub).

Dataset v3 ya generado en local: `data/processed/magicbrush_v3/`
- Train: 27.098 ejemplos
- Validation: 528 ejemplos
- Peso: ~19 GB
- Composición: MagicBrush filtrado (7.645 ej., personas ×2) + Instruct-CelebA (19.453 ej.)

---

## Fase A — Preparación local (ya hecha, $0)

- [x] **A1. Dataset MagicBrush a 512 px**
  `python src/scripts/prepare_magicbrush.py --resolution 512 --num_proc 4`
  Éxito: `data/processed/magicbrush/stats.json` con train 8807 / validation 528.

- [x] **A2. Filtro de pares ruidosos sobre MagicBrush**
  `python src/scripts/filter_noisy_pairs.py --input_dir data/processed/magicbrush --output_dir data/processed/magicbrush_v3_prefilter --device auto --batch_size 8`
  Éxito: `data/processed/magicbrush_v3_prefilter/stats.json` con ~7.400 ejemplos.

- [x] **A3. Instruct-CelebA descargado y procesado**
  Requiere los 3 archivos `.zip.00x` en `data/raw/instruct_celeba/`.
  `python src/scripts/prepare_instruct_celeba.py --zip_dir data/raw/instruct_celeba --celebamask_hq_dataset v-xchen-v/celebamask_hq --output_dir data/processed/instruct_celeba --max_samples 20000 --seed 42 --keep_extracted`
  Éxito: `data/processed/instruct_celeba/stats.json` con ~19.000–20.000 ejemplos.
  **⚠️ Corrección retroactiva (2026-07-29): este comando produjo pares rotos.**
  El emparejamiento por índice posicional de `v-xchen-v/celebamask_hq` no
  coincide con `face_id`; la auditoría de v4 (`outputs/audit_v3/`) demostró
  que ~95 % de los pares son personas distintas. El comando correcto usa
  `--celebamask_hq_dir` con `CelebA-HQ-img/` real (ver `docs/V4_RUNBOOK.md`
  A4.1–A4.3).

- [x] **A4. Mix v3**
  `python src/scripts/prepare_v3_mix.py --magicbrush_dir data/processed/magicbrush_v3_prefilter --instruct_celeba_dir data/processed/instruct_celeba --output_dir data/processed/magicbrush_v3 --person_repeat 2 --seed 42`
  Éxito: `data/processed/magicbrush_v3/stats.json` con train ~27.000 / validation 528.

- [x] **A5. Tests y verificación local**
  ```bash
  pytest tests/ -q
  python -c "from datasets import load_from_disk; d=load_from_disk('data/processed/magicbrush_v3'); print(d)"
  cat data/processed/magicbrush_v3/stats.json
  ```
  Éxito: 29 tests verdes, dataset carga, stats coherente.

- [x] **A6. Liberar espacio local (intermedios)**
  Comando usado:
  ```bash
  rm -rf data/processed/magicbrush_512 \
         data/processed/magicbrush_v3_prefilter \
         data/processed/instruct_celeba \
         data/raw/instruct_celeba/extracted
  ```
  Éxito: solo queda `data/processed/magicbrush_v3/` (~19 GB) como dataset para entrenar.

- [ ] **A7. Commit y push del repo** (código, tests y docs; sin datos ni pesos).
  Éxito: `git status` limpio; el pod podrá clonar el repo actualizado.

---

## Fase B — Pod de Vast.ai (reloj corriendo: minimizar horas)

- [ ] **B1. Elegir oferta**
  GPU: RTX 3090 / 4090 / A10 / A100 (≥ 24 GB). Evitar RTX 50xx/Blackwell.
  Driver CUDA del host ≥ 12.1. Disco local ≥ 100 GB. On-demand (no spot).
  Imagen: `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`.
  **Sin Network Volume.** Alarma de gasto en $20.

- [ ] **B2. Clonar e instalar** (en el pod)
  ```bash
  git clone <REPO> /workspace/Grafito && cd /workspace/Grafito
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```
  Éxito: instalación sin errores (torch 2.2.2+cu121 ya viene en la imagen y
  no debe reinstalarse).

- [ ] **B3. Pre-flight (obligatorio; si falla, destruir la instancia al momento)**
  ```bash
  python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
  # debe imprimir exactamente: 2.2.2 12.1 True
  pytest tests/ -q
  ```
  Éxito: versión correcta + suite verde.

- [ ] **B4. Subir el dataset v3** (desde local)
  ```bash
  rsync -avz --progress -e "ssh -p <PUERTO>" \
    data/processed/magicbrush_v3 \
    root@<IP>:/workspace/Grafito/data/processed/
  ```
  Éxito: en el pod, `cat data/processed/magicbrush_v3/stats.json` coincide con A4.

- [ ] **B5. Subir checkpoints v1 y v2** (desde local)
  Estos son necesarios para la evaluación conjunta.
  ```bash
  rsync -avz --progress -e "ssh -p <PUERTO>" \
    models/checkpoints/grafito-magicbrush \
    root@<IP>:/workspace/Grafito/models/checkpoints/

  rsync -avz --progress -e "ssh -p <PUERTO>" \
    models/checkpoints/grafito-magicbrush-v2 \
    root@<IP>:/workspace/Grafito/models/checkpoints/
  ```
  Éxito: ambos checkpoints cargan en el pod con `scripts/test_checkpoint.py`.

- [ ] **B6. Smoke de entrenamiento (5 pasos, ~2 min)**
  ```bash
  accelerate launch --mixed_precision="fp16" src/scripts/train_instruct_pix2pix.py \
    --pretrained_model_name_or_path="models/checkpoints/grafito-magicbrush-v2" \
    --dataset_name="data/processed/magicbrush_v3" \
    --original_image_column="original_image" \
    --edited_image_column="edited_image" \
    --edit_prompt_column="edit_prompt" \
    --resolution=512 --random_flip \
    --train_batch_size=2 --gradient_accumulation_steps=8 \
    --gradient_checkpointing \
    --max_train_steps=5 --learning_rate=5e-5 \
    --mixed_precision=fp16 \
    --conditioning_dropout_prob=0.1 \
    --output_dir=/tmp/smoke_v3
  ```
  Éxito: 5 pasos sin OOM ni errores. Si OOM: batch 1 + acumulación 16.

- [ ] **B7. Run completo v3** (en `tmux` para que no muera si cae el SSH)
  ```bash
  tmux new -s train_v3
  export MODEL_NAME="models/checkpoints/grafito-magicbrush-v2"
  export DATASET_NAME="data/processed/magicbrush_v3"
  export OUTPUT_DIR="models/checkpoints/grafito-v3"

  accelerate launch --mixed_precision="fp16" src/scripts/train_instruct_pix2pix.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_NAME \
    --original_image_column="original_image" \
    --edited_image_column="edited_image" \
    --edit_prompt_column="edit_prompt" \
    --resolution=512 --random_flip \
    --train_batch_size=2 --gradient_accumulation_steps=8 \
    --gradient_checkpointing \
    --max_train_steps=10000 \
    --learning_rate=5e-5 --max_grad_norm=1 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.10 \
    --checkpointing_steps=1000 --checkpoints_total_limit=2 \
    --mixed_precision=fp16 --seed=42 \
    --validation_image="assets/example.jpg" \
    --validation_prompt="add a black hat" \
    --num_validation_images=2 \
    --output_dir=$OUTPUT_DIR
  ```
  Duración estimada: ~7–9 h en RTX 4090, ~11–13 h en RTX 3090.
  Éxito: `models/checkpoints/grafito-v3/` con pipeline completo al terminar.

- [ ] **B8. Smokes de retrato durante el run**
  Cada ~1000 pasos, generar en el pod los casos críticos de `review/`:
  ```bash
  python scripts/test_checkpoint.py \
    --checkpoint models/checkpoints/grafito-v3 \
    --image review/magicbrush-validacion-quitar.png \
    --prompt "remove his glasses" \
    --resolution 512 --seed 0 --output outputs/v3_smoke_remove.jpg

  python scripts/test_checkpoint.py \
    --checkpoint models/checkpoints/grafito-v3 \
    --image review/magicbrush-validacion-reemplazar.png \
    --prompt "replace the red cap with a blue beanie" \
    --resolution 512 --seed 0 --output outputs/v3_smoke_replace.jpg
  ```
  Éxito: ojos abiertos, frente limpia, objeto eliminado/reemplazado sin destruir la cara.
  Si la calidad empeora claramente: parar, conservar checkpoint y decidir.

- [ ] **B9. Monitoreo** (2–3 veces durante el run)
  Revisar pérdida y validaciones por época. Si diverge o los smokes de retrato
  degradan la cara: parar, conservar el mejor checkpoint y decidir.

---

## Fase C — Evaluación, descarga y cierre (el mismo día que B7 termina)

- [ ] **C1. Evaluación cuantitativa conjunta a 512 px** (en el pod)
  ```bash
  DATASET="data/processed/magicbrush"
  for CHK in \
    "timbrooks/instruct-pix2pix" \
    "models/checkpoints/grafito-magicbrush" \
    "models/checkpoints/grafito-magicbrush-v2" \
    "models/checkpoints/grafito-v3"
  do
    NAME=$(basename "$CHK")
    python src/scripts/evaluate.py \
      --checkpoint "$CHK" \
      --dataset "$DATASET" \
      --split validation \
      --resolution 512 \
      --num_inference_steps 20 \
      --image_guidance_scale 1.5 \
      --guidance_scale 7.0 \
      --seed 42 \
      --output "outputs/eval_v3_${NAME}.json"
  done
  ```
  Éxito: tabla v3 vs v2 vs v1 vs base. Umbrales de adopción:
  - LPIPS v3 ≤ 0.18 y < LPIPS v2
  - CLIP v3 ≥ 0.25 y ≥ CLIP v2 − 0.005

- [ ] **C2. Panel cualitativo fijo** (8 casos × 2 seeds, del plan §6.2)
  Prompts del panel en `assets/test_prompts.json` o manualmente:
  1. Estatua `add a hat`
  2. Retrato `add a black hat`
  3. Retrato `make the background blue`
  4. Interior doméstico (tinte magenta)
  5. Objeto simple
  6. Paisaje / escena amplia
  7. Imagen con bordes claros
  8. Edición de estilo `make it look like a painting`
  Más los casos críticos de `review/`:
  9. `remove his glasses`
  10. `replace the red cap with a blue beanie`
  11. `make the background blue` sobre el mismo retrato.
  Rúbrica 1–5 en fidelidad facial, adherencia, artefactos, cast de color.
  Éxito: ≥ 4/5 en quitar/reemplazar en retratos y en caras generales.

- [ ] **C3. Decisión v3 vs v2**
  Según §1.2 y §10 de `docs/TRAINING_V3_PLAN.md`.
  Si v3 no supera: conservar v2, documentar y seguir con C5/C6 igualmente.

- [ ] **C4. Descargar el checkpoint ganador** (a local)
  Si se adopta v3:
  ```bash
  rsync -avz --progress -e "ssh -p <PUERTO>" \
    root@<IP>:/workspace/Grafito/models/checkpoints/grafito-v3 \
    models/checkpoints/grafito-magicbrush-v3
  ```
  Éxito: `models/checkpoints/grafito-magicbrush-v3/` ~4 GB con `unet`, `vae`,
  `text_encoder`, `tokenizer`, `scheduler`, `model_index.json`.

- [ ] **C5. Verificación en local (antes de destruir nada)**
  ```bash
  python scripts/test_checkpoint.py \
    --checkpoint models/checkpoints/grafito-magicbrush-v3 \
    --image assets/example.jpg --prompt "add a black hat" \
    --resolution 512 --output outputs/test_v3.jpg
  ```
  Éxito: carga y edita correctamente en el iMac (CPU/MPS).

- [ ] **C6. Descargar métricas y panel**
  ```bash
  rsync -avz --progress -e "ssh -p <PUERTO>" \
    root@<IP>:/workspace/Grafito/outputs/eval_v3_* \
    outputs/
  rsync -avz --progress -e "ssh -p <PUERTO>" \
    root@<IP>:/workspace/Grafito/outputs/v3_* \
    outputs/
  ```

- [ ] **C7. Destruir la instancia**
  Solo después de C5. Confirmar en el panel de Vast.ai: sin instancia corriendo,
  sin volumen, sin cargos pendientes. **Este paso cierra el grifo.**

- [ ] **C8. Backup y documentación**
  Subir `grafito-magicbrush-v3` a Google Drive (segunda copia).
  Actualizar `docs/EVALUATION.md` (métricas y panel), `docs/TRAINING.md`
  (comando final usado), `docs/CHANGELOG.md` (decisión v3 vs v2) y
  `docs/TRAINING_V3_PLAN.md` (checklist de ejecución marcado).

---

## Notas de recuperación

- Si se cae el SSH durante B7, reconectar con `tmux a -t train_v3`.
- Si hay OOM: cambiar `--train_batch_size=1 --gradient_accumulation_steps=16`.
- Si el checkpoint se llama `diffusion_pytorch_model-001.safetensors`, renombrar
  a `diffusion_pytorch_model.safetensors` antes de cargar en diffusers.
- Para regenerar el dataset v3 desde cero, primero re-descargar MagicBrush y
  Instruct-CelebA, luego seguir la Fase A completa.
