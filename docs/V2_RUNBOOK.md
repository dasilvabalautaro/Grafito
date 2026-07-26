# Runbook v2 — del dataset al modelo descargado

Procedimiento operativo del entrenamiento v2, tarea por tarea, con comandos
exactos y criterio de éxito. El porqué de cada decisión está en
`docs/TRAINING_V2_PLAN.md`; este documento es solo el paso a paso.

Leyenda: `[x]` hecha · `[ ]` pendiente. `<IP>`/`<PUERTO>` = los que asigne
Vast.ai a la instancia.

---

## Fase A — Preparación local (coste $0)

- [x] **A1. Puerta de inferencia 512 px**
  Resultado: 2 min 4 s en CPU, sin OOM. Objetivo: 512 px. (2026-07-26)

- [ ] **A2. Dataset a 512 px**
  `python src/scripts/prepare_magicbrush.py --resolution 512 --num_proc 4`
  Éxito: `data/processed/magicbrush/stats.json` con train 8807 / validation 528.

- [ ] **A3. Auditoría del dataset** (verificar las imágenes de entrenamiento)
  `python scripts/audit_dataset.py --output data/processed/magicbrush/audit.json`
  Éxito: cifras anotadas en `docs/TRAINING_V2_PLAN.md` §0.2. Si el tinte o los
  bordes correlacionan con el dataset, corregir antes de seguir.

- [ ] **A4. Mix v2** (sobremuestreo de personas ×2)
  `python src/scripts/prepare_v2_mix.py`
  Éxito: `data/processed/magicbrush_v2/stats.json` coherente
  (train_mixed = 8807 + n_personas).

- [ ] **A5. Commit y push del repo** (código, tests y docs; sin datos ni pesos).
  Éxito: `git status` limpio; el pod podrá clonar el repo actualizado.

---

## Fase B — Pod de Vast.ai (reloj corriendo: minimizar horas)

- [ ] **B1. Elegir oferta**
  GPU: RTX 3090 / 4090 / A10 / A100 (≥ 24 GB). Evitar RTX 50xx/Blackwell.
  Driver CUDA del host ≥ 12.1. Disco local ≥ 80 GB. On-demand (no spot).
  Imagen: `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`.
  **Sin Network Volume.** Alarma de gasto en $20.

- [ ] **B2. Clonar e instalar** (en el pod)
  ```bash
  git clone <url-del-repo> /workspace/Grafito && cd /workspace/Grafito
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

- [ ] **B4. Subir el dataset** (desde local)
  ```bash
  rsync -avz -e "ssh -p <PUERTO>" data/processed/magicbrush_v2 \
    root@<IP>:/workspace/Grafito/data/processed/
  ```
  Éxito: en el pod, `cat data/processed/magicbrush_v2/stats.json` coincide con A4.

- [ ] **B5. Smoke de entrenamiento (5 pasos, ~2 min)**
  ```bash
  accelerate launch --mixed_precision="fp16" src/scripts/train_instruct_pix2pix.py \
    --pretrained_model_name_or_path="timbrooks/instruct-pix2pix" \
    --dataset_name="data/processed/magicbrush_v2" \
    --resolution=512 --train_batch_size=2 --gradient_accumulation_steps=8 \
    --gradient_checkpointing --max_train_steps=5 --learning_rate=5e-5 \
    --mixed_precision=fp16 --output_dir=/tmp/smoke
  ```
  Éxito: 5 pasos sin OOM ni errores. Si OOM: batch 1 + acumulación 16.

- [ ] **B6. Run completo** (en `tmux` para que no muera si cae el SSH)
  ```bash
  tmux new -s train
  accelerate launch --mixed_precision="fp16" src/scripts/train_instruct_pix2pix.py \
    --pretrained_model_name_or_path="timbrooks/instruct-pix2pix" \
    --dataset_name="data/processed/magicbrush_v2" \
    --original_image_column="original_image" \
    --edited_image_column="edited_image" \
    --edit_prompt_column="edit_prompt" \
    --resolution=512 --random_flip \
    --train_batch_size=2 --gradient_accumulation_steps=8 \
    --gradient_checkpointing \
    --max_train_steps=6000 \
    --learning_rate=5e-5 --max_grad_norm=1 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.1 \
    --checkpointing_steps=1000 --checkpoints_total_limit=2 \
    --mixed_precision=fp16 --seed=42 \
    --validation_image="assets/example.jpg" --validation_prompt="add a hat" \
    --num_validation_images=2 \
    --output_dir="models/checkpoints/grafito-v2"
  ```
  Duración estimada: 20 h (4090) / 29 h (3090). Planificar el inicio para
  poder cerrar (Fase C) el mismo día que termina.
  Éxito: `models/checkpoints/grafito-v2` con pipeline completo al terminar.

- [ ] **B7. Monitoreo** (2–3 veces durante el run)
  Revisar progreso y las imágenes de validación por época (caras y tinte).
  Si la pérdida diverge o las validaciones empeoran claramente: parar,
  conservar checkpoint y decidir (ahorra horas de GPU).

---

## Fase C — Evaluación, descarga y cierre (el mismo día que B6 termina)

- [ ] **C1. Evaluación cuantitativa en el pod**
  `python src/scripts/evaluate.py --checkpoint models/checkpoints/grafito-v2`
  Éxito: tabla v2 vs v1 vs base anotada. Umbrales: LPIPS ≤ 0.18,
  CLIP ≥ 0.25 sin regresión.

- [ ] **C2. Panel cualitativo** (8 casos × 2 seeds, del plan §6.2)
  Rúbrica 1–5 en fidelidad, adherencia, artefactos, cast. Anotar en
  `docs/EVALUATION.md`. Éxito: ≥ 4/5 en caras sin degradación.

- [ ] **C3. Decisión v2 vs v1**
  Según §1 del plan. Si v2 no supera los criterios: se conserva v1, se
  documenta, y se sigue con C5/C6 igualmente (destruir instancia).

- [ ] **C4. Descargar el checkpoint ganador** (a local)
  ```bash
  rsync -avz -e "ssh -p <PUERTO>" \
    root@<IP>:/workspace/Grafito/models/checkpoints/grafito-v2 \
    models/checkpoints/grafito-magicbrush-v2
  ```
  Éxito: `models/checkpoints/grafito-magicbrush-v2/` ~4 GB con `unet`, `vae`,
  `text_encoder`, `tokenizer`, `scheduler`, `model_index.json`.

- [ ] **C5. Verificación en local (antes de destruir nada)**
  Si el UNet llega como `diffusion_pytorch_model-001.safetensors`, renombrar
  a `diffusion_pytorch_model.safetensors` (incidencia de v1).
  ```bash
  python scripts/test_checkpoint.py \
    --checkpoint models/checkpoints/grafito-magicbrush-v2 \
    --image assets/example.jpg --prompt "add a hat" \
    --resolution 512 --output outputs/test_v2.jpg
  ```
  Éxito: carga y edita correctamente en el iMac.

- [ ] **C6. Destruir la instancia**
  Solo después de C5. Confirmar en el panel de Vast.ai: sin instancia corriendo,
  sin volumen, sin cargos pendientes. **Este paso cierra el grifo.**

- [ ] **C7. Backup y documentación**
  Subir `grafito-magicbrush-v2` a Drive (segunda copia).
  Actualizar `docs/EVALUATION.md` (métricas y panel), `docs/TRAINING.md`
  (comando final usado), `docs/CHANGELOG.md` (decisión v2 vs v1) y
  `docs/TRAINING_V2_PLAN.md` (checklist marcado).
