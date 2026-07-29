# Runbook v4 — de la auditoría de datos al modelo descargado

Procedimiento operativo del entrenamiento v4, tarea por tarea, con comandos
exactos y criterio de éxito. El porqué de cada decisión está en
`docs/TRAINING_V4_PLAN.md`; este documento es solo el paso a paso.

Leyenda: `[x]` hecha · `[ ]` pendiente. `<IP>`/`<PUERTO>` = los que asigne
Vast.ai a la instancia. `<REPO>` = URL del repo (GitHub).

Diferencias clave frente a v3: **6000 pasos** (no 10000), checkpoints cada
1000 **sin límite**, evaluación de los intermedios 4000/5000/6000, y una
**puerta de auditoría de datos** antes de gastar GPU.

Dataset disponible en local: `data/processed/magicbrush_v3/`
- Train: 27.098 ejemplos (MagicBrush filtrado 7.645 ×2 personas + Instruct-CelebA 19.453)
- Validation: 528 ejemplos (idéntico a v1/v2/v3)
- Peso: ~19 GB

---

## Fase A — Auditoría de datos y preparación local ($0)

- [x] **A1. Script de auditoría**
  `src/scripts/audit_v3_data.py` con tests en `tests/test_audit_v3_data.py`.
  Separa la mezcla por procedencia (`face_id` no nulo = Instruct-CelebA) y
  mide por par: similitud CLIP imagen-imagen (emparejamiento), LPIPS
  (magnitud de edición), CLIP texto-imagen (adherencia) y presencia facial,
  con una muestra de control de MagicBrush. Genera láminas de contacto por
  atributo.

- [x] **A2. Ejecutar la auditoría** (~30–60 min en MPS)
  ```bash
  source .venv/bin/activate
  python src/scripts/audit_v3_data.py \
    --dataset_dir data/processed/magicbrush_v3 \
    --sample_per_attribute 100 \
    --magicbrush_control 200 \
    --output_json outputs/audit_v3/audit_v3.json \
    --sheets_dir outputs/audit_v3
  ```
  Éxito: `outputs/audit_v3/audit_v3.json` + `sheet_<atributo>.png` por
  atributo + `sheet_control_magicbrush.png`.
  **Ejecutada 2026-07-29.**

- [x] **A3. Revisión humana de resultados**
  1. Leer el resumen por atributo (consola/JSON) contra el control MagicBrush.
  2. Abrir las láminas `outputs/audit_v3/sheet_*.png` y verificar a ojo:
     - ¿la editada es la **misma persona** que la original? (emparejamiento)
     - ¿la edición corresponde al prompt?
     - ¿hay artefactos sintéticos graves?
  Éxito: decisión tomada según la puerta del plan §0.5.
  **Resultado 2026-07-29: emparejamiento sistemáticamente roto** (CLIP
  img-img 0.54–0.59 vs 0.94 del control; LPIPS 0.55 vs 0.11; 90–98 % de pares
  sospechosos en los 8 atributos; láminas muestran personas distintas).

- [ ] **A4. Decisión de la puerta: opción A elegida (2026-07-29) — re-emparejar Instruct-CelebA**

  Pasos concretos:

  - [x] **A4.1. Descargar las fuentes con nombres de archivo reales** (hecho 2026-07-29)
    ```bash
    mkdir -p data/raw/instruct_celeba && cd data/raw/instruct_celeba
    for i in 001 002 003; do
      curl -sL --retry 3 -o "Instruct_CelebA_Dataset_2.zip.$i" \
        "https://github.com/Junyi136/Instruct-Edit/releases/download/vv1.0/Instruct_CelebA_Dataset_2.zip.$i"
    done && cd -

    mkdir -p data/raw/celebamask_hq
    curl -sL --retry 3 -o data/raw/celebamask_hq/CelebAMask-HQ.zip \
      "https://huggingface.co/datasets/liusq/CelebAMask-HQ/resolve/main/CelebAMask-HQ.zip"
    ```
    Éxito: 3 zips (~5.7 GB) + `CelebAMask-HQ.zip` (~2.9 GB) que contiene
    `CelebAMask-HQ/CelebA-HQ-img/<face_id>.jpg` (30.000 imágenes).

  - [x] **A4.2. Extraer solo CelebA-HQ-img** (hecho 2026-07-29: 30.000 jpg)
    ```bash
    unzip -q -o data/raw/celebamask_hq/CelebAMask-HQ.zip \
      "CelebAMask-HQ/CelebA-HQ-img/*" -d data/raw/celebamask_hq/extracted
    ```
    Éxito: `data/raw/celebamask_hq/extracted/CelebAMask-HQ/CelebA-HQ-img/`
    con 30.000 jpg.

  - [x] **A4.3. Reprocesar Instruct-CelebA con emparejamiento por nombre** (hecho 2026-07-29: 19.755 ejemplos, 0 originales faltantes)
    `prepare_instruct_celeba.py` ya no acepta el fallback posicional roto;
    aborta si falta >1 % de originales.
    ```bash
    python src/scripts/prepare_instruct_celeba.py \
      --zip_dir data/raw/instruct_celeba \
      --celebamask_hq_dir data/raw/celebamask_hq/extracted/CelebAMask-HQ/CelebA-HQ-img \
      --output_dir data/processed/instruct_celeba \
      --max_samples 20000 --seed 42 --keep_extracted
    ```
    Éxito: `data/processed/instruct_celeba/stats.json` con ~19–20k ejemplos
    y `missing_originals` ≈ 0.

  - [x] **A4.4. Re-auditar el nuevo Instruct-CelebA (misma puerta que A2/A3)** (SUPERADA 2026-07-29: CLIP img-img 0.85–0.96, 0 % mal emparejamiento salvo gender 9 % esperable, LPIPS 0.02–0.14, 0 % reemplazo de escena; láminas verificadas)
    Reconstruir una mini-mezcla de verificación o auditar directamente:
    ```bash
    python src/scripts/audit_v3_data.py \
      --dataset_dir data/processed/magicbrush_v4 \
      --sample_per_attribute 100 \
      --output_json outputs/audit_v4/audit_v4.json \
      --sheets_dir outputs/audit_v4
    ```
    (se ejecuta tras A4.5; la mezcla conserva la procedencia).
    Éxito: CLIP img-img medio ≈ control MagicBrush (~0.9), pares sospechosos
    de mal emparejamiento < 5 %, láminas con la misma persona en ambas
    columnas. **Si no pasa, no se alquila GPU.**

  - [x] **A4.5. Reconstruir la mezcla como `magicbrush_v4`** (hecho 2026-07-29: train 27.400 / validation 528)
    La parte MagicBrush de `magicbrush_v3` (ejemplos con `face_id` nulo) ya
    pasó los filtros de v3, así que se reutiliza des-duplicando el
    sobremuestreo ×2 de personas:
    ```bash
    python - <<'EOF'
    from datasets import load_from_disk, DatasetDict
    d = load_from_disk("data/processed/magicbrush_v3")
    train = d["train"]
    seen, keep = set(), []
    for i, (im, t, f) in enumerate(zip(train["img_id"], train["turn_index"], train["face_id"])):
        if f is not None or (im, t) in seen:
            continue
        seen.add((im, t))
        keep.append(i)
    mb = train.select(keep)
    DatasetDict({"train": mb, "validation": d["validation"]}).save_to_disk(
        "data/processed/magicbrush_v4_mb_base"
    )
    print("MagicBrush base:", len(mb))  # esperado: 6765
    EOF

    python src/scripts/prepare_v3_mix.py \
      --magicbrush_dir data/processed/magicbrush_v4_mb_base \
      --instruct_celeba_dir data/processed/instruct_celeba \
      --output_dir data/processed/magicbrush_v4 \
      --person_repeat 2 --seed 42
    ```
    Éxito: `data/processed/magicbrush_v4/stats.json` con train ~26.000
    (7.645 MagicBrush ×2 personas + ~19k Instruct-CelebA) y validation 528.

  - [x] **A4.6. Liberar intermedios** (hecho 2026-07-29: ~18 GB liberados; se conservan los 3 zips fuente y CelebA-HQ-img/ hasta validar el run)
    ```bash
    rm -rf data/raw/instruct_celeba/extracted \
           data/raw/instruct_celeba/Instruct_CelebA_Dataset_2_merged.zip \
           data/raw/celebamask_hq/CelebAMask-HQ.zip
    ```
    (Conservar los 3 zips fuente y `CelebA-HQ-img/` extraído hasta validar
    el run; luego decidir.)

  - **B. (descartada) Entrenar sin Instruct-CelebA.**

- [ ] **A5. Tests y verificación local**
  ```bash
  pytest tests/ -q
  python -c "from datasets import load_from_disk; d=load_from_disk('data/processed/magicbrush_v4'); print(d)"
  ```
  Éxito: suite verde y el dataset carga.

- [ ] **A6. Commit y push del repo** (código, tests, docs; sin datos ni pesos).
  Commit local hecho 2026-07-29 (rama `main`, árbol limpio). **Falta el push,
  lo hace el usuario**: `git push`. Sin push, el pod clonará un repo viejo.
  Éxito: `git status` limpio y el commit visible en GitHub; el pod podrá
  clonar el repo actualizado.

- [ ] **A7. Empaquetar y subir a Drive** (local, gratis, sin reloj de pod)
  Principio: tu conexión casera sube con calma y gratis; el pod descargará
  desde Drive a velocidad de datacenter durante las horas pagadas.

  **A7.1. Generar los `.tar`** (hecho 2026-07-29 con este comando):
  ```bash
  tar -cf magicbrush_v4.tar -C data/processed magicbrush_v4
  tar -cf grafito-magicbrush-v2.tar -C models/checkpoints grafito-magicbrush-v2
  shasum -a 256 magicbrush_v4.tar grafito-magicbrush-v2.tar
  ```
  Quedan en la raíz del repo (`magicbrush_v4.tar` 19 GB,
  `grafito-magicbrush-v2.tar` 3.6 GB; `*.tar` está en `.gitignore`).
  **No se commitean**: son artefactos de transferencia; bórralos cuando el
  run termine. SHA-256 generados (verificalos en B4/B5 con `sha256sum`):
  - `magicbrush_v4.tar`: `61c56d3e079bd06b02db77f2a04705f9536c01fbd1b30e47f3d19f66cebf1a82`
  - `grafito-magicbrush-v2.tar`: `45ce9a6eb7510c78cd99333cea5abce9294cb67526b6918cf10cf392304442ae`

  No comprimir con gzip: las imágenes ya están en PNG dentro de los arrow;
  el `.tar` solo empaqueta y transfiere mucho más rápido que miles de
  archivos sueltos.

  **A7.2. Subir a Drive** (pendiente, lo hace el usuario):
  1. En [drive.google.com](https://drive.google.com), crea una carpeta
     llamada `grafito-v4`.
  2. Arrastra `magicbrush_v4.tar` y `grafito-magicbrush-v2.tar` a esa
     carpeta (o clic derecho → "Subir archivo"). ~23 GB en total: según tu
     subida puede tardar horas; es normal y no cuesta nada. Verifica antes
     que tu Drive tiene ≥ 25 GB libres.
  3. Cuando termine, comprueba que el tamaño mostrado en Drive coincide con
     el local (`ls -lh`).
  4. Obtén el **file ID** de cada archivo: clic derecho → Compartir →
     "Cualquier persona con el enlace" → copiar enlace. El ID es la parte
     entre `/d/` y `/view`:
     `https://drive.google.com/file/d/<ESTE_ES_EL_ID>/view?usp=sharing`
  5. Anota los dos IDs: `<ID_DATASET>` (magicbrush_v4.tar) y `<ID_V2>`
     (grafito-magicbrush-v2.tar). Se usan en B4 y B5.

  Alternativa por terminal si tienes `rclone` configurado con tu Drive:
  ```bash
  rclone mkdir drive:grafito-v4
  rclone copy --progress magicbrush_v4.tar drive:grafito-v4/
  rclone copy --progress grafito-magicbrush-v2.tar drive:grafito-v4/
  rclone link drive:grafito-v4/magicbrush_v4.tar    # para sacar el ID
  rclone link drive:grafito-v4/grafito-magicbrush-v2.tar
  ```

  Éxito: los dos `.tar` en Drive con tamaño idéntico al local y los dos
  file IDs anotados.

---

## Fase B — Pod de Vast.ai (reloj corriendo: minimizar horas)

- [ ] **B1. Elegir oferta**
  GPU: RTX 3090 / 4090 / A10 / A100 (≥ 24 GB). Evitar RTX 50xx/Blackwell.
  Driver CUDA del host ≥ 12.1. **Disco local ≥ 130 GB** (los 6 checkpoints
  intermedios con estado de optimizador pesan ~10 GB cada uno). On-demand
  (no spot). Imagen: `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`.
  **Sin Network Volume.** Alarma de gasto en $15.

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

- [ ] **B4. Descargar el dataset desde Drive** (en el pod, rápido)
  `<ID_DATASET>` = file ID de `magicbrush_v4.tar` anotado en A7.2.
  ```bash
  pip install gdown
  gdown --id <ID_DATASET> -O /tmp/magicbrush_v4.tar
  # o equivalente: gdown "https://drive.google.com/uc?id=<ID_DATASET>" -O /tmp/magicbrush_v4.tar
  sha256sum /tmp/magicbrush_v4.tar   # debe coincidir con el SHA-256 de A7.1
  tar -xf /tmp/magicbrush_v4.tar -C data/processed/ && rm /tmp/magicbrush_v4.tar
  ```
  (Si `gdown` se atraganta con archivos grandes: `rclone` configurado con tu
  Drive, o pedir el enlace directo. Drive→pod suele tardar minutos, no horas.)
  Éxito: SHA-256 coincide y `cat data/processed/magicbrush_v4/stats.json`
  muestra train 27400 / validation 528.

- [ ] **B5. Descargar el checkpoint v2 desde Drive** (en el pod)
  `<ID_V2>` = file ID de `grafito-magicbrush-v2.tar` anotado en A7.2.
  v2 es el punto de partida del entrenamiento, la referencia de la evaluación
  y el donante de componentes de pipeline para materializar intermedios.
  ```bash
  gdown --id <ID_V2> -O /tmp/grafito-magicbrush-v2.tar
  sha256sum /tmp/grafito-magicbrush-v2.tar   # debe coincidir con A7.1
  tar -xf /tmp/grafito-magicbrush-v2.tar -C models/checkpoints/ && rm /tmp/grafito-magicbrush-v2.tar
  ```
  Éxito: SHA-256 coincide y el checkpoint carga en el pod con
  `scripts/test_checkpoint.py`.
  Alternativa si Drive falla: `rsync` directo desde local como en v3 (lento
  según tu subida, pero es plan B válido).

- [ ] **B6. Smoke de entrenamiento (5 pasos, ~2 min)**
  ```bash
  accelerate launch --mixed_precision="fp16" src/scripts/train_instruct_pix2pix.py \
    --pretrained_model_name_or_path="models/checkpoints/grafito-magicbrush-v2" \
    --dataset_name="data/processed/magicbrush_v4" \
    --original_image_column="original_image" \
    --edited_image_column="edited_image" \
    --edit_prompt_column="edit_prompt" \
    --resolution=512 --random_flip \
    --train_batch_size=2 --gradient_accumulation_steps=8 \
    --gradient_checkpointing \
    --max_train_steps=5 --learning_rate=5e-5 \
    --mixed_precision=fp16 \
    --conditioning_dropout_prob=0.1 \
    --output_dir=/tmp/smoke_v4
  ```
  Éxito: 5 pasos sin OOM ni errores. Si OOM: batch 1 + acumulación 16.

- [ ] **B7. Run completo v4** (en `tmux` para que no muera si cae el SSH)
  ```bash
  tmux new -s train_v4
  export MODEL_NAME="models/checkpoints/grafito-magicbrush-v2"
  export DATASET_NAME="data/processed/magicbrush_v4"
  export OUTPUT_DIR="models/checkpoints/grafito-v4"

  accelerate launch --mixed_precision="fp16" src/scripts/train_instruct_pix2pix.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_NAME \
    --original_image_column="original_image" \
    --edited_image_column="edited_image" \
    --edit_prompt_column="edit_prompt" \
    --resolution=512 --random_flip \
    --train_batch_size=2 --gradient_accumulation_steps=8 \
    --gradient_checkpointing \
    --max_train_steps=6000 \
    --learning_rate=5e-5 --max_grad_norm=1 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.10 \
    --checkpointing_steps=1000 \
    --mixed_precision=fp16 --seed=42 \
    --validation_image="assets/example.jpg" \
    --validation_prompt="add a black hat" \
    --num_validation_images=2 \
    --output_dir=$OUTPUT_DIR
  ```
  Ojo: **sin `--checkpoints_total_limit`** (queremos conservar los 6).
  Duración estimada: ~4.5–5 h en RTX 4090, ~6.5–7 h en RTX 3090.
  Éxito: `models/checkpoints/grafito-v4/` con pipeline completo +
  `checkpoint-1000` … `checkpoint-6000`.

- [ ] **B8. Smokes de retrato sobre intermedios durante el run**
  Los `checkpoint-N` son estados de entrenamiento (solo traen `unet/`); para
  probarlos hay que **materializarlos** como pipeline completo usando v2 como
  donante:
  ```bash
  materialize() {
    N=$1
    DEST=models/checkpoints/grafito-v4-step$N
    mkdir -p $DEST
    cp -r models/checkpoints/grafito-magicbrush-v2/vae \
          models/checkpoints/grafito-magicbrush-v2/text_encoder \
          models/checkpoints/grafito-magicbrush-v2/tokenizer \
          models/checkpoints/grafito-magicbrush-v2/scheduler \
          models/checkpoints/grafito-magicbrush-v2/model_index.json $DEST/
    cp -r models/checkpoints/grafito-v4/checkpoint-$N/unet $DEST/unet
  }
  materialize 3000   # y luego 5000 cuando exista
  ```
  Generar los casos críticos en cada punto:
  ```bash
  for CHK in models/checkpoints/grafito-v4-step3000; do
    python scripts/test_checkpoint.py --checkpoint $CHK \
      --image review/magicbrush-validacion-quitar.png \
      --prompt "remove his glasses" \
      --resolution 512 --seed 0 --output outputs/v4_smoke_remove.jpg
    python scripts/test_checkpoint.py --checkpoint $CHK \
      --image review/magicbrush-validacion-reemplazar.png \
      --prompt "replace the red cap with a blue beanie" \
      --resolution 512 --seed 0 --output outputs/v4_smoke_replace.jpg
    python scripts/test_checkpoint.py --checkpoint $CHK \
      --image review/magicbrush-validacion-reemplazar.png \
      --prompt "make the background light blue" \
      --resolution 512 --seed 0 --output outputs/v4_smoke_background.jpg
  done
  ```
  Éxito: sujeto preservado en `background` y `replace`; sin sobre-edición.
  Si un intermedio es claramente mejor que la tendencia, se puede parar el
  run antes: es exactamente lo que v4 busca.

- [ ] **B9. Monitoreo** (2–3 veces durante el run)
  Revisar pérdida y validaciones. Si diverge: parar, conservar el mejor
  checkpoint materializado y decidir.

---

## Fase C — Evaluación, descarga y cierre (el mismo día que B7 termina)

- [ ] **C1. Materializar los intermedios finales**
  ```bash
  materialize 4000
  materialize 5000
  # checkpoint-6000 == pipeline final en models/checkpoints/grafito-v4
  ```
  Nota (heredada de v3): los pipelines guardados por el script no incluyen
  `safety_checker`; copiarlo del modelo base si algún script lo requiere
  (`test_checkpoint.py` ya carga con `safety_checker=None`).

- [ ] **C2. Evaluación cuantitativa a 512 px** (en el pod)
  El split validation de `magicbrush_v4` es idéntico al de v1/v2/v3 (528
  ejemplos); se usa como dataset del arnés.
  ```bash
  DATASET="data/processed/magicbrush_v4"
  for CHK in \
    "timbrooks/instruct-pix2pix" \
    "models/checkpoints/grafito-magicbrush-v2" \
    "models/checkpoints/grafito-v4-step4000" \
    "models/checkpoints/grafito-v4-step5000" \
    "models/checkpoints/grafito-v4"
  do
    NAME=$(echo "$CHK" | tr '/' '-')
    python src/scripts/evaluate.py \
      --checkpoint "$CHK" \
      --dataset "$DATASET" \
      --split validation \
      --resolution 512 \
      --num_inference_steps 20 \
      --image_guidance_scale 1.5 \
      --guidance_scale 7.0 \
      --seed 42 \
      --output "outputs/eval_v4_${NAME}.json"
  done
  ```
  Éxito: tabla base vs v2 vs v4-4000 vs v4-5000 vs v4-6000. Umbrales del plan
  §1.2: LPIPS < 0.2405 y CLIP ≥ 0.25.

- [ ] **C3. Panel cualitativo fijo** (6 casos × 2 seeds, mismos prompts y seeds
  que el panel v3)
  1. `remove his glasses` (informativo, no es criterio).
  2. `make the background light blue` (retrato) — **criterio clave**.
  3. `make his jacket bright yellow` — sobre-edición.
  4. `add a black hat` (retrato).
  5. Estatua `add a hat`.
  6. Estatua `make it look like a painting` (informativo).

  Para cada uno de v2, v4-4000, v4-5000 y v4-6000. Rúbrica 1–5 en fidelidad
  facial, adherencia, artefactos, cast de color.
  Éxito: sujeto preservado en `background` 2/2 y panel general ≥ 4/5 en el
  mejor intermedio.

- [ ] **C4. Decisión del ganador**
  Según §8 de `docs/TRAINING_V4_PLAN.md`: adoptar el mejor intermedio que
  cumpla toda la tabla; si ninguno, rollback a v2 y documentar la curva.

- [ ] **C5. Subir resultados a Drive desde el pod** (rápido, antes de destruir)
  El checkpoint ganador, las métricas y el panel viajan pod→Drive a velocidad
  de datacenter; la bajada a local se hace después, gratis y sin reloj.
  ```bash
  tar -cf /tmp/grafito-v4-ganador.tar -C models/checkpoints grafito-v4-step<N>
  tar -cf /tmp/resultados_v4.tar outputs/eval_v4_*.json outputs/v4_* outputs/panel_v4 2>/dev/null
  pip install gdown || true
  # Subida: usar rclone (recomendado para subir) o la web de Drive desde tu
  # máquina tras un rsync pequeño. Alternativa simple: rsync directo a local
  # si tu subida lo permite (~4 GB el checkpoint).
  rclone copy /tmp/grafito-v4-ganador.tar drive:grafito-v4/
  rclone copy /tmp/resultados_v4.tar drive:grafito-v4/
  ```
  Éxito: ambos `.tar` visibles en Drive con tamaño correcto.

- [ ] **C6. Descargar a local y verificar (antes de destruir nada)**
  ```bash
  # En local, desde Drive (o rsync si se eligió esa vía):
  tar -xf grafito-v4-ganador.tar -C models/checkpoints/
  mv models/checkpoints/grafito-v4-step<N> models/checkpoints/grafito-magicbrush-v4
  python scripts/test_checkpoint.py \
    --checkpoint models/checkpoints/grafito-magicbrush-v4 \
    --image assets/example.jpg --prompt "add a black hat" \
    --resolution 512 --output outputs/test_v4.jpg
  ```
  Éxito: `grafito-magicbrush-v4/` ~4 GB con `unet`, `vae`, `text_encoder`,
  `tokenizer`, `scheduler`, `model_index.json`; carga y edita correctamente
  en el iMac (CPU/MPS).

- [ ] **C7. Descargar métricas y panel a local**
  ```bash
  tar -xf resultados_v4.tar -C .   # desde el tar bajado de Drive
  ```

- [ ] **C8. Destruir la instancia**
  Solo después de C6. Confirmar en el panel de Vast.ai: sin instancia
  corriendo, sin volumen, sin cargos pendientes. **Este paso cierra el grifo.**

- [ ] **C9. Backup y documentación**
  Subir el checkpoint ganador a Google Drive (segunda copia).
  Actualizar `docs/EVALUATION.md` (curva por pasos y panel),
  `docs/CHANGELOG.md` (decisión v4) y marcar este checklist.

---

## Notas de recuperación

- Si se cae el SSH durante B7, reconectar con `tmux a -t train_v4`.
- Si hay OOM: cambiar a `--train_batch_size=1 --gradient_accumulation_steps=16`.
- Si falta disco en el pod: una vez materializado un `checkpoint-N`, su
  estado de optimizador ya no es necesario para evaluar (solo para reanudar);
  se puede borrar `checkpoint-N/optimizer.bin` previo aviso.
- Si el `unet` del checkpoint se llama `diffusion_pytorch_model-001.safetensors`,
  renombrar a `diffusion_pytorch_model.safetensors` antes de cargar.
- Si el run se interrumpe: reanudar con `--resume_from_checkpoint latest` el
  mismo día (los 6 checkpoints se conservan).
