# Plan de entrenamiento v2 — Grafito MagicBrush

Fecha: 2026-07-26. Estado: propuesta actualizada tras decidir entrenamiento en GPU rentada.

Este documento define el segundo entrenamiento del modelo con especificaciones
exactas. Su objetivo es corregir los defectos observados en la v1 (ver
«Observaciones de calidad» en `docs/NEXT_DEMO.md`).

**Decisión de plataforma (2026-07-26):** el entrenamiento corre en una GPU
rentada de 24 GB (Vast.ai u similar); la restricción inmutable es que el
modelo final ejecute en el hardware local (iMac: i7-10700K, 32 GB RAM, Radeon
Pro 5500 XT 8 GB, MPS).

---

## 1. Objetivo y criterios de éxito

Mejorar `grafito-magicbrush` (v1) en cuatro frentes, en orden de prioridad:

1. **Nitidez**: subir la resolución de trabajo de 256 a 512 px (o 384 como fallback).
2. **Caras**: eliminar la degradación de boca/ojos en retratos.
3. **Color**: quitar el tinte magenta sistemático observado en fotos de interior.
4. **Bordes**: reducir las manchas de color esporádicas en bordes.

### Criterios medibles (obligatorios para adoptar v2)

| Métrica | v1 | Base IP2P | Meta v2 |
|---|---|---|---|
| LPIPS (validation MagicBrush, 528 ej.) | 0.1997 | 0.3316 | **≤ 0.18** |
| CLIP score (mismo split) | 0.2476 | 0.2591 | **≥ 0.25** (sin regresión) |
| Panel cualitativo (rúbrica 1–5, caras) | degradadas | — | **≥ 4/5 sin degradación facial** |

Regla de decisión: v2 se adopta solo si gana en LPIPS, no pierde en CLIP y gana
el panel cualitativo en caras. En caso contrario se conserva v1 (rollback
inmediato: v1 queda intacto en `models/checkpoints/grafito-magicbrush`).

---

## 2. Restricciones de hardware

### Inferencia (inmutable)

El modelo final debe ejecutar en el hardware local, sin excepciones:

- iMac: i7-10700K, 32 GB RAM, Radeon Pro 5500 XT 8 GB (MPS), torch 2.2.2,
  fp32 + `attention_slicing`, diffusers 0.27.2.
- Implicación: se mantiene la arquitectura **InstructPix2Pix / SD 1.5**. Se
  descarta SDXL (no cabe con margen en 8 GB bajo MPS).
- El artefacto final es un pipeline completo en el mismo formato que v1
  (`unet`, `vae`, `text_encoder`, `tokenizer`, `scheduler`), consumible por
  `scripts/demo.py` y `scripts/test_checkpoint.py` sin cambios.

### Entrenamiento (nube, una sola ejecución acotada)

- GPU de 24 GB (RTX 3090/4090 o A10) en Vast.ai o similar, con la receta de
  fine-tune completo del UNet ya probada en v1. LoRA queda descartado: con
  24 GB no hace falta y el fine-tune completo ya demostró mejora en v1.
- **Protocolo de costes (lección de v1, obligatorio):**
  1. **Sin Network Volume.** Solo disco local efímero de la instancia
     (~80 GB: modelo base 5 GB + dataset ~6 GB + 2 checkpoints × ~13 GB +
     entorno). El volumen persistente fue la fuente del cobro continuo.
  2. Presupuesto máximo **$20** con alerta; estimado real $10–15 (§7).
  3. Al terminar: evaluación en el propio pod → bajar el checkpoint ganador
     (~4 GB) por `rsync`/`scp` a local → verificar que carga en local →
     **destruir la instancia el mismo día**.
  4. `--checkpointing_steps=1000 --checkpoints_total_limit=2` para acotar disco.
  5. Si la instancia se interrumpe: reanudar con `--resume_from_checkpoint`
     en una instancia nueva el mismo día.

---

## 3. Fase 0 — Validaciones previas (puertas go/no-go)

### 0.1 Inferencia a 512 px en el hardware local (define la resolución objetivo)

```bash
python scripts/test_checkpoint.py \
  --checkpoint models/checkpoints/grafito-magicbrush \
  --image assets/example.jpg --prompt "add a hat" \
  --resolution 512 --output outputs/gate_512.jpg
```

- **Pasa**: completa sin OOM y en tiempo razonable (≤ 5 min en CPU / ≤ 2 min
  en MPS). La resolución objetivo de entrenamiento es **512**.
- **Falla**: el objetivo baja a **384 px** y se repite la prueba con
  `--resolution 384`. (Un modelo entrenado a 512 también ejecuta a 384/256;
  la puerta define a qué resolución se *evalúa y demuestra* el modelo.)

**Resultado (2026-07-26): PASA.** 512 px en CPU: 2 min 4 s, sin OOM
(`outputs/gate_512.jpg`). Resolución objetivo de entrenamiento: **512 px**.

### 0.2 Auditoría del dataset (en local)

Script corto sobre el dataset procesado (numpy/PIL, sin modelo):

- % de pares original/editada con diferencia media de tono (hue) alta
  → investiga el origen del tinte magenta.
- % de imágenes con píxeles anómalos de alta saturación en los 8 px de borde
  → investiga las manchas de borde.
- Conteo de sesiones con personas (keywords de la Fase 1).

Entregable: cifras anotadas en este documento antes de entrenar. Si el tinte o
los bordes correlacionan con el dataset, corregir en `prepare_magicbrush.py`
antes de seguir.

**Resultado (2026-07-26, `scripts/audit_dataset.py` sobre 8807 ejemplos):**

- **Color: dataset limpio.** ΔRGB medio (editada−original) = R+0.54 G+0.62
  B+0.75 (escala 0–255) y solo 0.33 % de pares con patrón magenta. El tinte
  magenta visto en el retrato de prueba **no viene del dataset**: es artefacto
  del modelo/entrenamiento. No hay corrección de datos que aplicar aquí.
- **Bordes: hallazgo real.** 28.4 % marcado con el heurístico grueso, pero la
  inspección visual separa dos casos: contenido legítimo saturado en el borde
  (alfombras, maletas) y **tiras finas de calibración de color
  (cian/magenta/amarillo) en esquinas de escaneos de libros** — el mismo tipo
  de mancha que produce v1 en las esquinas. Decisión: filtrar con detector
  estricto de esquinas (alta precisión, recall parcial a propósito):
  **44 ejemplos (0.50 %) eliminados** en `prepare_v2_mix.py`
  (`--drop_corner_strips`, activado por defecto). Los detectores más laxos
  marcaban contenido legítimo (falsos positivos del 28–48 %); se descartaron.
  La mitigación residual del artefacto queda en inferencia (re-tirar seed o
  recortar borde) y en el panel de evaluación.
- **Personas: 15.22 % (1340)** — base del sobremuestreo ×2 de la Fase 1.

### 0.3 Selección de GPU y presupuesto

**Compatibilidad torch 2.2.2 + CUDA 12.1 (obligatorio; fue el fallo de v1):**

- **Imagen Docker:** `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`
  (verificada en Docker Hub el 2026-07-26; también existe `-devel`).
  Garantiza torch 2.2.2 compilado contra CUDA 12.1, el mismo que instala
  `requirements.txt` en Linux — así `pip install -r requirements.txt` no
  reinstala ni cambia torch.
- **GPUs válidas:** RTX 3090 / RTX 4090 / A10 / A100 (≥ 24 GB), todas
  soportadas por los builds cu121 de torch 2.2.2.
- **GPUs a evitar:** RTX 50xx / Blackwell (sm_120: exigen CUDA ≥ 12.8, no
  funcionarían con torch 2.2.2) y cualquier GPU < 24 GB.
- **Driver del host:** en Vast.ai cada oferta muestra su versión CUDA;
  exigir ≥ 12.1 (hosts con 12.4+ sirven: el driver es retrocompatible con
  contenedores 12.1).
- **Disco local ≥ 80 GB**, red suficiente para bajar 4 GB al final, y alarma
  de gasto en $20. Evitar spot para no arriesgar un run de ~20 h.

**Pre-flight en el pod antes del run (obligatorio; sin esto no se entrena):**

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# esperado exactamente: 2.2.2 12.1 True
pip install -r requirements.txt
pytest tests/ -q        # la suite debe pasar en el pod
# smoke de entrenamiento (5 pasos, ~2 min):
accelerate launch --mixed_precision="fp16" src/scripts/train_instruct_pix2pix.py \
  --pretrained_model_name_or_path="timbrooks/instruct-pix2pix" \
  --dataset_name="data/processed/magicbrush_v2" \
  --resolution=512 --train_batch_size=2 --gradient_accumulation_steps=8 \
  --gradient_checkpointing --max_train_steps=5 --learning_rate=5e-5 \
  --mixed_precision=fp16 --output_dir=/tmp/smoke
```

Si `torch.cuda.is_available()` es False, la versión no cuadra, o el smoke
falla: **destruir la instancia de inmediato y elegir otra** (cuesta céntimos;
entrenar sobre una máquina rota fue lo caro de v1).

---

## 4. Fase 1 — Datos (en local, se sube al pod)

1. Regenerar el dataset a la resolución aprobada en 0.1:

```bash
python src/scripts/prepare_magicbrush.py --resolution 512 --num_proc 4
```

2. Construir la mezcla v2 (nuevo script `src/scripts/prepare_v2_mix.py`):

- **Base**: MagicBrush train completo (8807 ejemplos).
- **Sobremuestreo de personas ×2**: sesiones cuyo `edit_prompt` contenga
  alguna de: `man, woman, person, people, face, boy, girl, child, baby,
  portrait, selfie, he, she`. Estimado: 15–25 % del dataset. Se duplican en
  la mezcla para corregir el punto débil con caras.
- Salida: `data/processed/magicbrush_v2/` (DatasetDict) + `stats.json`.
- El split validation se mantiene **idéntico al de v1** (mismos 528 ejemplos)
  para que las métricas sean comparables.
- Subir `data/processed/magicbrush_v2/` al pod por `rsync` (~6 GB a 512 px).

---

## 5. Fase 2 — Entrenamiento (en el pod)

### 5.1 Especificación exacta

Receta de v1, cambiando solo lo que ataca los defectos: resolución, mezcla de
datos, `conditioning_dropout_prob` y pasos. Lo demás, idéntico a v1.

| Parámetro | Valor | Cambio vs v1 |
|---|---|---|
| Punto de partida | `timbrooks/instruct-pix2pix` | igual |
| Resolución | 512 (o 384 según 0.1) | **sube de 256** |
| Dataset | `magicbrush_v2` (mezcla con personas ×2) | **nuevo** |
| `conditioning_dropout_prob` | 0.1 | sube de 0.05 |
| Pasos máximos | 6000 (reducir a 5000 si el coste aprieta) | sube de 5000 |
| Batch / acumulación | 2 / 8 (efectivo 16) | ajuste por VRAM a 512 |
| Learning rate / scheduler | 5e-5 / constant | igual |
| Warmup | 0 | igual |
| Precisión | fp16 mixed precision | igual |
| Gradient checkpointing | activado | igual |
| `max_grad_norm` | 1.0 | igual |
| Seed | 42 | igual |
| Checkpointing | cada 1000, conservar 2 | igual |
| Text encoder / VAE | congelados | igual |

No requiere cambios en `src/scripts/train_instruct_pix2pix.py`: todos los
argumentos ya existen.

### 5.2 Comando (referencia, resolución 512)

```bash
export MODEL_NAME="timbrooks/instruct-pix2pix"
export DATASET_NAME="data/processed/magicbrush_v2"
export OUTPUT_DIR="models/checkpoints/grafito-v2"

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
  --conditioning_dropout_prob=0.1 \
  --checkpointing_steps=1000 --checkpoints_total_limit=2 \
  --mixed_precision=fp16 --seed=42 \
  --validation_image="assets/example.jpg" --validation_prompt="add a hat" \
  --num_validation_images=2 \
  --output_dir=$OUTPUT_DIR
```

Añadir una segunda imagen de validación tipo retrato para vigilar caras
durante el run.

---

## 6. Fase 3 — Evaluación y entrega

### 6.1 Cuantitativa (en el pod, GPU)

```bash
python src/scripts/evaluate.py --checkpoint models/checkpoints/grafito-v2
```

Sobre el validation completo (528 ejemplos), v2 vs v1 vs base. Umbrales de §1.
Correr en el pod es mucho más rápido que en local y evita bajar pesos en vano.

### 6.2 Panel cualitativo fijo

8 casos × 2 seeds, mismos prompts para v1 y v2:

1. Estatua `add a hat` (regresión de v1).
2. Retrato tipo `foto.jpg` `add a black hat` (el caso que falla en v1).
3. Retrato `make the background blue` (fidelidad facial con edición de fondo).
4. Interior doméstico (detector de tinte magenta).
5. Objeto simple (caso típico MagicBrush).
6. Paisaje / escena amplia.
7. Imagen con bordes claros (detector de manchas de borde).
8. Edición de estilo `make it look like a painting`.

Rúbrica 1–5 por caso: fidelidad, adherencia al prompt, artefactos, cast de
color. Resultados en `docs/EVALUATION.md`.

### 6.3 Entrega y cierre de costes

1. Bajar el checkpoint ganador (~4 GB) a `models/checkpoints/grafito-magicbrush-v2/`.
2. Verificar carga en local con `scripts/test_checkpoint.py` (ojo al nombre
   del safetensors: si llega como `diffusion_pytorch_model-001.safetensors`,
   renombrar a `diffusion_pytorch_model.safetensors` — incidencia de v1).
3. Verificar inferencia en MPS a la resolución objetivo (puerta 0.1 inversa).
4. **Destruir la instancia el mismo día.** Confirmar en el panel de Vast.ai
   que no queda volumen ni instancia facturando.
5. Backup del checkpoint en Drive (segunda copia).
6. Decisión final v2 vs v1 documentada en `docs/CHANGELOG.md`.

---

## 7. Estimación de tiempos y coste

| Fase | Estimado |
|---|---|
| 0.1 + 0.2 (puertas, local) | 2–3 h |
| 0.3 + setup del pod + subir dataset | 1–2 h |
| Entrenamiento: 6000 pasos × 8 acum., 512 px, batch 2, grad-ckpt en RTX 4090 (~1.5 s/it) | ~20 h (~$10–14 a $0.50–0.70/h) |
| En RTX 3090 (~2.2 s/it) | ~29 h (~$12–17) |
| Evaluación en el pod | 0.5–1 h |
| Descarga, verificación local y destrucción | 1 h |
| **Coste total previsto** | **$12–19** (techo con alarma: $20) |

El run es continuo (~20–29 h); elegir horario para poder destruir la instancia
el mismo día que termina.

---

## 8. Riesgos y mitigaciones

| Riesgo | Prob. | Mitigación |
|---|---|---|
| Sobrecoste por olvidar la instancia encendida | Media | Alarma de gasto $20; checklist §6.3 con destrucción el mismo día |
| Interrupción del run | Baja | Checkpoints cada 1000; reanudar con `--resume_from_checkpoint` el mismo día |
| OOM a 512 con batch 2 | Baja | Bajar batch a 1 y subir acumulación a 16 (mismo efectivo 16) |
| El tinte magenta viene del dataset | Media | Puerta 0.2 decide corrección antes de entrenar |
| Sobreajuste a la mezcla (sobremuestreo) | Baja | `conditioning_dropout_prob=0.1`; imágenes de validación por época |
| v2 no supera a v1 | Media | Criterios de §1: se conserva v1 y el coste queda acotado a ~$15 |
| 512 no ejecuta en local | Media | Puerta 0.1: entrenar a 384 (un modelo 512 ejecuta a 384/256, pero la demo se valida a la resolución entrenada) |

---

## 9. Checklist de ejecución (cierre: 2026-07-27)

- [x] 0.1 Puerta de inferencia local superada: 512 px en CPU, 2 min 4 s, sin OOM (2026-07-26). Objetivo: 512.
- [x] 0.2 Auditoría de dataset anotada (tinte: limpio, 0.33 % patrón magenta; bordes: 44 tiras de calibración en esquinas = 0.50 %, filtradas en el mix; personas: 15.22 %).
- [x] 0.3 GPU: 2× RTX 4090, driver 560, 256 GB disco local, sin Network Volume. Desvío documentado: la imagen disponible traía torch 2.12/py3.12; se mantuvo el pin creando venv con Python 3.11 (vía `uv`) + torch 2.2.2+cu121 de PyPI. Se añadió `torchvision==0.17.2` a requirements (el pre-flight lo detectó).
- [x] Pre-flight: torch 2.2.2+cu121, `pytest` 19 verdes, smoke de 5 pasos `EXIT_0`.
- [x] `src/scripts/prepare_v2_mix.py` + tests (`tests/test_prepare_v2_mix.py`).
- [x] Dataset v2 generado y verificado **en el pod** (stats idénticas a local: train 10099 / validation 528). Desvío: no se subió desde local; la ruta HF del pod estaba capada (~0.5 MB/s) y la subida casera medía 0.45 MB/s; se resolvió con `huggingface-cli download` (8 hilos) + procesado en el pod.
- [x] Entrenamiento completado: 6000 pasos, **4 h 40 min** (2.80 s/it), `EXIT_0`.
- [x] Evaluación anotada en `docs/EVALUATION.md`: LPIPS 0.2405 / CLIP 0.2509 a 512; panel 8×2 con caras y tinte resueltos, esquinas residuales.
- [x] Checkpoint en local (3.6 GB, sin `safety_checker`), carga y edición verificadas a 512 en CPU y en MPS (~31 s con attention slicing + VAE slicing + VAE tiling).
- [x] Instancia destruida (2026-07-27, confirmado por el usuario). Coste estimado del run: ~$8–12 dentro del techo de $20.
- [ ] Backup del checkpoint en Drive (acción del usuario).
- [x] Decisión documentada: **v2 adoptado** para el demo (2026-07-27).
