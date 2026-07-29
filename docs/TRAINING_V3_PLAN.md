# Plan de entrenamiento v3 — Grafito MagicBrush + Instruct-CelebA

Fecha: 2026-07-27.  
Estado: **plan detallado para aprobación**. Se activa por los fallos
catastróficos observados en `review/magicbrush-validacion-quitar.png` y
`review/magicbrush-validacion-reemplazar.png`.

---

## 0. Decisión de activación (go/no-go)

El entrenamiento v3 se activa porque v2 **falla catastróficamente en retratos**
cuando la edición toca la cara:

- `review/magicbrush-validacion-quitar.png`: `remove his glasses` no solo no
  quita las gafas correctamente, sino que elimina la gorra, cierra los ojos y
  alisa la frente.
- `review/magicbrush-validacion-reemplazar.png`: `replace the red cap with a
  blue beanie` cambia el gorro, pero deja manchas oscuras en la piel, textura
  irregular en la frente y una integración deficiente con las cejas.

Estas dos imágenes son evidencia suficiente: **v3 debe resolver quitar y
reemplazar objetos en retratos sin destruir la cara**. Si no lo logra, se
conserva v2 y no se publica v3.

**Ámbito de uso:** investigación. Esto permite incorporar **Instruct-CelebA**
(derivado de CelebA, licencia no comercial) como refuerzo facial.

---

## 1. Objetivos y criterios de éxito

### 1.1 Objetivos, por prioridad

1. **Eliminar fallos catastróficos en quitar y reemplazar objetos sobre
   retratos**: ojos abiertos y alineados, frente sin manchas, objeto
   sustituido/eliminado de forma limpia.
2. **Eliminar manchas de color esporádicas en esquinas** sin depender del
   recorte de bordes de la demo.
3. **Mejorar adherencia fina al prompt**: que `make the background blue` no
   tiña la camiseta, y que color/material explícito se respeten con más
   consistencia.
4. **Cerrar la brecha de LPIPS**: bajar de 0.2405 a **≤ 0.18** a 512 px.
5. **No regresar en CLIP ni en caras**.

### 1.2 Métricas y umbrales obligatorios

| Métrica | v2 a 512 | Base a 512 | Meta v3 | Nota |
|---|---|---|---|---|
| LPIPS vs target (menor mejor) | 0.2405 | 0.3208 | **≤ 0.18** | Comparación manzana-a-manzana a 512 px. |
| CLIP similarity (mayor mejor) | 0.2509 | 0.2523 | **≥ 0.25** | Sin regresión respecto a v2. |
| Panel caras generales (rúbrica 1–5) | 4/5 | — | **≥ 4/5** | Boca y ojos preservados en ediciones sin tocar la cara. |
| Panel quitar/reemplazar en retratos | 1–2/5 | — | **≥ 4/5** | Sin ojos cerrados, sin manchas en piel, sin deformación facial. |
| Panel adherencia fina | 3/5 | — | **≥ 4/5** | Fondo no tiñe objeto; color/material obedecido. |
| Manchas de esquina sin recorte | esporádicas | — | **ninguna en el panel** | El recorte de la demo sigue permitido, pero no debe ser obligatorio. |

Regla de adopción: v3 se adopta solo si gana en el panel de quitar/reemplazar
en retratos, gana en LPIPS y en panel general, y no pierde en CLIP. Si falla
el panel de retratos, **rollback inmediato a v2**.

---

## 2. Principios para no quemar presupuesto

- **El objetivo principal es el dataset**: v3 no es más pasos sobre los mismos
  datos, sino datos que enseñen explícitamente a editar caras conservando la
  identidad.
- **Instruct-CelebA es obligatorio**, no opcional: MagicBrush no tiene
  suficientes ejemplos faciales de remove/replace.
- **Un solo cambio principal a la vez**: cambiamos el dataset (MagicBrush
  filtrado + Instruct-CelebA) y aumentamos pasos. No tocamos resolución,
  arquitectura, batch efectivo ni learning rate sin evidencia.
- **No entrenar sin pasar las puertas 0.x**: si el entorno del pod no verifica
  `torch 2.2.2+cu121`, se destruye la instancia y se elige otra.
- **Techo de gasto duro**: `$20` con alarma; instancia destruida el mismo día
  del run.
- **Sin Network Volume**: solo disco efímero del pod.
- **Evaluación en el pod antes de bajar nada**: no pagamos transferencia de un
  checkpoint que no supere los umbrales.

---

## 3. Fase 0 — Puertas go/no-go

### 0.1 Confirmar los fallos críticos en local

Reproducir exactamente los dos casos de `review/` con v2:

```bash
source .venv/bin/activate
python scripts/test_checkpoint.py \
  --checkpoint models/checkpoints/grafito-magicbrush-v2 \
  --image review/magicbrush-validacion-quitar.png \
  --prompt "remove his glasses" \
  --resolution 512 --seed 0 --output outputs/v3_repro_remove.jpg

python scripts/test_checkpoint.py \
  --checkpoint models/checkpoints/grafito-magicbrush-v2 \
  --image review/magicbrush-validacion-reemplazar.png \
  --prompt "replace the red cap with a blue beanie" \
  --resolution 512 --seed 0 --output outputs/v3_repro_replace.jpg
```

- **Pasa la puerta**: se confirma que v2 falla en ambos casos.
- **Salida**: guardar las reproducciones en `outputs/` como línea base de v3.

### 0.2 Verificar entorno local y checkpoint v2

```bash
pytest tests/ -q
python scripts/test_checkpoint.py \
  --checkpoint models/checkpoints/grafito-magicbrush-v2 \
  --image assets/example.jpg --prompt "add a black hat" \
  --resolution 512 --output outputs/gate_v3_local.jpg
```

- **Pasa**: v2 carga y edita a 512 px sin errores.

### 0.3 Puerta de inferencia local a 512 px

```bash
python scripts/test_checkpoint.py \
  --checkpoint models/checkpoints/grafito-magicbrush-v2 \
  --image assets/example.jpg --prompt "add a black hat" \
  --resolution 512 --output outputs/gate_512_v3.jpg
```

- **Pasa**: completa sin OOM en local (CPU ≤ 5 min, MPS ≤ 2 min).
- Resolución objetivo de entrenamiento: **512 px**.

### 0.4 Dataset v3 preparado y verificado

Ver §4. Antes de alquilar GPU debe existir
`data/processed/magicbrush_v3/` con `stats.json` y validation idéntico a
v1/v2.

---

## 4. Fase 1 — Datos (local, ~6–8 h, $0)

### 4.1 Regenerar MagicBrush a 512 px

```bash
python src/scripts/prepare_magicbrush.py --resolution 512 --num_proc 4
```

Salida: `data/processed/magicbrush/`.

### 4.2 Verificación: MagicBrush no basta para caras

Conteos exactos sobre MagicBrush train (8.807 ejemplos):

| Acción | Total | En personas | % de personas | Ejemplos relevantes para caras |
|---|---|---|---|---|
| `remove` | 637 (7.2 %) | 69 (0.8 %) | 5.1 % | Casi todos son "remove the person", no quitar gafas/gorro. |
| `replace` | 403 (4.6 %) | 25 (0.3 %) | 1.9 % | Muy pocos sobre atributos faciales. |
| `change`/`make` | 1.745 (19.8 %) | 183 (2.1 %) | 13.7 % | Incluye pelo, ropa, expresión; no necesariamente objetos sobre cara. |

**Conclusión**: MagicBrush tiene solo **~277 ejemplos de personas con
remove/replace/change** y la inmensa mayoría **no** enseña a quitar o
reemplazar un accesorio sobre una cara. El boost ×3 interno no es suficiente
para corregir los fallos de `review/`. Por eso v3 requiere **Instruct-CelebA**.

### 4.3 Segunda ronda de filtrado de tiras de esquina

En v2 se filtraron 44 ejemplos seguros (0.50 %) con un detector estricto.
Persisten manchas esporádicas, así que v3 hace un **barrido de alta recall
seguido de revisión visual**:

1. Ejecutar un detector más laxo (por ejemplo `corner_px=48`, `sat>0.6`,
   `min_pixels=15`, `min_dispersion>0.12`) sobre `edited_image` del train.
2. Exportar los candidatos a una lámina de imágenes (por ejemplo 16×16
   thumbs) en `outputs/corner_candidates_v3/`.
3. Revisar visualmente la lámina y marcar solo las tiras reales de
   calibración. No eliminar alfombras, maletas ni contenido legítimo
   saturado.
4. Guardar la lista de índices a descartar en
   `data/processed/magicbrush_v3/dropped_corner_strips.json`.

**Entregable**: número final de tiras eliminadas y justificación de por qué
los descartados son realmente tiras.

### 4.4 Filtro de pares ruidosos (adherencia al prompt)

Este filtro ataca directamente la adherencia fina. Se ejecuta en local con
CPU; es lento pero gratuito.

Criterios para marcar un par como ruidoso (condición **OR**):

- **Cambio visual casi nulo**: `LPIPS(original, edited) < 0.03` y el prompt no
  indica explícitamente conservar la imagen (no contiene `keep`, `same`,
  `unchanged`).
- **Prompt sin relación con la imagen**: `CLIP_similarity(prompt, edited) <
  0.18`.
- **Prompt genérico sin señal**: contiene solo frases como `make it better`,
  `enhance`, `improve quality`, etc.

Herramienta propuesta: crear `src/scripts/filter_noisy_pairs.py` que lea
`data/processed/magicbrush/`, calcule las métricas y escriba una lista de
índices descartables. Ejecutar:

```bash
python src/scripts/filter_noisy_pairs.py \
  --input_dir data/processed/magicbrush \
  --output_dir data/processed/magicbrush_v3 \
  --lpips_threshold 0.03 \
  --clip_threshold 0.18
```

**Entregable**: cuántos ejemplos se descartan por cada criterio y ejemplos
revisados a mano.

### 4.5 Instruct-CelebA: descarga y conversión (obligatorio)

Instruct-CelebA (CoIE) es el dataset que aporta ejemplos faciales de
remove/replace/change. Como el uso es investigación, su licencia (no
comercial por herencia de CelebA) es aceptable.

#### 4.5.1 Descargas necesarias

1. **Instruct-CelebA** (~5,7 GB comprimido, ~6 GB extraído):
   - `https://github.com/Junyi136/Instruct-Edit/releases/download/vv1.0/Instruct_CelebA_Dataset_2.zip.001`
   - `https://github.com/Junyi136/Instruct-Edit/releases/download/vv1.0/Instruct_CelebA_Dataset_2.zip.002`
   - `https://github.com/Junyi136/Instruct-Edit/releases/download/vv1.0/Instruct_CelebA_Dataset_2.zip.003`
2. **CelebAMask-HQ** (~2–3 GB): imágenes originales 512×512 necesarias para
   emparejar con los `face_id` de Instruct-CelebA. Descargar solo la carpeta
   `CelebA-HQ-img/`.

#### 4.5.2 Procesamiento

Crear `src/scripts/prepare_instruct_celeba.py` que:

1. Una los 3 archivos `.zip.00x` y extraiga el zip interior.
2. Recorra la estructura:
   ```
   instruct_dataset/train/<attribute>/<face_id>/instruct.json
   instruct_dataset/train/<attribute>/<face_id>/<face_id>_<value>.jpg
   ```
3. Para cada `face_id`, cargue la imagen original desde
   `data/raw/celebamask_hq/CelebA-HQ-img/<face_id>.jpg`.
4. Genere tripletes `(original_image, edited_image, edit_prompt)`.
5. Aplique un filtro de calidad facial (descartar pares donde la cara no se
   detecte o se deforme gravemente).
6. Submuestree estratificado a **~20.000 ejemplos**, dando peso a atributos
   cercanos al problema:

| Atributo | Pares disponibles | Muestra objetivo |
|---|---|---|
| `glasses` | ~8.588 | **todo** |
| `hair` | ~40.137 | 5.000 |
| `beard` | ~4.004 | **todo** |
| `eyes` | ~7.676 | **todo** |
| `age` | ~37.894 | 3.000 |
| `gender` | ~20.532 | 2.000 |
| `expression` | ~27.947 | 2.000 |
| `skin` | ~15.204 | 1.000 |
| `anime` | ~19.798 | 0 (dominio muy diferente; omitir) |
| **Total aprox.** | — | **~20.000** |

Salida: `data/processed/instruct_celeba/` como `DatasetDict` con split
`train` (no se usa validation de Instruct-CelebA; el validation de v3 sigue
siendo el de MagicBrush para comparabilidad).

### 4.6 Filtro de calidad facial

Aplicar en **ambos** datasets (MagicBrush e Instruct-CelebA):

- Detectar cara en `original_image` y `edited_image`.
- Descartar el par si:
  - Hay cara en la original pero no en la editada, **o**
  - La confianza de detección en la editada es muy baja, **o**
  - La proporción de la cara cambia drásticamente.

Se puede usar `opencv-python` + Haar/SSD o cualquier detector ya disponible.
El objetivo no es exigir perfección, sino **descartar los casos donde la cara
queda irreconocible**, porque esos son los que el modelo aprende a reproducir.

### 4.7 Construir la mezcla v3

Crear `src/scripts/prepare_v3_mix.py` que:

1. Cargue `data/processed/magicbrush/`.
2. Aplique el filtro de tiras de §4.3.
3. Aplique el filtro de pares ruidosos de §4.4.
4. Aplique el filtro de calidad facial de §4.6.
5. Sobremuestree **personas ×2** en MagicBrush (igual que v2).
6. Cargue `data/processed/instruct_celeba/`.
7. Concatene ambos datasets.

El split `validation` se copia **sin cambios** desde MagicBrush, idéntico al
de v1/v2.

Comando:

```bash
python src/scripts/prepare_v3_mix.py \
  --magicbrush_dir data/processed/magicbrush \
  --instruct_celeba_dir data/processed/instruct_celeba \
  --output_dir data/processed/magicbrush_v3 \
  --person_repeat 2
```

**Entregables**:

- `data/processed/magicbrush_v3/` listo para entrenar.
- `stats.json` con conteos originales, filtrados y finales.
- `pytest tests/test_prepare_v3_mix.py -q` verde.

### 4.8 Verificación del mix

```bash
python -c "from datasets import load_from_disk; d=load_from_disk('data/processed/magicbrush_v3'); print(d)"
cat data/processed/magicbrush_v3/stats.json
```

- Train final esperado: ~25.000–30.000 ejemplos (~8k MagicBrush filtrado +
  ~20k Instruct-CelebA).
- Validation: **528 exactos**.

---

## 5. Fase 2 — Entrenamiento (pod, ~7–9 h, ~$10–16)

### 5.1 Especificación exacta

| Parámetro | Valor v3 | Cambio vs v2 | Justificación |
|---|---|---|---|
| Punto de partida | `models/checkpoints/grafito-magicbrush-v2` | nuevo | v2 es una base sólida; arrancar desde él conserva nitidez 512 y eliminación del tinte magenta. |
| Resolución | 512 | igual | Objetivo local es 512 px; Instruct-CelebA también está a 512. |
| Dataset | `data/processed/magicbrush_v3` | nuevo | MagicBrush filtrado + sobremuestreo personas ×2 + Instruct-CelebA submuestreado a ~20k. |
| `conditioning_dropout_prob` | 0.10 | igual | v2 ya encontró equilibrio; no tocar sin evidencia. |
| Pasos máximos | 10000 | +67 % | El dataset es mayor (~28k ejemplos); 10000 pasos ≈ 5.7 épocas, suficiente para aprovechar Instruct-CelebA sin sobreajustar. |
| Batch / acumulación | 2 / 8 (efectivo 16) | igual | Probado en v2, cabe en 24 GB. |
| Learning rate / scheduler | 5e-5 / constant | igual | Funcionó en v1 y v2. |
| Warmup | 0 | igual | |
| Precisión | fp16 mixed | igual | |
| Gradient checkpointing | activado | igual | |
| `max_grad_norm` | 1.0 | igual | |
| Seed | 42 | igual | |
| Checkpointing | cada 1000, conservar 2 | igual | |
| Text encoder / VAE | congelados | igual | |

### 5.2 Comando de entrenamiento

```bash
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

Durante el run, cada ~1000 pasos generar smokes obligatorios sobre retrato:

```bash
python scripts/test_checkpoint.py \
  --checkpoint models/checkpoints/grafito-v3 \
  --image assets/foto.jpg --prompt "remove his glasses" \
  --resolution 512 --seed 0 --output outputs/smoke_remove.jpg

python scripts/test_checkpoint.py \
  --checkpoint models/checkpoints/grafito-v3 \
  --image assets/foto.jpg --prompt "replace the red cap with a blue beanie" \
  --resolution 512 --seed 0 --output outputs/smoke_replace.jpg
```

### 5.3 Pre-flight obligatorio en el pod

Antes de lanzar el entrenamiento:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# Esperado exactamente: 2.2.2 12.1 True

pip install -r requirements.txt
pytest tests/ -q

# Smoke de entrenamiento (5 pasos, ~2 min)
accelerate launch --mixed_precision="fp16" src/scripts/train_instruct_pix2pix.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=512 --train_batch_size=2 --gradient_accumulation_steps=8 \
  --gradient_checkpointing --max_train_steps=5 --learning_rate=5e-5 \
  --mixed_precision=fp16 --output_dir=/tmp/smoke_v3
```

Si algo falla: destruir la instancia y elegir otra. No entrenar sobre un pod
roto.

### 5.4 Infraestructura del pod

- **Imagen**: `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`.
- **GPU**: RTX 3090 / RTX 4090 / A10 / A100 ≥ 24 GB.
- **Driver**: CUDA ≥ 12.1 en el host.
- **Disco local**: ≥ 100 GB (dataset más grande: ~6 GB Instruct-CelebA +
  ~3 GB CelebAMask-HQ + ~6 GB MagicBrush + checkpoints).
- **Sin Network Volume**.
- **Alarma de gasto**: `$20`.
- **No spot**: el run es de una sola ejecución.

---

## 6. Fase 3 — Evaluación (pod)

### 6.1 Evaluación cuantitativa conjunta a 512 px

La deuda técnica de v2 fue no medir v1 a 512. En v3 se cierra:

1. Subir al pod:
   - `models/checkpoints/grafito-magicbrush-v2` (desde local).
   - `models/checkpoints/grafito-magicbrush` (v1; desde Drive, Drive→pod es
     rápido).
2. Evaluar los cuatro modelos con el **mismo** arnés, seeds y resolución:

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

3. Extraer la tabla comparativa y anotarla en `docs/EVALUATION.md`.

### 6.2 Panel cualitativo fijo

8 casos × 2 seeds, mismos prompts y seeds para v1, v2 y v3:

1. Estatua `add a hat` (regresión de v1).
2. Retrato `add a black hat` (cara + objeto).
3. Retrato `make the background blue` (adherencia fina).
4. Interior doméstico (tinte magenta, heredado de v1).
5. Objeto simple (caso típico MagicBrush).
6. Paisaje / escena amplia.
7. Imagen con bordes claros (manchas de borde).
8. Edición de estilo `make it look like a painting`.

Más los **casos críticos de `review/`** con la imagen original del retrato:

9. `remove his glasses`
10. `replace the red cap with a blue beanie`
11. `make the background blue` sobre el mismo retrato.

Rúbrica 1–5 por caso: fidelidad facial, adherencia al prompt, artefactos, cast
de color. Resultados en `docs/EVALUATION.md`.

---

## 7. Cronograma y coste estimado

| Fase | Duración estimada | Coste |
|---|---|---|
| 0.1–0.3 Puertas locales | 0.5–1 h | $0 |
| 4.1–4.4 Preparar MagicBrush filtrado | 2–3 h | $0 |
| 4.5 Descargar y procesar Instruct-CelebA + CelebAMask-HQ | 2–3 h | $0 |
| 4.6–4.8 Mezcla v3 y tests | 1–2 h | $0 |
| Setup del pod + subir v1/v2 + dataset | 1–2 h | ~$0–1 |
| Entrenamiento 10000 pasos (RTX 4090, ~2.8 s/it) | ~7.5–8.5 h | ~$5–7 |
| RTX 3090 (~4.0 s/it) | ~11–12 h | ~$6–10 |
| Evaluación conjunta a 512 px (4 modelos × 528 ej.) | 1–1.5 h | ~$0.5–1 |
| Descarga, verificación local y destrucción | 1 h | $0 |
| **Total** | **~16–22 h** | **~$10–16** |

**Techo de gasto**: `$20` con alarma.  
**Regla de oro**: si el coste acumulado pasa de `$15` sin que el run haya
terminado, detener y evaluar si continuar.

---

## 8. Riesgos y mitigaciones

| Riesgo | Prob. | Mitigación |
|---|---|---|
| Instruct-CelebA no mejora los fallos catastróficos | Media | Si el panel de quitar/reemplazar no llega a 4/5, rollback a v2. |
| Overfit a caras por el peso de Instruct-CelebA | Media | Submuestrear a ~20k y mantener MagicBrush como base; panel con casos no-persona. |
| Instruct-CelebA arrastra artefactos sintéticos | Media | Filtro de calidad facial; si persisten, reducir la muestra de Instruct-CelebA. |
| El filtro de calidad facial descarta señal útil | Baja | Umbrales conservadores; revisión manual de una muestra. |
| El filtro de pares ruidosos elimina señal útil | Media | Umbrales conservadores; revisión manual de muestras descartadas. |
| No se alcanza LPIPS ≤ 0.18 | Media | 10000 pasos + datos más limpios; si no baja de 0.20, se conserva v2. |
| Manchas de esquina persistentes | Media | Si siguen, aceptar mitigación por recorte y no hacer v4 solo por eso. |
| Interrupción del pod | Baja | Checkpoints cada 1000; reanudar con `--resume_from_checkpoint` el mismo día. |
| OOM a 512 con batch 2 | Baja | Bajar batch a 1 y subir acumulación a 16 (mismo efectivo 16). |
| v3 no supera a v2 | Media | Criterios de §1.2: rollback a v2; coste queda acotado a ~$16. |

---

## 9. Checklist de ejecución

- [x] Fallos de `review/` reproducidos con v2 y guardados en `outputs/`.
- [x] Conteos de MagicBrush confirmados (remove/replace en personas < 100).
- [x] Instruct-CelebA descargado y procesado en
      `data/processed/instruct_celeba/`.
- [x] CelebAMask-HQ disponible para emparejar originales.
- [x] `data/processed/magicbrush_v3/` generado y `stats.json` revisado.
- [x] `pytest tests/ -q` pasa en local.
- [ ] Pod con `torch 2.2.2+cu121` verificado.
- [ ] Smoke de entrenamiento de 5 pasos con `EXIT_0`.
- [ ] v1 y v2 disponibles en el pod.
- [ ] Entrenamiento v3 completado (`max_train_steps=10000`).
- [ ] Evaluación conjunta a 512 px ejecutada para los 4 modelos.
- [ ] Panel cualitativo generado, incluyendo los casos críticos de `review/`.
- [ ] Panel quitar/reemplazar en retratos ≥ 4/5.
- [ ] Checkpoint ganador bajado a `models/checkpoints/grafito-magicbrush-v3/`.
- [ ] Verificación local a 512 px (CPU/MPS) superada.
- [ ] Instancia del pod destruida el mismo día.
- [ ] Decisión de adopción documentada en `docs/CHANGELOG.md` y
      `docs/EVALUATION.md`.

---

## 10. Criterios de adopción y rollback

### 10.1 Adoptar v3

Se adopta v3 si **todas** estas condiciones se cumplen:

- Panel quitar/reemplazar en retratos ≥ **4/5** (casos `remove his glasses` y
  `replace the red cap with a blue beanie`).
- LPIPS v3 a 512 ≤ **0.18**.
- LPIPS v3 < LPIPS v2.
- CLIP v3 ≥ **0.25** y CLIP v3 ≥ CLIP v2 − 0.005.
- Panel caras general ≥ 4/5.
- Panel adherencia fina ≥ 4/5.
- Ninguna mancha de esquina en el panel sin recorte.

### 10.2 Rollback a v2

Si falla cualquiera de las condiciones anteriores, especialmente el panel de
quitar/reemplazar en retratos:

- Conservar `models/checkpoints/grafito-magicbrush-v2` como producción.
- No reemplazar la demo.
- Documentar en `docs/EVALUATION.md` por qué v3 no superó a v2.
- El coste queda acotado al run (~$10–16).

### 10.3 Publicación

Siguiendo la decisión de 2026-07-27, **no publicar en Hugging Face** hasta que
v3 (o la versión que quede adoptada) esté estabilizada por uso real del demo.

---

## 11. Notas de seguimiento

- Fecha de aprobación del plan: ejecutado el 2026-07-28
- Fallos críticos confirmados: `remove his glasses`, `replace the red cap with a blue beanie`
- Dataset facial añadido: Instruct-CelebA submuestreado a ~20k ejemplos
- Coste real del run: ~11 h de RTX 4090 en Vast.ai (preparación + 7 h 11 min de run + evaluación), dentro del techo de $20
- Métricas finales (LPIPS / CLIP): v3 0.2329 / 0.2511 — v2 0.2405 / 0.2509 — base 0.3046 / 0.2512 (528 ejemplos a 512 px, misma corrida)
- Panel quitar/reemplazar en retratos: **suspenso** — `remove his glasses` ❌ 0/2 (sustituye la escena), `make the background light blue` ❌ 0/2 (elimina al sujeto, regresión frente a v2); `add a black hat` ✅ 2/2, `make his jacket bright yellow` ✅ 2/2
- Decisión final (adoptar v3 / conservar v2): **conservar v2 (rollback, §10.2)** — 2026-07-28
