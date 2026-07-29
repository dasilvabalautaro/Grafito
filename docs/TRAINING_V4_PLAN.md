# Plan de entrenamiento v4 — Grafito: menos pasos sobre el dataset v3

Fecha: 2026-07-29.
Estado: **plan detallado para aprobación**. Se activa por la lección
documentada en `docs/EVALUATION.md` (sección v3): los smokes intermedios del
run v3 mostraron sobre-edición creciente con los pasos.

---

## 0. Decisión de activación (go/no-go)

El run v3 (v2 + 10000 pasos sobre `magicbrush_v3`) terminó en rollback a v2
porque:

- `remove his glasses` siguió fallando (Instruct-CelebA **no** corrigió la
  eliminación de objetos).
- `make the background light blue` **regresó**: v3 elimina al sujeto, algo que
  v2 hacía bien.

Pero los smokes sobre checkpoints intermedios del propio run v3 mostraron que
el modelo pasa por un punto mejor antes de sobre-ajustar:

- En **checkpoint-5000**, el caso `replace` conservaba la chaqueta.
- En **checkpoint-7000**, la chaqueta ya se cambiaba sin pedirlo.

Esos checkpoints intermedios se descartaron (`checkpoints_total_limit=2`), así
que recuperar ese punto exige un run nuevo. v4 es ese run: **la misma receta
de v3, detenida en 6000 pasos, guardando y evaluando los intermedios**.

**Lo que v4 NO promete:** arreglar `remove his glasses`. La evidencia de v3
dice que ni Instruct-CelebA ni más pasos lo resuelven; queda como limitación
conocida.

---

## 0.5 Puerta previa obligatoria: auditoría de datos de Instruct-CelebA

Antes de alquilar GPU, el componente Instruct-CelebA del dataset v3 debe pasar
una auditoría de calidad (principio data-centric: si los datos fallan, ningún
ajuste de pasos lo compensa). La mezcla conserva la procedencia (`face_id` /
`attribute` no nulos = Instruct-CelebA), así que la auditoría se hace en local
sobre `data/processed/magicbrush_v3` sin re-descargas, a coste $0.

Chequeos (script `src/scripts/audit_v3_data.py`):

1. **Emparejamiento original↔editada** (el sospechoso principal): el pipeline
   asumió que el índice de `v-xchen-v/celebamask_hq` coincide con `face_id`.
   Si no, los pares enseñan "sustituir una persona por otra", coherente con el
   fallo de v3. Se mide con similitud CLIP imagen-imagen por par y revisión
   visual de láminas.
2. **Magnitud de la edición**: distribución de LPIPS(original, editada).
   Valores muy altos = reemplazo de escena, no edición.
3. **Adherencia prompt↔editada**: similitud CLIP texto-imagen por atributo.
4. **Prompts**: distribución por atributo y muestra de prompts vagos.
5. **Revisión visual**: láminas de contacto (original | editada | prompt) por
   atributo en `outputs/audit_v3/` para inspección humana.

**Criterio de la puerta:**

- Si el emparejamiento es correcto y la fracción de pares defectuosos es baja,
  v4 sigue tal cual (mismo dataset, menos pasos).
- Si hay una fracción relevante de pares mal emparejados o ediciones rotas, se
  filtran, se reconstruye la mezcla (`magicbrush_v4`) y v4 pasa a tener dos
  cambios documentados (datos limpios + pasos). La hipótesis de pasos se
  evalúa igualmente con los checkpoints intermedios.
- Si el emparejamiento está sistemáticamente roto, se descarta Instruct-CelebA
  por completo y se reevalúa el diseño de v4 antes de gastar un dólar.

**Resultado de la auditoría (2026-07-29, `outputs/audit_v3/`): emparejamiento
sistemáticamente roto.** Sobre 100 pares por atributo (800 en total) con 200
pares de control de MagicBrush:

| Métrica | Instruct-CelebA (8 atributos) | Control MagicBrush |
|---|---|---|
| Similitud CLIP img-img (emparejamiento) | 0.54–0.59 | **0.94** |
| Pares bajo umbral de emparejamiento (<0.70) | **90–98 %** | 0.5 % |
| LPIPS original↔editada (magnitud) | 0.53–0.56 | **0.11** |
| Pares con reemplazo de escena (>0.35) | **~100 %** | 5.5 % |

La revisión visual de `sheet_glasses.png` lo confirma: original y editada son
**personas distintas** (distinta edad, género y etnia). La causa está en
`_cache_celebamask_hq_images` de `src/scripts/prepare_instruct_celeba.py`:
asumió que el índice posicional del dataset `v-xchen-v/celebamask_hq` coincide
con el `face_id` de Instruct-CelebA, y no es así.

Esto explica retroactivamente el fallo de v3: el modelo aprendió de ~19k
pares "sustituir una persona por otra", exactamente lo que hacía en
`remove his glasses` y `make the background light blue`.

**Re-emparejamiento y re-auditoría (2026-07-29): puerta SUPERADA.** Se
corrigió `prepare_instruct_celeba.py` (emparejamiento por nombre de archivo
con `CelebA-HQ-img/` real de `liusq/CelebAMask-HQ`; el fallback posicional se
eliminó y el script aborta si falta >1 % de originales). Nuevo dataset:
19.755 ejemplos, 0 originales faltantes. Nueva mezcla
`data/processed/magicbrush_v4/` (27.400 train = 7.645 MagicBrush ×2 personas
+ 19.755 Instruct-CelebA; validation 528 idéntico). Re-auditoría
(`outputs/audit_v4/`):

| Métrica | Instruct-CelebA v4 | Control MagicBrush | Instruct-CelebA v3 (roto) |
|---|---|---|---|
| Similitud CLIP img-img | **0.85–0.96** | 0.95 | 0.54–0.59 |
| Mal emparejamiento (<0.70) | **0 %** (gender 9 %, esperable) | 0 % | 90–98 % |
| LPIPS original↔editada | **0.02–0.14** | 0.11 | 0.53–0.56 |
| Reemplazo de escena (>0.35) | **0 %** | 2 % | ~100 % |

Las láminas confirman visualmente misma persona y edición coherente con el
prompt. `gender` marca 9 % en emparejamiento porque cambiar la presentación
de género altera mucho la cara (es la edición pedida, no un error).

**Consecuencia para v4:** el run usa `magicbrush_v4`. La hipótesis de pasos
se evalúa como estaba previsto (6000 pasos, intermedios 4000/5000/6000), pero
ahora sobre datos limpios; si la regresión de fondos desaparece ya en los
intermedios, habrá confirmado que la causa era el dataset envenenado.

---

## 1. Objetivos y criterios de éxito

### 1.1 Hipótesis

> Entrenando desde v2 sobre el dataset v4 (Instruct-CelebA ya re-emparejado),
> el óptimo de conservación está en ~4000–6000 pasos, no en 10000. Detener
> ahí debería conservar la ganancia de LPIPS de v3 sin la regresión de fondos
> ni la sobre-edición — y con datos limpios, la regresión de fondos no debería
> aparecer en absoluto si su causa era el emparejamiento roto.

### 1.2 Métricas y umbrales

Referencia a 512 px (misma corrida, 528 ejemplos): v2 LPIPS 0.2405 / CLIP
0.2509 · v3 LPIPS 0.2329 / CLIP 0.2511 · base LPIPS 0.3046 / CLIP 0.2512.

| Criterio | Umbral v4 |
|---|---|
| LPIPS vs target | **< 0.2405** (mejor que v2; idealmente ≤ 0.2329 de v3) |
| CLIP similarity | **≥ 0.25** y ≥ v2 − 0.005 |
| `make the background light blue` (retrato) | **sujeto preservado en 2/2 seeds** (la regresión de v3 debe desaparecer) |
| `make his jacket bright yellow` | ✅ 2/2 (sin sobre-edición: solo la chaqueta cambia) |
| `add a black hat` | ✅ 2/2 |
| Panel general (rúbrica 1–5) | ≥ 4/5, sin regresión frente a v2 |

Regla de adopción: se adopta el **mejor checkpoint intermedio** (4000, 5000 o
6000) que cumpla la tabla. Si ninguno la cumple, rollback a v2 y se documenta.

---

## 2. Principios (heredados de v3)

- **Cambios respecto a v3**: los pasos (10000 → 6000), la retención de
  checkpoints (sin límite, para evaluar intermedios) y el dataset
  (`magicbrush_v3` → `magicbrush_v4`, mismo diseño pero con Instruct-CelebA
  re-emparejado y auditado; ver §0.5). El cambio de datos no es opcional:
  la auditoría demostró que v3 estaba envenenado. Resolución, lr, batch
  efectivo, dropout y seed, idénticos.
- **Techo de gasto duro**: `$15` con alarma (el run es más corto que v3).
- **Sin Network Volume**, instancia destruida el mismo día.
- **Evaluación en el pod antes de bajar nada**.

---

## 3. Especificación exacta

| Parámetro | Valor v4 | Cambio vs v3 | Justificación |
|---|---|---|---|
| Punto de partida | `models/checkpoints/grafito-magicbrush-v2` | igual | Réplica directa del hallazgo del smoke. |
| Dataset | `data/processed/magicbrush_v4` | **nuevo** | Misma composición que v3 pero con Instruct-CelebA re-emparejado y auditado (§0.5). |
| Pasos máximos | **6000** | −40 % | El smoke ubicó el punto bueno en 5000 y el malo en 7000; 6000 da margen y permite comparar 4000/5000/6000. |
| Checkpointing | cada 1000, **sin límite** (`--checkpoints_total_limit` omitido) | nuevo | Conservar 1000…6000 para evaluar intermedios; es la pregunta central de v4. |
| Resolución / batch / acumulación | 512 / 2 / 8 (efectivo 16) | igual | Probado en v2 y v3. |
| Learning rate / scheduler | 5e-5 / constant, warmup 0 | igual | La lección apunta a pasos, no a lr; no tocar dos cosas a la vez. |
| `conditioning_dropout_prob` | 0.10 | igual | |
| Precisión / grad-ckpt / seed | fp16 / activado / 42 | igual | |
| Text encoder / VAE | congelados | igual | |

Nota: al partir de v2, el paso 6000 de v4 equivale a ~12000 pasos acumulados
desde el modelo base. La hipótesis es que lo que importa no es el acumulado
sino cuántos pasos se dan sobre el dataset v3 (el smoke de v3 lo vio a
5000/7000 de este run).

---

## 4. Comandos

### 4.1 Entrenamiento (pod, en `tmux`)

Mismo pre-flight de `docs/V3_RUNBOOK.md` Fase B (torch 2.2.2+cu121, tests
verdes, smoke de 5 pasos). Luego:

```bash
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

Duración estimada: ~4.5–5 h en RTX 4090 (2.8 s/it), ~6.5–7 h en RTX 3090.

Nota operativa (de v3): los checkpoints guardados por el script no incluyen
`safety_checker`; hay que copiarlo del modelo base en cada `checkpoint-N`
antes de cargarlos con `from_pretrained` en smokes y evaluación.

### 4.2 Smokes de retrato durante el run

Igual que en v3 (`docs/V3_RUNBOOK.md` B8), sobre los checkpoints
`checkpoint-3000`, `checkpoint-5000` y el final, con los casos de `review/`:

- `remove his glasses` (se reporta, pero no es criterio de adopción).
- `replace the red cap with a blue beanie`.
- Añadir: `make the background light blue` — es la regresión a vigilar.

Si un checkpoint intermedio claramente supera al final, se puede parar el run
antes: ese es exactamente el comportamiento que v4 busca explotar.

### 4.3 Evaluación cuantitativa (pod)

Nota operativa: `data/processed/magicbrush` ya no existe en local (se limpió);
el split `validation` de `data/processed/magicbrush_v4` es idéntico (528
ejemplos), así que el arnés usa ese dataset con `--split validation`.

Los `checkpoint-N` son estados de entrenamiento (solo traen `unet/`), no
pipelines cargables. Antes de evaluar hay que **materializarlos** como
`grafito-v4-step<N>` copiando los componentes del pipeline desde v2 (comando
`materialize` en `docs/V4_RUNBOOK.md`, Fase C1).

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

(`grafito-v4` sin sufijo = pipeline final, equivalente a checkpoint-6000. Los
checkpoints intermedios de v3 no existen; la comparación de pasos es interna
a v4.)

### 4.4 Panel cualitativo

Mismos casos que el panel v3 (`docs/EVALUATION.md`, sección v3), mismas seeds,
para los checkpoints 4000/5000/6000 y v2:

1. `remove his glasses` (informativo).
2. `make the background light blue` (retrato) — **criterio clave**.
3. `make his jacket bright yellow` — sobre-edición.
4. `add a black hat` (retrato).
5. Estatua `add a hat`.
6. Estatua `make it look like a painting` (informativo).

---

## 5. Cronograma y coste estimado

| Fase | Duración | Coste |
|---|---|---|
| Setup del pod + subir dataset v3 y checkpoints v2/v3 | 1–1.5 h | ~$0.5–1 |
| Entrenamiento 6000 pasos (RTX 4090) | ~4.5–5 h | ~$3–4 |
| Evaluación (5 configuraciones × 528 ej.) | ~1.5 h | ~$1 |
| Descarga, verificación local, destrucción | 1 h | $0 |
| **Total** | **~8–9.5 h** | **~$5–7** |

Techo: `$15` con alarma. Regla de oro: si se pasa de `$10` sin terminar,
parar y decidir.

---

## 6. Riesgos y mitigaciones

| Riesgo | Prob. | Mitigación |
|---|---|---|
| El punto bueno del smoke no se reproduce | Media | Se evalúan 4000/5000/6000; si ninguno gana, rollback a v2 y la pregunta queda respondida por ~$6. |
| 6000 pasos ya sobre-editan como 10000 | Baja | Los intermedios dan la curva; el mejor se adopta aunque no sea el último. |
| `remove his glasses` sigue fallando | Alta | Aceptado por diseño: no es criterio de adopción en v4. |
| La regresión de fondo viene del dataset (Instruct-CelebA), no de los pasos | Media | Si 4000/5000/6000 eliminan al sujeto igual que v3, la hipótesis de pasos queda refutada y el siguiente experimento sería un mix sin Instruct-CelebA. Documentar y no insistir. |
| OOM a 512 con batch 2 | Baja | Batch 1 + acumulación 16 (mismo efectivo). |
| Interrupción del pod | Baja | Checkpoints cada 1000; reanudar con `--resume_from_checkpoint` el mismo día. |

---

## 7. Checklist de ejecución

- [ ] Commit y push del repo (este plan incluido) antes de alquilar el pod.
- [ ] Pod con `torch 2.2.2+cu121` verificado y `pytest tests/ -q` verde.
- [ ] Smoke de entrenamiento de 5 pasos con `EXIT_0`.
- [ ] Dataset v3 y checkpoints v2/v3 subidos al pod.
- [ ] Run v4 completado (6000 pasos, intermedios conservados).
- [ ] Evaluación cuantitativa de las 5 configuraciones.
- [ ] Panel cualitativo de 4000/5000/6000 vs v2.
- [ ] Checkpoint ganador bajado a `models/checkpoints/grafito-magicbrush-v4/`.
- [ ] Verificación local a 512 px (CPU/MPS) superada.
- [ ] Instancia destruida el mismo día.
- [ ] Decisión documentada en `docs/EVALUATION.md` y `docs/CHANGELOG.md`.

---

## 8. Criterios de adopción y rollback

### 8.1 Adoptar v4 (el mejor intermedio)

Se adopta si algún checkpoint de v4 cumple **toda** la tabla de §1.2, en
particular: sujeto preservado en `make the background light blue` y LPIPS
mejor que v2.

### 8.2 Rollback a v2

Si ningún checkpoint la cumple: v2 sigue en producción, se documenta la curva
LPIPS/panel vs pasos en `docs/EVALUATION.md` (es información útil aunque sea
negativa) y el coste queda acotado a ~$5–7.
