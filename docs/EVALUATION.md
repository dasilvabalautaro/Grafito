# Protocolo de evaluación

## Conjuntos de evaluación

1. **MagicBrush validation:** 528 ejemplos (automático).
2. **Pruebas manuales:** 20–30 ejemplos representativos seleccionados del validation set.
3. **Stress tests:** 5 casos difíciles (fondos complejos, cambios sutiles, múltiples objetos).

## Métricas

| Métrica | Descripción | Cómo calcular |
|---|---|---|
| LPIPS vs target | Distancia perceptual entre la imagen generada y la imagen objetivo (menor mejor). | `lpips.LPIPS` sobre pares generado/target |
| CLIP similarity | Similitud entre el prompt de edición y la imagen generada (mayor mejor). | `CLIP(image_generated) · CLIP(prompt)` |
| LPIPS región no editada | Distancia perceptual en zona no editada (menor mejor). | Futuro: `LPIPS(original, edited)` en región negra de `mask_img` |
| Face similarity | Conservación de identidad facial. | Embedding de modelo de reconocimiento facial (futuro) |

## Script de evaluación

`src/scripts/evaluate.py` genera automáticamente el reporte comparando el checkpoint entrenado con `timbrooks/instruct-pix2pix`.

### Instalar dependencias de evaluación

```bash
pip install -e ".[eval]"
```

O manualmente:

```bash
pip install lpips>=0.1.4 scikit-image>=0.22.0
pip install git+https://github.com/openai/CLIP.git
```

### Ejecutar evaluación

```bash
python src/scripts/evaluate.py \
  --checkpoint models/checkpoints/grafito-magicbrush \
  --dataset data/processed/magicbrush \
  --split validation \
  --num_samples 100 \
  --output outputs/eval_report.json
```

Para evaluar todo el split de validation, omití `--num_samples`.

## Protocolo

1. Generar ediciones con el checkpoint entrenado y con `timbrooks/instruct-pix2pix` (línea base).
2. Usar los mismos seeds y parámetros de inferencia en ambos.
3. Calcular métricas sobre MagicBrush validation.
4. Realizar revisión visual mezclando resultados.
5. Ajustar `image_guidance_scale` y `guidance_scale` si es necesario.

## Criterio de aprobación provisional

El checkpoint se aprueba si:

- Mejora la línea base en LPIPS vs target.
- Mejora o mantiene CLIP similarity.
- Pasa la mayoría de pruebas cualitativas manuales.

## Resultados — Primer entrenamiento Grafito

Fecha: 2026-07-25
Checkpoint: `models/checkpoints/grafito-magicbrush`
Dataset: `osunlp/MagicBrush` validation (528 ejemplos)
Configuración: 10 000 steps, resolución 256, batch 4, acumulación 4, lr 5e-5.

| Métrica | Checkpoint entrenado | Línea base (`timbrooks/instruct-pix2pix`) | Mejora |
|---|---|---|---|
| LPIPS vs target (menor mejor) | 0.1997 | 0.3316 | **+0.1319** (~40% relativo) |
| CLIP similarity (mayor mejor) | 0.2476 | 0.2591 | -0.0114 (~4% relativo) |

### Interpretación

- **LPIPS mejoró sustancialmente**: el modelo afinado genera imágenes perceptualmente mucho más cercanas al target de MagicBrush.
- **CLIP se mantuvo casi igual**: ligera pérdida en alineación textual, dentro del margen ruido.

### Fortalezas observadas

El modelo entrenado es especialmente superior en ediciones de **agregar o reemplazar objetos**, por ejemplo:

- "Make the piece of paper hanging on the wall a mirror"
- "Have there be a dolphin jumping out of the water"
- "replace the baseball bat for a laser sword"
- "Change the hat to a cowboy hat"

### Debilidades observadas

El modelo base a veces es más conservador y funciona mejor en cambios sutiles de **color o textura**, por ejemplo:

- "Make it a black sheep"
- "remove the yellow flowers"
- "let the cat be white"

### Próximos experimentos

- Probar `conditioning_dropout_prob=0.05` para mejorar fidelidad al prompt.
- Evaluar con menos steps (5000-7000) para reducir sobre-ediciones en cambios sutiles.
- Añadir métrica de LPIPS en región no editada usando `mask_img`.

## Resultados — Segundo entrenamiento (v2)

Fecha: 2026-07-27
Checkpoint: `models/checkpoints/grafito-magicbrush-v2` (**adoptado** para el demo)
Dataset: MagicBrush validation (528 ejemplos, idéntico al de v1)
Configuración: 6000 steps a 512 px, batch 2, acumulación 8, lr 5e-5, `conditioning_dropout_prob=0.1`, mezcla con sobremuestreo ×2 de personas (10099 ejemplos) y filtro de 44 tiras de calibración. 4 h 40 min en RTX 4090.

### Métricas a 512 px (resolución nativa de v2)

| Métrica | v2 a 512 | Base a 512 | Mejora |
|---|---|---|---|
| LPIPS vs target (menor mejor) | 0.2405 | 0.3208 | **+0.0803** (~25% relativo) |
| CLIP similarity (mayor mejor) | 0.2509 | 0.2523 | -0.0013 (empate) |

Referencia cruzada (resoluciones distintas, comparabilidad limitada): v1 a 256 obtuvo LPIPS 0.1997 / CLIP 0.2476; el mismo arnés a 256 con v2 da LPIPS 0.3643 / CLIP 0.2457 — v2 está entrenado para 512 y a 256 queda fuera de su distribución. CLIP de v2 supera al de v1 (0.2509 > 0.2476).

### Panel cualitativo (8 casos × 2 seeds, `outputs/eval_v2/panel/`)

- **Caras: problema de v1 resuelto.** Boca y ojos preservados en retratos (v1 los derretía).
- **Tinte magenta: eliminado** en las 16 muestras (en v1 era sistemático en interiores).
- **Nitidez 512 real** frente a la suavidad de 256.
- **Persisten:** manchas de color esporádicas en esquinas (el filtro de tiras redujo pero no erradica), adherencia fina irregular (ej.: `make the background blue` tiñó la camiseta), texto en imágenes ilegible (limitación de SD 1.5).

### Decisión

v2 adoptado para el demo (2026-07-27). Cumple los objetivos prioritarios del plan (caras, tinte, nitidez) sin regresión de CLIP. La meta estricta de LPIPS ≤ 0.18 no se alcanzó; la comparación con v1 queda confundida por el cambio de resolución (v1 a 512 nunca se midió).

## Herramientas

- `lpips`
- `clip` (OpenAI)
- `scikit-image` (para métricas futuras)
