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

## Herramientas

- `lpips`
- `clip` (OpenAI)
- `scikit-image` (para métricas futuras)
