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

## Herramientas

- `lpips`
- `clip` (OpenAI)
- `scikit-image` (para métricas futuras)
