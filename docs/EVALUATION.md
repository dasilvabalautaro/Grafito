# Protocolo de evaluación

## Conjuntos de evaluación

1. **MagicBrush dev:** 528 ejemplos (automático).
2. **Pruebas manuales:** 20–30 ejemplos representativos seleccionados del dev set.
3. **Stress tests:** 5 casos difíciles (fondos complejos, cambios sutiles, múltiples objetos).

## Métricas

| Métrica | Descripción | Cómo calcular |
|---|---|---|
| CLIP directional similarity | El cambio entre original y editada coincide con la instrucción. | `CLIP(image_edit - image_orig) · CLIP(prompt)` |
| SSIM región no editada | Conservación de píxeles fuera de la máscara. | `SSIM(original, edited)` en región negra de `mask_img` |
| LPIPS región no editada | Distancia perceptual en zona no editada (menor mejor). | `LPIPS(original, edited)` en región negra de `mask_img` |
| Face similarity | Conservación de identidad facial. | Embedding de modelo de reconocimiento facial |

## Protocolo

1. Generar ediciones con el checkpoint entrenado y con `timbrooks/instruct-pix2pix` (línea base).
2. Usar los mismos seeds y parámetros de inferencia en ambos.
3. Calcular métricas sobre MagicBrush dev.
4. Realizar revisión visual ciega mezclando resultados.
5. Ajustar `image_guidance_scale` y `guidance_scale` si es necesario.

## Criterio de aprobación

El checkpoint se aprueba si:

- Mejora la línea base en CLIP directional similarity en ≥ 3 puntos porcentuales.
- Mejora o mantiene SSIM en región no editada.
- Pasa 4 de 5 pruebas cualitativas manuales.

## Herramientas

- `lpips`
- `scikit-image`
- `clip` (OpenAI)
- `insightface` o similar para face similarity (opcional)
