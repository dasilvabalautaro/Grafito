# Plan del siguiente paso — Demo Grafito

## Estado actual del proyecto

### Entrenamiento completado
- **Checkpoint**: `models/checkpoints/grafito-magicbrush`
- **Modelo base**: `timbrooks/instruct-pix2pix`
- **Dataset**: `osunlp/MagicBrush` (8807 train / 528 validation)
- **Resolución**: 256 px

### Métricas de evaluación (validation completo)

| Métrica | Entrenado | Base | Mejora |
|---|---|---|---|
| LPIPS | 0.1997 | 0.3316 | **+39.8%** ✅ |
| CLIP | 0.2476 | 0.2591 | -4.4% |

### Scripts disponibles
- `src/scripts/prepare_magicbrush.py` — procesamiento de dataset.
- `src/scripts/train_instruct_pix2pix.py` — entrenamiento (parches para DatasetDict local, validation_image local/URL, seed opcional, UNet IP2P ya entrenado).
- `src/scripts/evaluate.py` — evaluación cuantitativa LPIPS + CLIP.
- `scripts/test_checkpoint.py` — prueba rápida de un checkpoint.
- `scripts/cleanup_artifacts.sh` — limpieza de disco.

### Backup
- Todo el proyecto (sin `.venv` ni cachés) ya está respaldado en Google Drive.

---

## Objetivo del demo

Crear una aplicación web simple para editar imágenes con el checkpoint entrenado.

### Flujo esperado
1. El usuario sube una imagen.
2. Escribe un prompt de edición (ej: "add a hat", "make the background blue").
3. Clic en "Editar".
4. Ve imagen original y editada lado a lado.
5. Puede descargar el resultado.

---

## Propuesta técnica recomendada

### App web con Gradio

Gradio es la forma más rápida de tener una interfaz profesional en el navegador.

#### Ventajas
- Interfaz lista en minutos.
- Compartible con link local o temporal.
- No requiere frontend ni backend separados.
- Fácil de mostrar y testear.

#### Ubicación del script
- `scripts/demo.py` (script de desarrollo/demo).

#### Dependencias nuevas
```bash
pip install gradio
```

---

## Pasos para implementar el demo

### 1. Crear `scripts/demo.py`

El script debe:
- Cargar el pipeline desde `models/checkpoints/grafito-magicbrush`.
- Permitir subir imagen y escribir prompt.
- Aplicar los mismos parámetros de inferencia usados en evaluación.
- Mostrar original y editada lado a lado.
- Permitir descargar la imagen editada.

### 2. Parámetros de inferencia recomendados

- `resolution`: 256 (misma del entrenamiento).
- `num_inference_steps`: 20.
- `image_guidance_scale`: 1.5.
- `guidance_scale`: 7.0.
- `seed`: 42 (o aleatoria).

### 3. Ejecutar el demo

```bash
cd /workspace/Grafito
source .venv/bin/activate
python scripts/demo.py
```

Gradio abre en `http://127.0.0.1:7860`.

### 4. Compartir link temporal (opcional)

```bash
python scripts/demo.py --share
```

Eso genera un link público temporal para mostrar a otros.

---

## Archivos a crear/modificar

1. `scripts/demo.py` — app Gradio.
2. `docs/CHANGELOG.md` — registrar el demo.
3. `docs/TRAINING.md` — no cambia.
4. `docs/EVALUATION.md` — no cambia.

---

## Criterios de éxito del demo

- La app carga el checkpoint sin errores.
- Edita una imagen con un prompt representativo del dataset.
- El resultado es visualmente coherente y cercano al prompt.
- No consume más de ~10 GB de VRAM por inferencia.

---

## Próximos pasos después del demo

1. Probar el demo con varios prompts e imágenes.
2. Anotar problemas o artefactos que aparezcan.
3. Con esa información, decidir ajustes para el siguiente entrenamiento:
   - Más/menos steps.
   - Ajustar learning rate.
   - Añadir `conditioning_dropout_prob`.
   - Cambiar resolución a 512.
4. Entrenar una segunda versión si los resultados del demo lo ameritan.

---

## Notas importantes

- El checkpoint está en `models/checkpoints/grafito-magicbrush` dentro del Network Volume de Vast.ai. Si cambias de máquina, necesitas bajarlo o usar el backup de Drive.
- El modelo base ya está en caché de Hugging Face en el pod.
- La app Gradio es solo para desarrollo/demo, no es producto final.
