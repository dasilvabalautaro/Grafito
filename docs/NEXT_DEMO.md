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

Instaladas y validadas (versiones fijas también en `requirements.txt`):

```bash
pip install gradio==4.44.1 fastapi==0.114.2 pydantic==2.8.2
```

Notas de compatibilidad:
- `gradio==4.44.1` (última 4.x): la 5.x exigiría subir el `huggingface_hub` fijado.
- `fastapi==0.114.2` (starlette<0.39): gradio 4.44 usa la API antigua de `TemplateResponse` y con starlette moderno no sirve la página.
- `pydantic==2.8.2`: gradio-client 1.3 no entiende los JSON schemas de pydantic>=2.10.

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

- `resolution`: 512 (nativa del checkpoint v2; en MPS requiere `attention_slicing` + VAE slicing + VAE tiling, ya integrados en `scripts/demo.py`; ~31 s por imagen).
- `num_inference_steps`: 20.
- `image_guidance_scale`: 1.5.
- `guidance_scale`: 7.0.
- `seed`: 42 (o aleatoria).

### 3. Ejecutar el demo

El demo corre en hardware local (Mac con MPS; Radeon 5500 XT 8 GB). Ya no se usa el pod de Vast.ai.

```bash
cd Grafito
source .venv/bin/activate
python scripts/demo.py
```

Gradio abre en `http://127.0.0.1:7860`.

Si el checkpoint aún no está en `models/checkpoints/grafito-magicbrush`, el script
cae automáticamente al modelo base `timbrooks/instruct-pix2pix` con un aviso en consola.

### 4. Compartir link temporal (opcional)

```bash
python scripts/demo.py --share
```

Eso genera un link público temporal para mostrar a otros.

---

## Archivos a crear/modificar

1. `scripts/demo.py` — app Gradio.
2. `requirements.txt` / `pyproject.toml` — dependencia `gradio>=4.0,<5.0` (extra `demo`).
3. `docs/CHANGELOG.md` — registrar el demo.
4. `docs/TRAINING.md` — no cambia.
5. `docs/EVALUATION.md` — no cambia.

---

## Criterios de éxito del demo

- La app carga el checkpoint sin errores.
- Edita una imagen con un prompt representativo del dataset.
- El resultado es visualmente coherente y cercano al prompt.
- No consume más de ~10 GB de VRAM por inferencia.

---

## Observaciones de calidad (primeras pruebas del checkpoint, 2026-07-26)

Pruebas con `scripts/test_checkpoint.py` sobre `assets/example.jpg`, prompt `add a hat`, 20 pasos, CPU, seeds 42 y 123.

### Lo que ya funciona bien

- **Adherencia al prompt:** el objeto añadido se integra con iluminación y perspectiva coherentes (no parece un parche pegado).
- **Fidelidad:** pose, fondo y encuadre se conservan bien — coherente con el LPIPS 0.1997 del checkpoint frente a 0.3316 del base.

### Artefactos detectados

- **Manchas de color esporádicas en bordes** (~10 px, tipo paleta cian/magenta): cambian de posición según la seed, no son sistemáticas. Mitigación en demo: re-tirar la seed o recortar 8–10 px de borde.
- **Reinterpretación de detalles finos** (ej.: pezón → tachuela metálica, pequeños hoyuelos). Visible solo con zoom.
- **Suavidad general** propia de 256 px, más la compresión JPEG al guardar.
- **Caras de personas degradadas** (prueba con `foto.jpg`, retrato): la boca se "derrite" y los ojos se distorsionan en todas las variantes probadas. Es el punto débil principal del checkpoint — MagicBrush tiene pocos retratos.
- **Tinte magenta sistemático** en la foto de retrato (independiente de la seed): apunta a tendencia del checkpoint, no a lotería de seeds.
- **Prompt específico > subir `image_guidance`:** `add a black hat` dio mejor integración del objeto y mejor preservación de la cara que `add a hat` con `image_guidance_scale` 2.2. Recomendación de uso: prompts con color/atributos concretos.

### Insumos para la v2

- Subir resolución a **512 px**: es la limitación de calidad más evidente.
- Revisar ejemplos de MagicBrush con bordes anómalos como posible origen de las manchas de borde.
- Guardar salidas en PNG para no mezclar artefactos de compresión con los del modelo.
- Evaluar si el tinte magenta en fotos de interior es reproducible con más imágenes; de ser sistemático, revisar balance de color del dataset procesado.
- Si el caso de uso incluye personas, considerar complementar el entrenamiento con ediciones sobre retratos (el punto débil actual).

---

## Próximos pasos después del demo

1. Probar el demo con varios prompts e imágenes.
2. Anotar problemas o artefactos que aparezcan (primeras anotaciones en «Observaciones de calidad»).
3. Con esa información, decidir ajustes para el siguiente entrenamiento:
   - Más/menos steps.
   - Ajustar learning rate.
   - Añadir `conditioning_dropout_prob`.
   - Cambiar resolución a 512.
4. Entrenar una segunda versión si los resultados del demo lo ameritan.

---

## Notas importantes

- **Coste cero de nube:** el demo corre 100% en local. La instancia de entrenamiento de Vast.ai se destruyó el 2026-07-27; no queda nada facturando.
- El demo usa por defecto `models/checkpoints/grafito-magicbrush-v2` (adoptado el 2026-07-27; ver resultados en `docs/EVALUATION.md`). v1 se conserva en `models/checkpoints/grafito-magicbrush`.
- Si el checkpoint aún no está en local, el script cae al modelo base `timbrooks/instruct-pix2pix` (en caché local de Hugging Face) con un aviso en consola.
- En MPS se activan `attention_slicing` + VAE slicing + VAE tiling para caber en los 8 GB de VRAM (validado a 512 px, ~31 s; 512 px sin tiling hace OOM).
- La app Gradio es solo para desarrollo/demo, no es producto final.
