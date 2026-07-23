# Grafito — Documento de inicio del proyecto

> **Versión:** 0.2 — 23 de julio de 2026  
> **Responsable técnico:** Kimi Code CLI (lead)  
> **Propietario del producto:** David Silva  
> **Estado:** borrador para aprobación

---

## 0. Resumen ejecutivo

**Grafito** es un editor de imágenes local que modifica una foto a partir de una instrucción en lenguaje natural del tipo *"ponle un gorro y conserva todo lo demás"*. Hoy el proyecto resuelve ese caso mediante una API de FLUX; el objetivo de esta fase es conseguir una alternativa **local, privada y gratuita por inferencia** que corra en hardware de gente común.

La estrategia elegida es:

1. **Base:** [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix) (SD 1.5 con arquitectura InstructPix2Pix).
2. **Ajuste:** fine-tune parcial del UNet sobre [MagicBrush](https://huggingface.co/datasets/osunlp/MagicBrush) usando el script oficial de Diffusers.
3. **Inferencia:** pipeline local reutilizable en CPU/MPS/CUDA con fallback de memoria.
4. **Éxito:** superar la línea base de IP2P crudo en 5 pruebas de edición con métricas cuantitativas.

---

## 1. Contexto y motivación

### 1.1 ¿Qué problema resolvemos?

Los modelos de difusión actuales (FLUX, Qwen-Image-Edit, OmniGen) logran ediciones de alta calidad, pero:

- Requieren GPU de gama alta o llamadas a API de pago.
- Procesan las imágenes fuera de la máquina del usuario.
- No permiten reentrenar fácilmente para un estilo o dominio propio.

Para muchos usuarios, una edición local de **calidad "suficientemente buena"** es preferible si es privada, rápida y gratuita.

### 1.2 ¿Por qué ahora?

- Existen datasets abiertos de edición real anotada por humanos (MagicBrush).
- Diffusers tiene pipelines y scripts oficiales para InstructPix2Pix.
- SD 1.5/IP2P cabe en 6–8 GB de VRAM y corre en MPS de Mac.

### 1.3 Lecciones del documento anterior

El borrador técnico anterior (`NEXT_LOCAL_EDITOR_TRAINING.md`, enero 2026) acertó en la dirección general pero tenía lagunas importantes:

- **Asumía LoRA** sin verificar que el script oficial de Diffusers solo implementa fine-tune completo del UNet. Este documento corrige ese punto.
- No definía alcance, hitos ni presupuesto.
- Mezclaba entrenamiento, inferencia y evaluación sin una estructura de proyecto.
- Los criterios de éxito eran cualitativos.

---

## 2. Visión y propuesta de valor

### 2.1 Visión

> Un editor de imágenes por instrucciones que funcione **offline** en el ordenador del usuario, capaz de añadir, quitar, reemplazar y cambiar el fondo conservando la identidad y el resto de la escena.

### 2.2 Propuesta de valor

- **Privacidad:** la imagen nunca sale del dispositivo.
- **Coste cero por inferencia:** se paga solo el entrenamiento/ajuste (una vez o por GPU de alquiler).
- **Control:** se puede afinar para un dominio o estilo concreto.
- **Hardware accesible:** el prototipo debe correr en una GPU de 8 GB o en MPS de Mac.

### 2.3 Usuarios objetivo (fase 0)

- Creadores de contenido que quieren probar ediciones rápidas sin subir fotos a la nube.
- Desarrolladores que quieren un componente de edición local integrable en sus apps.

---

## 3. Alcance y no-alcance de la fase 0

### 3.1 Dentro del alcance

- Elegir y validar un modelo base local para edición por instrucciones.
- Preparar un dataset de entrenamiento a partir de MagicBrush.
- Entrenar/afinar el modelo base en GPU de alquiler.
- Implementar un script de inferencia local reutilizable.
- Definir y ejecutar una batería de evaluación cuantitativa y cualitativa.
- Documentar el proceso para que sea reproducible.

### 3.2 Fuera del alcance

- Interfaz gráfica de usuario (GUI). Se trabaja por CLI/notebook.
- Edición de texto legible en imágenes (logos, carteles).
- Entrenamiento desde cero de un nuevo modelo de difusión.
- Distribución comercial del producto final.
- Soporte de múltiples idiomas (se parte del inglés, aunque se acepten prompts en español vía traducción opcional).

### 3.3 Límites de calidad aceptados

- No se pretende igualar a FLUX en escenas complejas.
- El objetivo es una edición local **decente y útil**, no la mejor del mercado.

---

## 4. Estado del arte y decisión de modelo base

### 4.1 Criterios de selección

| Criterio | Peso | Justificación |
|---|---|---|
| Inferencia en hardware común | Alta | El producto diferencial es "corre en tu máquina". |
| Tooling de entrenamiento maduro | Alta | Queremos reproducibilidad, no inventar scripts. |
| Calidad de conservación de identidad | Alta | El caso de uso clave es "cambia X, conserva el resto". |
| Licencia permisiva | Media | Debe permitir investigación y prototipado. |
| Tamaño de modelo | Media | Menor tamaño = menor latencia y memoria. |

### 4.2 Comparativa de modelos base

| Modelo | Tamaño | Inferencia local | Entrenamiento documentado | Calidad | Veredicto |
|---|---|---|---|---|---|
| **SD 1.5 InstructPix2Pix** ([timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)) | 860 M | 6–8 GB, MPS incluido | Script oficial Diffusers | Modesta/alta si se afina | **Empezar aquí** |
| **SDXL IP2P-768** ([diffusers/sdxl-instructpix2pix-768](https://huggingface.co/diffusers/sdxl-instructpix2pix-768)) | 2.6 B | 10–16 GB | Script oficial SDXL | Alta | Upgrade si la calidad lo justifica |
| PixArt-Σ | ~600 M | Ligero | Menos tooling de edición | Alta/eficiente | Explorar, no primero |
| SD/SDXL-Turbo, LCM | = base | Muy rápido | No mejora edición | Media | Distilar al final |
| **FLUX / OmniGen / Qwen-Image-Edit** | 3.8–20 B | No en hardware común | No en GPU de alquiler sencilla | Muy alta | Descartados para local |

### 4.3 Decisión: SD 1.5 InstructPix2Pix

**Recomendación:** prototipo con **SD 1.5 InstructPix2Pix + fine-tune parcial del UNet sobre MagicBrush**.

Razones:

1. Ya trae la arquitectura de 8 canales de entrada en la primera convolución del UNet; no hay que modificarla desde cero.
2. El script oficial de Diffusers (`train_instruct_pix2pix.py`) es directamente usable.
3. Es el único de la lista que garantiza inferencia local en Mac MPS y GPU de 8 GB.
4. MagicBrush está anotado por humanos y se alinea con el objetivo de "conservar mientras se edita".

**Upgrade condicional:** si tras el primer entrenamiento la calidad no es aceptable, evaluar SDXL IP2P-768. Eso requiere GPU de alquiler de 16–24 GB.

### 4.4 Advertencia importante sobre LoRA

El documento anterior proponía usar **LoRA** para ahorrar memoria. Tras revisar el script oficial de Diffusers (`train_instruct_pix2pix.py`), **no incluye soporte nativo de LoRA/PEFT**: entrena todos los parámetros del UNet. Por tanto:

- **Camino principal documentado:** fine-tune parcial del UNet (congelar VAE y text encoder), usando el script oficial.
- **Optimización futura:** investigar/adaptar PEFT/LoRA sobre las capas de atención del UNet para reducir memoria. Esto es técnicamente posible pero requiere validación extra, especialmente porque la capa `conv_in` de 8 canales no se beneficia de LoRA.
- Si la VRAM es crítica, se puede reducir resolución, batch y usar gradient checkpointing antes que intentar LoRA sin validar.

---

## 5. Arquitectura de software propuesta

```
grafito/
├── src/
│   ├── grafito/
│   │   ├── __init__.py
│   │   ├── editor.py          # Pipeline de inferencia IP2P + carga de LoRA/fine-tune
│   │   ├── memory.py          # Gestión de dispositivo: cuda/mps/cpu + offload
│   │   ├── image_utils.py     # Carga, recorte de marco, resize, exif
│   │   └── config.py          # Configuración por defecto (resolución, guidance, etc.)
│   ├── scripts/
│   │   ├── prepare_magicbrush.py   # Descarga y formatea MagicBrush
│   │   ├── train_instructpix2pix.py # Wrapper del script oficial con nuestros defaults
│   │   └── evaluate.py             # Métricas y reporte
│   └── cli/
│       └── edit.py             # python -m grafito.edit --image ... --prompt ...
├── notebooks/
│   ├── 01_baseline_ip2pix.ipynb
│   └── 02_explore_magicbrush.ipynb
├── data/
│   ├── raw/                    # Datasets descargados (no versionados)
│   └── processed/              # Splits y metadata
├── models/
│   └── checkpoints/            # Pesos entrenados (no versionados)
├── tests/
│   ├── test_editor.py
│   └── test_image_utils.py
├── docs/
│   └── EVALUATION.md
├── README.md
├── AGENTS.md
└── pyproject.toml / requirements.txt
```

### 5.1 Componentes clave

#### `editor.py`

- Carga `StableDiffusionInstructPix2PixPipeline`.
- Aplica checkpoint propio o base.
- Soporta `enable_model_cpu_offload()` y `enable_sequential_cpu_offload()`.
- Permite ajustar `image_guidance_scale` y `guidance_scale`.

#### `memory.py`

- Detecta dispositivo: CUDA > MPS > CPU.
- Selecciona dtype: `float16` (CUDA), `float32` (MPS/CPU, con advertencia de rendimiento).
- Fallback automático a CPU offload si la memoria es insuficiente.

#### `image_utils.py`

- `load_image(path_or_url)` con `PIL.ImageOps.exif_transpose`.
- `resize_for_model(image, max_size=512)` manteniendo aspect ratio.
- `_trim_light_border(image)` para eliminar bordes blancos/negros si los hay.

### 5.2 Formato de entrada/salida

```python
{
    "input_image": PIL.Image.Image,   # RGB
    "edit_prompt": "add a brown hat",
    "output_image": PIL.Image.Image,
    "seed": 42,
    "image_guidance_scale": 1.5,
    "guidance_scale": 7.5,
    "num_inference_steps": 20
}
```

---

## 6. Plan de datos

### 6.1 Datasets principales

| Dataset | Tamaño | Licencia | Uso |
|---|---|---|---|
| **MagicBrush** ([osunlp/MagicBrush](https://huggingface.co/datasets/osunlp/MagicBrush)) | train: 8.807 ejemplos; dev: 528 | CC-BY-4.0 | Dataset principal de afinamiento. Incluye máscaras. |
| **InstructPix2Pix (CLIP-filtered)** ([timbrooks/instructpix2pix-clip-filtered](https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered)) | ~100k+ | Revisar en HF | Refuerzo de volumen si MagicBrush solo no basta. |
| UltraEdit, HQ-Edit, SEED-Data-Edit | Variables | Revisar | Datasets recientes; valorar calidad y licencia antes de usarlos. |

### 6.2 Estrategia de datos

1. **Fase 1:** entrenar solo con MagicBrush train (8.807 ejemplos). Es la señal más limpia para "conservar mientras editas".
2. **Fase 2 (si es necesario):** mezclar una fracción (20–30 %) del dataset InstructPix2Pix para aumentar diversidad.
3. **Test:** usar MagicBrush dev (528 ejemplos) como held-out. El test oficial de MagicBrush está oculto; solicitarlo solo si se quiere publicar métricas comparables.

### 6.3 Preprocesamiento

```python
# Esquema HF Datasets esperado por el script oficial
{
    "original_image_column": PIL.Image.Image,  # imagen original
    "edited_image_column":   PIL.Image.Image,  # imagen editada
    "edit_prompt_column":    "add a brown hat"
}
```

Pasos:

1. Descargar dataset con `datasets.load_dataset("osunlp/MagicBrush")`.
2. Renombrar columnas: `source_img → original_image`, `target_img → edited_image`, `instruction → edit_prompt`.
3. Filtrar ejemplos con imágenes corruptas o prompts vacíos.
4. Redimensionar a 256×256 para entrenamiento rápido; luego subir a 512×512.
5. Guardar en formato Parquet o Arrow para entrenamiento eficiente.

### 6.4 Máscaras

MagicBrush incluye `mask_img` (región blanca = zona a editar). El script oficial de IP2P **no usa máscaras**, pero se pueden aprovechar más adelante para:

- Evaluar conservación: comparar píxeles fuera de la máscara.
- Entrenar un modelo inpainting/edit híbrido.

Para la fase 0, las máscaras se reservan para evaluación.

---

## 7. Plan de entrenamiento

### 7.1 Hardware recomendado

| Escenario | GPU | VRAM | Coste aprox. (jul. 2026) |
|---|---|---|---|
| Mínimo viable | RTX 3090 / 4090 / A10 | 24 GB | ~$0.40–0.80/hora en RunPod/Vast |
| Cómodo | A100 40 GB | 40 GB | ~$1.50–2.50/hora |
| Local (solo pruebas) | MPS Mac / CPU | 16+ GB RAM compartida | Gratis, pero muy lento |

**Presupuesto estimado:** 20–40 horas de GPU de 24 GB para el primer experimento (~$15–30).

### 7.2 Script base

Usar el script oficial de Diffusers:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers/examples/instruct_pix2pix
pip install -r requirements.txt
```

Script: `train_instruct_pix2pix.py`  
Documentación: [InstructPix2Pix training guide](https://huggingface.co/docs/diffusers/training/instructpix2pix)

### 7.3 Hiperparámetros de partida

| Parámetro | Valor inicial | Nota |
|---|---|---|
| Modelo base | `timbrooks/instruct-pix2pix` | Ya tiene arquitectura IP2P. |
| Resolución | 256 → luego 512 | 256 entrena rápido y el paper generaliza a 512. |
| Batch | 4 | Ajustar según VRAM. |
| Gradient accumulation | 4 | Batch efectivo = 16. |
| Learning rate | 5e-5 | Default del script oficial; probar 1e-4 si converge lento. |
| Scheduler | constante / cosine | Empezar con constante; cambiar si hay inestabilidad. |
| Pasos | 5.000–15.000 | Con MagicBrush (~9k ejemplos), 5k–10k suelen bastar. |
| Precision | fp16 / bf16 | fp16 en CUDA; bf16 si está disponible. |
| Gradient checkpointing | Sí | Obligatorio para 24 GB. |
| xformers/SDPA | Sí | Reducir memoria. |
| Conditioning dropout | 0.05 | Habilita CFG para imagen y texto. |

### 7.4 Congelación de componentes

- **UNet:** entrenable.
- **VAE:** congelado.
- **Text encoder (CLIP):** congelado.
- **conv_in de 8 canales:** se mantiene la arquitectura heredada; sus pesos se entrenan como parte del UNet.

### 7.5 Inferencia: guías de control

IP2P tiene dos escalares críticos:

- `guidance_scale`: cuánto influye la instrucción de texto.
- `image_guidance_scale`: cuánto se conserva la imagen original.

Valores de partida:

```python
num_inference_steps = 20
guidance_scale = 7.5        # seguir la instrucción
image_guidance_scale = 1.5  # conservar identidad/fondo
```

Se debe hacer una búsqueda en grid sobre el set de validación para encontrar el equilibrio óptimo.

---

## 8. Plan de evaluación

### 8.1 Criterios cualitativos de aceptación

Un checkpoint se aprueba si supera la línea base de IP2P crudo en **4 de 5** pruebas manuales, sin ajustar parámetros por objeto:

1. **Añadir** un elemento (gorro, gafas, ropa) conservando rostro y fondo.
2. **Quitar** un elemento.
3. **Reemplazar** color o material.
4. **Cambiar fondo** conservando el sujeto.
5. **Conservación estricta:** si la orden dice "conserva X", los píxeles fuera del cambio no se alteran perceptiblemente.

### 8.2 Métricas cuantitativas

| Métrica | Qué mide | Umbral de mejora vs base |
|---|---|---|
| **CLIP directional similarity** | El cambio visual coincide con la instrucción. | ≥ base + 0.03 |
| **SSIM región no editada** | Conservación de píxeles fuera de la máscara. | ≥ base + 0.02 |
| **LPIPS región no editada** | Distancia perceptual en zona no editada (menor = mejor). | ≤ base − 0.02 |
| **FID / LPIPS global** (opcional) | Calidad general en un held-out set. | Mejorar o mantener. |
| **Face similarity** (si aplica) | Conservación de identidad facial. | ≥ base + 0.03 |

### 8.3 Conjunto de evaluación

- **Automático:** MagicBrush dev (528 ejemplos).
- **Manual:** 20–30 ejemplos representativos seleccionados del dev set.
- **Stress test:** 5 casos difíciles (cambios sutiles, fondos complejos, múltiples objetos).

### 8.4 Protocolo

1. Generar ediciones con el checkpoint y con IP2P crudo usando los mismos prompts y seeds.
2. Calcular métricas automáticas sobre MagicBrush dev.
3. Revisión visual ciega: mezclar resultados y votar cuál conserva/mejor obedece.
4. Ajustar `image_guidance_scale` y repetir si es necesario.

---

## 9. Hitos y cronograma

| Hito | Duración estimada | Entregable |
|---|---|---|
| **H0 — Documento de inicio aprobado** | 0 días | Este documento firmado/aceptado. |
| **H1 — Línea base de inferencia** | 3–5 días | Notebook/script que corre IP2P crudo en el hardware objetivo. |
| **H2 — Dataset preparado** | 3–5 días | MagicBrush formateado, filtrado, con splits. |
| **H3 — Primer entrenamiento** | 5–10 días | Checkpoint afinado y evaluado con métricas. |
| **H4 — Iteración de guías y calidad** | 3–5 días | Valores óptimos de `guidance_scale`/`image_guidance_scale`. |
| **H5 — Integración en backend local** | 3–5 días | Script `src/cli/edit.py` funcional con CPU offload. |
| **H6 — Decisión de escalado** | 2–3 días | Informe: ¿subir a SDXL, iterar datos, o dar por válido? |

**Duración total estimada:** 3–5 semanas de trabajo efectivo, más el tiempo de entrenamiento en GPU.

---

## 10. Presupuesto de hardware y costes

| Concepto | Estimación | Nota |
|---|---|---|
| GPU de alquiler (primeros experimentos) | $15–40 | 20–50 h en RTX 3090/4090/A10. |
| GPU de alquiler (si se sube a SDXL) | $50–150 | Más VRAM y más pasos. |
| Almacenamiento HF / Drive | $0–5 | Datasets y checkpoints. |
| Tiempo de ingeniería | 3–5 semanas | Principal coste del proyecto. |

**Coste mínimo viable:** ~$20 en GPU + tiempo de desarrollo.

---

## 11. Riesgos y mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| IP2P crudo ya cumple lo suficiente | Media | Bajo | Hacer la línea base primero; si es suficiente, se ahorra entrenamiento. |
| IP2P afinado no conserva identidad | Media | Alto | Usar MagicBrush; ajustar `image_guidance_scale`; entrenar más pasos o subir resolución. |
| Fine-tune no cabe en 24 GB | Baja | Medio | Gradient checkpointing, resolución 256, batch pequeño, acumulación. |
| Dataset MagicBrush tiene sesgos/límite de calidad | Media | Medio | Explorar UltraEdit/HQ-Edit como refuerzo; hacer EDA previo. |
| Calidad inferior a FLUX API | Alta | Bajo | Esto es aceptado por diseño; mantener FLUX API como fallback opcional. |
| Texto legible sigue sin funcionar | Alta | Bajo | Fuera de alcance de la fase 0. |
| Licencias restrictivas en datos/pesos | Baja | Alto | Revisar licencias antes de publicar pesos o producto (ver §12). |

---

## 12. Licencias y cumplimiento

### 12.1 Modelos

| Recurso | Licencia | Implicaciones |
|---|---|---|
| Stable Diffusion v1.5 | CreativeML OpenRAIL M | Permite uso comercial con restricciones éticas; hay que adjuntar la licencia. |
| timbrooks/instruct-pix2pix | MIT | Muy permisiva. |
| diffusers/sdxl-instructpix2pix-768 | Revisar en HF | Probablemente OpenRAIL M + condiciones del dataset. |

### 12.2 Datasets

| Recurso | Licencia | Implicaciones |
|---|---|---|
| MagicBrush | CC-BY-4.0 | Hay que atribuir. Uso comercial permitido. |
| InstructPix2Pix dataset | Revisar en HF | Proviene de LAION/Wikipedia; verificar términos. |

### 12.3 Producto

- Si se distribuyen pesos afinados, deben respetar las licencias de base y dataset.
- Si se integra en una app, hay que incluir atribuciones y no usarlo para contenido prohibido (OpenRAIL M).
- Antes de cualquier publicación comercial, revisión legal de licencias.

---

## 13. Roles y responsabilidades

| Rol | Quién | Responsabilidades |
|---|---|---|
| **Product Owner / Sponsor** | David Silva | Define prioridades, aprueba alcance, decide si subir a SDXL, asigna presupuesto. |
| **Lead Técnico** | Kimi Code CLI | Diseña arquitectura, prepara datos, entrena, evalúa, documenta, toma decisiones técnicas día a día. |
| **Revisor de calidad** | David Silva (con apoyo) | Revisión visual de resultados, validación de criterios cualitativos. |

---

## 14. Checklist de arranque

- [ ] Aprobar este documento de inicio.
- [ ] Definir presupuesto de GPU de alquiler.
- [ ] Crear entorno virtual/conda e instalar dependencias base (`diffusers`, `transformers`, `accelerate`, `datasets`, `torch`, `safetensors`).
- [ ] Reproducir inferencia de `timbrooks/instruct-pix2pix` como línea base (H1).
- [ ] Descargar y formatear MagicBrush (H2).
- [ ] Lanzar primer entrenamiento con hiperparámetros de la §7.3 (H3).
- [ ] Evaluar checkpoint con métricas de la §8.2 y pruebas de la §8.1 (H4).
- [ ] Integrar el mejor checkpoint en `src/cli/edit.py` con manejo de memoria (H5).
- [ ] Redactar informe de decisión de escalado (H6).
- [ ] Revisar licencias antes de publicar cualquier peso o demo.

---

## 15. Próximos pasos inmediatos

1. **Aprobar este documento** (o indicar cambios).
2. **Decidir si se ejecuta Opción A, B o C:**
   - **A:** solo documento (ya entregado en este archivo).
   - **B:** documento + estructura inicial de repo (`README.md`, `AGENTS.md`, carpetas, `pyproject.toml`, notebook de baseline).
   - **C:** documento + estructura + prototipo funcional de inferencia local.
3. Si se elige B o C, crear la estructura de carpetas y archivos iniciales.
4. Si se elige C, implementar y probar `src/cli/edit.py` con IP2P crudo.

---

## 16. Apéndice: recursos y referencias

- [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)
- [diffusers/sdxl-instructpix2pix-768](https://huggingface.co/diffusers/sdxl-instructpix2pix-768)
- [osunlp/MagicBrush](https://huggingface.co/datasets/osunlp/MagicBrush)
- [timbrooks/instructpix2pix-clip-filtered](https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered)
- [Diffusers InstructPix2Pix training guide](https://huggingface.co/docs/diffusers/training/instructpix2pix)
- [InstructPix2Pix paper / GitHub](https://github.com/timothybrooks/instruct-pix2pix)
- [Stable Diffusion v1.5 license](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)

---

**Resumen en una línea:** partir de SD 1.5 InstructPix2Pix, afinar su UNet con MagicBrush en una GPU de alquiler de 24 GB, evaluar con métricas de conservación y edición, e integrar la inferencia local en un backend reutilizable para Grafito.
