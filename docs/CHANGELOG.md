# Changelog — Grafito

Todas las actualizaciones importantes del proyecto se registran aquí.

## [Sin publicar] — 2026-07-28

### Añadido

- Checkpoint v3 `models/checkpoints/grafito-magicbrush-v3` (10000 pasos a 512 px desde v2, mezcla v3 de 27098 ejemplos con Instruct-CelebA, 7 h 11 min en RTX 4090 de Vast.ai). Descargado y verificado en local (MPS, edición correcta).
- Resultados v3 en `docs/EVALUATION.md`: LPIPS 0.2329 / CLIP 0.2511 a 512 px (v2: 0.2405 / 0.2509; base: 0.3046 / 0.2512, los tres medidos en la misma corrida de 528 ejemplos). Panel cualitativo en `outputs/panel_v3/` (6 casos × 2 seeds).
- Decisión documentada: **rollback a v2** según `docs/TRAINING_V3_PLAN.md` §10.2. v3 mejora LPIPS pero falla el panel de retratos (`remove his glasses` sustituye la escena; `make the background light blue` elimina al sujeto, regresión frente a v2). v2 sigue en producción; v3 se conserva solo como referencia.

### Cambiado

- `src/scripts/prepare_magicbrush.py`: `data_files` explícitos por split (la inferencia de `data_dir` falla offline con datasets 2.18).
- `src/scripts/prepare_instruct_celeba.py`: detección de la raíz del dataset extraído corregida (`extraction_exists` nunca encontraba el directorio).

### Validado

- Pod Vast.ai (RTX 4090): venv propio Python 3.10 con torch 2.2.2+cu121, 29 tests verdes, smoke de entrenamiento de 5 pasos, dataset v3 regenerado en el pod con stats idénticas al local.
- Smokes de retrato sobre checkpoints intermedios (3000/5000/7000) durante el run: sin divergencia; se observa sobre-edición creciente con los pasos.
- Nota operativa: los checkpoints guardados por el script de entrenamiento no incluyen `safety_checker`; hay que copiarlo del modelo base para que `from_pretrained` cargue el pipeline completo.

## [Sin publicar] — 2026-07-27

### Añadido

- Dataset v3 preparado y verificado en `data/processed/magicbrush_v3/`: **27.098 train / 528 validation**. Compuesto por MagicBrush filtrado (7.645 ejemplos tras filtros, con personas ×2) + Instruct-CelebA submuestreado (19.453 ejemplos).
- `src/scripts/prepare_instruct_celeba.py`: descarga, empareja y submuestra Instruct-CelebA con originales de CelebAMask-HQ (`v-xchen-v/celebamask_hq`), filtro facial opcional y submuestreo estratificado a ~20k.
- `src/scripts/filter_noisy_pairs.py`: filtra pares ruidosos de MagicBrush con LPIPS (cambio visual nulo), CLIP (adherencia al prompt) y prompts genéricos.
- `src/scripts/prepare_v3_mix.py`: construye la mezcla v3 aplicando filtro de tiras de esquina, filtro de calidad facial, sobremuestreo de personas ×2 y concatenación con Instruct-CelebA.
- Tests para los nuevos scripts: `tests/test_prepare_instruct_celeba.py` y `tests/test_prepare_v3_mix.py`.
- Checkpoint v2 `models/checkpoints/grafito-magicbrush-v2` (fine-tune completo a 512 px, 6000 pasos, 4 h 40 min en RTX 4090, mezcla 10099 ejemplos con personas ×2 y 44 tiras de calibración filtradas). **Adoptado para el demo** en sustitución de v1.
- Resultados v2 en `docs/EVALUATION.md`: LPIPS 0.2405 / CLIP 0.2509 a 512 (base: 0.3208 / 0.2523); panel cualitativo en `outputs/eval_v2/panel/` con caras y tinte magenta resueltos; persisten manchas de esquina esporádicas y adherencia fina irregular.
- `torchvision==0.17.2` fijado en `requirements.txt` y `pyproject.toml` (dependencia real del script de entrenamiento, detectada en el pre-flight).
- `docs/TRAINING_V3_PLAN.md`: plan detallado del tercer entrenamiento, reescrito tras confirmar fallos catastróficos de v2 en `review/` (`remove his glasses` y `replace the red cap with a blue beanie` destruyen la cara). Incluye boost facial ×3 sobre acciones remove/replace/change, filtro de calidad facial, 10000 pasos desde v2 y criterio de adopción obligatorio sobre el panel de retratos.
- Mitigaciones gratuitas del plan v3 en `scripts/demo.py`: multi-variante (1–3 seeds con galería y descarga de la seleccionada), recorte de bordes de 8 px contra manchas de esquina, y plantillas de prompt con atributos concretos. Incluye `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` y `torch.mps.empty_cache()` entre variantes para evitar OOM en MPS.
- `scripts/api_server.py`: microservicio FastAPI para integrar el modelo en una web (`GET /health`, `POST /edit` con imagen + prompt → PNG). CORS abierto para pruebas, lock de serialización de ediciones, recorte de bordes aplicado. Validado: edición 512 px en MPS ~40 s.
- `assets/test_prompts.json`: prompts de prueba consolidados (demo, regresión y panel cualitativo) para smoke tests y evaluación manual.
- `scripts/test_checkpoint.py`: soporta `--prompts-file` para generar un lote de imágenes (uno por prompt) en un directorio de salida, manteniendo el modo `--prompt` simple.

### Cambiado

- `scripts/demo.py`: checkpoint por defecto `grafito-magicbrush-v2` a 512 px; en MPS/CPU se activan `attention_slicing` + VAE slicing + VAE tiling (sin tiling, 512 hace OOM en la Radeon de 8 GB; con ellos ~31 s por imagen).
- `docs/NEXT_DEMO.md`: parámetros de inferencia a 512, estado de la infraestructura (instancia de Vast.ai destruida el 2026-07-27, coste del run ~$8–12 dentro del techo de $20) y nuevas funciones del demo.
- `docs/TRAINING_V2_PLAN.md`: checklist cerrado con resultados y desvíos documentados (venv Python 3.11 en el pod, dataset generado en el pod en vez de subirse).

### Corregido

- `src/scripts/prepare_instruct_celeba.py`: reescrito para evitar OOM y parones de `Dataset.from_generator`. Ahora cachea los originales necesarios desde Hugging Face en un directorio local (con reutilización) y procesa la construcción del dataset en shards de 1000 ejemplos.
- `src/scripts/filter_noisy_pairs.py`: redimensiona `edited_image` al tamaño de `original_image` antes de LPIPS para evitar error de tensores de feature maps distintos (MagicBrush tiene pares 500×500 vs 512×512).
- `src/scripts/prepare_v3_mix.py`: umbrales del detector de tiras de esquina ajustados a los valores heredados de v2 (`corner_px=32`, `sat>0.8`, `min_pixels=20`, `min_dispersion>0.15`) para no descartar el 99 % de MagicBrush como falsos positivos.
- `src/scripts/prepare_v3_mix.py`: lógica del filtro facial corregida para conservar ejemplos sin cara (casos no faciales) y descartar solo cuando la cara desaparece entre original y editada.

### Validado

- Pre-flight del pod: torch 2.2.2+cu121, pytest 19 verdes, smoke de 5 pasos.
- Dataset v3 verificado: carga desde disco, columnas correctas, imágenes a 512×512 (salvo pares legítimos 500×500 en MagicBrush), validation idéntico a v1/v2 (528 ejemplos).
- Checkpoint v2 carga y edita en local a 512 px tanto en CPU (2 min 4 s) como en MPS (31 s).
- Smoke test del demo con mitigaciones: 2 variantes a 512 px en MPS sin OOM (~31 s cada una), recorte aplicado (salida 496×496), descarga funcional.
- Tests unitarios: 29 passed.

### Pendiente

- Backup de `grafito-magicbrush-v2` en Google Drive (acción del usuario).
- Publicación en Hugging Face **pospuesta** (decisión 2026-07-27): primero estabilizar el modelo (uso real del demo + posible v3) antes de hacerlo público.

## [Sin publicar] — 2026-07-26

### Añadido

- `docs/TRAINING_V2_PLAN.md`: plan detallado del segundo entrenamiento. Actualizado el 2026-07-26: fine-tune completo en GPU rentada de 24 GB (una ejecución acotada, presupuesto $20, sin Network Volume) con la restricción de que el modelo final ejecute en el iMac local; mezcla de datos con sobremuestreo de retratos; criterios de éxito medibles frente a v1.

- Demo web `scripts/demo.py` con Gradio 4.44.1: subir imagen, instrucción de edición, parámetros ajustables (pasos, guidance scales, seed), vista original/editada lado a lado y botón de descarga. Usa `models/checkpoints/grafito-magicbrush` si existe; si no, cae a `timbrooks/instruct-pix2pix` con un aviso.
- Stack del demo fijado en `requirements.txt` y extra `demo` de `pyproject.toml`: `gradio==4.44.1`, `fastapi==0.114.2` (starlette<0.39) y `pydantic==2.8.2`, por compatibilidad con gradio-client 1.3 y el `huggingface_hub` fijado.

### Cambiado

- `docs/NEXT_DEMO.md`: el demo pasa a ejecutarse en hardware local (MPS, Radeon 5500 XT 8 GB) en lugar del pod de Vast.ai, para eliminar el coste de nube. En MPS/CPU se activa `attention_slicing` para reducir el pico de memoria.

### Validado

- Smoke test del demo: `python scripts/demo.py` sirve en local (HTTP 200) y una edición completa vía API (`turn him into a cyborg`, 20 pasos, MPS, seed 42) devuelve la imagen editada correctamente usando el modelo base (el checkpoint aún no está en local).
- Tests unitarios: 13 passed.
- Calidad del checkpoint verificada visualmente (`add a hat`, seeds 42 y 123): buena adherencia al prompt y fidelidad; artefactos de borde esporádicos según seed. Observaciones anotadas en `docs/NEXT_DEMO.md` como insumo para la v2.

### Añadido (preparación v2)

- `docs/V2_RUNBOOK.md`: runbook operativo del entrenamiento v2 (fases A/B/C con comandos exactos y criterio de éxito por tarea, hasta la descarga y verificación del modelo).
- `scripts/audit_dataset.py`: auditoría del dataset procesado (desplazamiento de color, anomalías de borde, cobertura de personas).
- `src/scripts/prepare_v2_mix.py`: mezcla v2 — sobremuestreo ×2 de ejemplos con personas y filtro de tiras de calibración de color en esquinas (`--drop_corner_strips`, activado por defecto); normaliza el split `dev` a `validation`.
- `tests/test_prepare_v2_mix.py`: tests de keywords de personas, del mix y del detector de tiras.
- `docs/TRAINING_V2_PLAN.md`: selección de máquina fijada (imagen `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`, GPUs 3090/4090/A10/A100, pre-flight obligatorio) y resultados de la Fase 0 anotados.

### Validado (preparación v2)

- Puerta 0.1: inferencia a 512 px en CPU, 2 min 4 s, sin OOM → resolución objetivo 512 px.
- Auditoría sobre 8807 ejemplos: el tinte magenta **no** viene del dataset (ΔRGB equilibrado, 0.33 % patrón magenta); 44 imágenes con tira de calibración en esquina (0.50 %) filtradas del mix; personas 15.22 %.
- Dataset a 512 px generado (train 8807 / dev 528, 9.1 GB) y mix v2 final: **train 10099 / validation 528** (`data/processed/magicbrush_v2/`).
- Tests unitarios: 19 passed.

### Corregido

- Checkpoint `grafito-magicbrush` restaurado desde Drive: el UNet venía con un nombre no estándar (`diffusion_pytorch_model-001.safetensors`) que diffusers no reconocía; se renombró a `diffusion_pytorch_model.safetensors` y carga correctamente. Verificado con `scripts/test_checkpoint.py` (`add a hat`, 20 pasos, CPU) — resultado en `outputs/test_entrenado.jpg`.

### Pendiente

- Dar de baja el volumen de Vast.ai (el checkpoint ya está verificado en local).

## [0.1.0] — 2026-07-23

### Añadido

- Documento de inicio completo (`NEXT_LOCAL_EDITOR_TRAINING.md`).
- Estructura inicial del repo: `src/grafito/`, `src/scripts/`, `scripts/`, `notebooks/`, `data/`, `models/checkpoints/`, `assets/`, `outputs/`, `tests/`, `docs/`.
- `README.md` con pitch, instalación y estructura.
- `AGENTS.md` con convenciones del proyecto.
- `pyproject.toml` y `requirements.txt` con dependencias iniciales y versiones fijas para reproducibilidad.
- Módulos base del editor:
  - `src/grafito/config.py` — configuración por defecto.
  - `src/grafito/image_utils.py` — carga, resize, recorte de bordes.
  - `src/grafito/memory.py` — selección de dispositivo y offload.
  - `src/grafito/editor.py` — pipeline de inferencia IP2P.
- CLI: `src/grafito/cli.py` (invocable con `python -m grafito.cli`).
- Scripts iniciales:
  - `src/scripts/prepare_magicbrush.py` — descarga y formatea MagicBrush.
  - `src/scripts/evaluate.py` — placeholder para evaluación.
  - `scripts/run_baseline.py` — validación rápida del pipeline base.
- Documentación en `docs/`:
  - `ARCHITECTURE.md`
  - `TRAINING.md`
  - `EVALUATION.md`
  - `CHANGELOG.md` (este archivo)
- Tests iniciales para `image_utils` y `memory`.
- Notebook de baseline: `notebooks/01_baseline_ip2pix.ipynb`.
- `.gitignore` para excluir datos, modelos, outputs y cachés.
- Preparación del dataset MagicBrush:
  - `src/scripts/prepare_magicbrush.py` ahora filtra ejemplos corruptos, redimensiona imágenes y guarda `stats.json`.
  - El dataset procesado se guarda como `DatasetDict` para poder cargarlo desde disco local.
  - Script de entrenamiento oficial parcheado en `src/scripts/train_instruct_pix2pix.py`:
  - Soporta `--dataset_name` con un `DatasetDict` local guardado por `save_to_disk`.
  - El argumento `--validation_image` acepta path local o URL para la imagen de validación.
  - Permite omitir `--seed` sin lanzar un `RuntimeError`.
  - Evita re-inicializar `conv_in` cuando el checkpoint ya es un UNet de InstructPix2Pix (8 canales), permitiendo fine-tuning desde `timbrooks/instruct-pix2pix`.
  - Tests para `prepare_magicbrush.py` en `tests/test_prepare_magicbrush.py`.
  - Descarga local robusta mediante `huggingface-cli download` como alternativa a `load_dataset` directo.
- Dependencia opcional de tracking con `wandb` añadida a `pyproject.toml` (`tracking`) y `requirements.txt`.
- Script de limpieza `scripts/cleanup_artifacts.sh` para liberar espacio de backups y dataset raw una vez procesado.
- Plan del siguiente paso en `docs/NEXT_DEMO.md`: demo con Gradio sobre el checkpoint entrenado.
- Script de prueba `scripts/test_checkpoint.py` para validar un checkpoint entrenado con una imagen y prompt (acepta paths locales o model IDs de Hugging Face).
- Script de evaluación cuantitativa `src/scripts/evaluate.py` con métricas LPIPS y CLIP score comparando el checkpoint entrenado contra `timbrooks/instruct-pix2pix`.
- `docs/EVALUATION.md` actualizado con el protocolo, uso de `evaluate.py` y resultados del primer entrenamiento (LPIPS 0.1997 vs 0.3316 de la línea base).

### Corregido / aclarado

- Se corrigió la suposición de LoRA en el documento original: el script oficial de Diffusers para InstructPix2Pix implementa fine-tune completo del UNet, no LoRA. LoRA queda como optimización futura.
- Se fijaron versiones de dependencias para compatibilidad con torch 2.2.2 en macOS x86_64:
  - `torch==2.2.2`, `diffusers==0.27.2`, `transformers==4.39.3`, `accelerate==0.28.0`, `datasets==2.18.0`, `peft==0.10.0`, `huggingface_hub==0.23.5`, `numpy<2`.
- Se ajustó `src/grafito/memory.py`:
  - `get_device()` ahora prueba MPS con un tensor antes de elegirlo.
  - `get_offload_strategy()` ya no aplica `model_cpu_offload` en MPS, ya que puede forzar dispositivo CUDA internamente en diffusers 0.27.2.

### Validado

- Baseline de IP2P ejecutado con éxito en MPS:
  - Modelo: `timbrooks/instruct-pix2pix`.
  - Prompt: `"turn him into a cyborg"`.
  - Resolución: 256×256, 10 pasos.
  - Tiempo de inferencia: ~6 s.
  - Resultado guardado en `outputs/baseline_edit.jpg`.
- CLI validado:
  - Comando: `python -m grafito.cli --image assets/example.jpg --prompt "turn him into a cyborg" --output outputs/cli_edit.jpg`.
  - Resultado guardado en `outputs/cli_edit.jpg`.
- Tests unitarios: 7 passed.

### Pendiente

- Implementar evaluación cuantitativa en `src/scripts/evaluate.py`.
- Lanzar primer entrenamiento con MagicBrush.
