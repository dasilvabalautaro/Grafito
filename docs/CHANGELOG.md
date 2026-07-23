# Changelog — Grafito

Todas las actualizaciones importantes del proyecto se registran aquí.

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
  - Script de entrenamiento oficial parcheado en `src/scripts/train_instruct_pix2pix.py` para soportar `--dataset_name` con un `DatasetDict` local guardado por `save_to_disk`.
  - Tests para `prepare_magicbrush.py` en `tests/test_prepare_magicbrush.py`.
  - Descarga local robusta mediante `huggingface-cli download` como alternativa a `load_dataset` directo.

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
- Preparar dataset MagicBrush y lanzar primer entrenamiento.
