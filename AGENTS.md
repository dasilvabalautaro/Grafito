# AGENTS.md — Convenciones para Grafito

Este archivo define las convenciones que deben seguir los agentes (humanos o asistentes) que trabajen en el proyecto Grafito.

## Propósito del proyecto

Grafito es un editor local de imágenes por instrucciones. La fase actual (0) busca un prototipo funcional basado en InstructPix2Pix afinado con MagicBrush.

## Principios generales

1. **Documentar antes de implementar:** cualquier cambio técnico significativo debe reflejarse primero en `docs/` y luego en código.
2. **Mínimo cambio viable:** no añadir abstracciones, dependencias o configuraciones que no sean necesarias para el paso actual.
3. **Reproducibilidad:** todo script de entrenamiento, evaluación o inferencia debe ser ejecutable desde cero con las instrucciones de `docs/TRAINING.md` y `docs/EVALUATION.md`.
4. **No tocar datos ni checkpoints versionados:** las carpetas `data/`, `models/checkpoints/` y `outputs/` están en `.gitignore`. Nunca subir pesos ni datasets al repo.
5. **Idioma:** documentación y mensajes de commit en español; código, nombres de archivos e identificadores en inglés.

## Estructura de carpetas

- `src/grafito/`: módulos reutilizables (`editor.py`, `memory.py`, `image_utils.py`, `config.py`) y CLI (`cli.py`).
- `src/scripts/`: scripts de una sola ejecución (preparar datos, entrenar, evaluar).
- `scripts/`: scripts de desarrollo y validación (por ejemplo, `run_baseline.py`).
- `notebooks/`: notebooks de exploración. Numerar con dos dígitos: `01_`, `02_`, etc.
- `docs/`: documentación. Cada cambio importante se refleja en `CHANGELOG.md`.
- `tests/`: tests unitarios con `pytest`.

## Decisiones técnicas ya tomadas

- **Modelo base:** `timbrooks/instruct-pix2pix` (SD 1.5).
- **Upgrade condicional:** `diffusers/sdxl-instructpix2pix-768` si la calidad lo exige.
- **Dataset principal:** `osunlp/MagicBrush`.
- **Entrenamiento:** fine-tune parcial del UNet con el script oficial de Diffusers. LoRA/PEFT es optimización futura, no camino principal.
- **Inferencia:** soportar CUDA, MPS y CPU. En MPS no se usa `enable_model_cpu_offload()` por defecto porque diffusers 0.27.2 puede forzar internamente dispositivo CUDA.
- **Versiones fijas (macOS x86_64 / torch 2.2.2):** `diffusers==0.27.2`, `transformers==4.39.3`, `accelerate==0.28.0`, `datasets==2.18.0`, `peft==0.10.0`, `huggingface_hub==0.23.5`, `numpy<2`.

## Flujo de trabajo

1. Antes de una tarea nueva, leer `NEXT_LOCAL_EDITOR_TRAINING.md` y `docs/ARCHITECTURE.md`.
2. Si la tarea implica una decisión técnica nueva, actualizar `docs/ARCHITECTURE.md` y `docs/CHANGELOG.md`.
3. Si la tarea cambia hiperparámetros o resultados de evaluación, actualizar `docs/TRAINING.md` o `docs/EVALUATION.md`.
4. Al finalizar, ejecutar los tests o verificaciones que cubran el cambio.

## Estilo de código

- Python >= 3.10.
- Formateador: `black` (línea de 100 caracteres).
- Imports ordenados: stdlib, terceros, locales.
- Type hints opcionales pero recomendadas en funciones públicas.
- Docstrings en Google style.

## Tests

- Todo módulo nuevo debe tener al menos un test básico.
- Ejecutar tests con `pytest tests/`.

## Comunicación

- Reportar bloqueos con datos concretos (comando, error, archivo, línea).
- No asumir disponibilidad de GPU: todo script debe poder correr en CPU/MPS aunque sea lento.
