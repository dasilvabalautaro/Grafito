# Arquitectura de Grafito

## Diagrama de componentes

```
┌─────────────────────────────────────────────────────────────┐
│                         Usuario                             │
│  CLI: python -m grafito.cli --image ... --prompt ...       │
└──────────────────────────┬──────────────────────────────────┘
                           │
           ┌───────────────▼────────────────┐
           │      src/grafito/cli.py        │
           │  Parseo de argumentos          │
           └───────────────┬────────────────┘
                           │
           ┌───────────────▼────────────────┐
           │      src/grafito/editor.py     │
           │  Carga pipeline IP2P, ejecuta  │
           │  inferencia y devuelve imagen  │
           └───────────────┬────────────────┘
                           │
     ┌─────────────────────┼─────────────────────┐
     │                     │                     │
┌────▼────┐         ┌──────▼──────┐      ┌──────▼──────┐
│ config  │         │ memory      │      │ image_utils │
│ .py     │         │ .py         │      │ .py         │
└─────────┘         └─────────────┘      └─────────────┘
```

## Módulos

### `src/grafito/config.py`

Define `EditConfig`, la configuración por defecto de la edición (modelo, resolución, pasos, escalares de guidance).

### `src/grafito/memory.py`

Selecciona automáticamente el dispositivo (`cuda`, `mps`, `cpu`) y el dtype adecuado. Decide si activar CPU offload para modelos grandes.

### `src/grafito/image_utils.py`

Funciones de carga, normalización, redimensionado manteniendo aspect ratio y recorte de bordes claros.

### `src/grafito/editor.py`

Expone `load_pipeline()` y `edit_image()`. Es la capa de alto nivel que orquesta el pipeline de Diffusers.

### `src/grafito/cli.py`

Punto de entrada por línea de comandos usando `click`. Invocable con `python -m grafito.cli`.

### `src/scripts/`

Scripts de una sola ejecución:

- `prepare_magicbrush.py`: descarga y formatea MagicBrush.
- `evaluate.py`: evalúa un checkpoint (placeholder).

## Flujo de datos

1. Entrada: imagen + prompt.
2. `image_utils.load_image` normaliza la imagen.
3. `image_utils.resize_for_model` la adapta al tamaño del modelo.
4. `editor.edit_image` codifica, difunde y decodifica.
5. Salida: imagen editada guardada en disco.

## Notas de implementación

### Selección de dispositivo (`memory.py`)

- CUDA se usa si está disponible.
- MPS se usa solo si un tensor de prueba se crea correctamente (algunos entornos x86_64 reportan MPS pero no lo ejecutan).
- CPU es el fallback.
- `model_cpu_offload` no se aplica en MPS con diffusers 0.27.2 porque puede forzar internamente dispositivo CUDA y lanzar `AssertionError: Torch not compiled with CUDA enabled`.

### Versiones de dependencias

Para reproducibilidad en macOS x86_64 con torch 2.2.2:

- `torch==2.2.2`
- `diffusers==0.27.2`
- `transformers==4.39.3`
- `accelerate==0.28.0`
- `datasets==2.18.0`
- `peft==0.10.0`
- `huggingface_hub==0.23.5`
- `numpy<2`

## Decisiones pendientes

- Si se añade LoRA en el futuro, `editor.py` deberá cargar los pesos PEFT sobre el UNet.
- La interfaz CLI puede evolucionar a una API REST si el proyecto lo requiere.
