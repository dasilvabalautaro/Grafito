# Grafito

Editor local de imágenes por instrucciones en lenguaje natural.

> **Estado:** fase 0 — prototipo de edición local con InstructPix2Pix.

## Propuesta

Grafito permite modificar una imagen a partir de una instrucción como *"ponle un gorro y conserva todo lo demás"*, sin enviar la foto a la nube. El objetivo de esta fase es conseguir una alternativa local, privada y gratuita por inferencia al flujo actual basado en API de FLUX.

## Estrategia técnica

- **Modelo base:** [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix) (SD 1.5 con arquitectura InstructPix2Pix).
- **Ajuste:** fine-tune parcial del UNet sobre [MagicBrush](https://huggingface.co/datasets/osunlp/MagicBrush).
- **Inferencia:** pipeline local con soporte CUDA / MPS / CPU y fallback de memoria.
- **Éxito:** superar la línea base de IP2P crudo en conservación y fidelidad a la instrucción.

## Instalación

```bash
# Clonar el repo
git clone <url-del-repo>
cd Grafito

# Crear entorno
python -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
# o, si se prefiere:
pip install -e .
```

## Uso rápido (línea base)

```bash
python -m grafito.cli \
  --image assets/ejemplo.jpg \
  --prompt "add a brown hat" \
  --output outputs/ejemplo_editado.jpg
```

Ver `notebooks/01_baseline_ip2pix.ipynb` para una demostración interactiva.

También puedes validar la instalación con:

```bash
PYTHONPATH=src python scripts/run_baseline.py
```

## Notas de compatibilidad

En macOS x86_64 con torch 2.2.2 se fijaron las siguientes versiones para evitar incompatibilidades:

- `diffusers==0.27.2`
- `transformers==4.39.3`
- `accelerate==0.28.0`
- `datasets==2.18.0`
- `peft==0.10.0`
- `huggingface_hub==0.23.5`
- `numpy<2`

En MPS no se usa `model_cpu_offload` por defecto porque diffusers 0.27.2 puede forzar internamente dispositivo CUDA.

## Estructura del proyecto

```
Grafito/
├── docs/                   # Documentación del proyecto
├── src/
│   ├── grafito/            # Código fuente del editor (incluye cli.py)
│   └── scripts/            # Scripts de datos, entrenamiento y evaluación
├── scripts/                # Scripts de desarrollo y validación
├── notebooks/              # Notebooks de exploración y baseline
├── data/                   # Datasets (no versionados)
├── models/checkpoints/     # Pesos entrenados (no versionados)
├── assets/                 # Imágenes de ejemplo
├── outputs/                # Resultados de inferencia (no versionado)
└── tests/                  # Tests unitarios
```

## Documentación

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — Arquitectura de software.
- [`docs/TRAINING.md`](docs/TRAINING.md) — Guía de entrenamiento.
- [`docs/EVALUATION.md`](docs/EVALUATION.md) — Protocolo de evaluación.
- [`docs/CHANGELOG.md`](docs/CHANGELOG.md) — Registro de cambios.
- [`NEXT_LOCAL_EDITOR_TRAINING.md`](NEXT_LOCAL_EDITOR_TRAINING.md) — Documento de inicio del proyecto.

## Próximos pasos

Ver la checklist de arranque en [`NEXT_LOCAL_EDITOR_TRAINING.md`](NEXT_LOCAL_EDITOR_TRAINING.md#14-checklist-de-arranque).

## Licencia

Este proyecto es un prototipo de investigación. Los modelos y datasets utilizados tienen sus propias licencias:

- Stable Diffusion v1.5: [CreativeML OpenRAIL M](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- InstructPix2Pix: [MIT](https://huggingface.co/timbrooks/instruct-pix2pix)
- MagicBrush: [CC-BY-4.0](https://huggingface.co/datasets/osunlp/MagicBrush)
