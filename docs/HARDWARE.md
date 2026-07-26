# Hardware soportado por el modelo

El modelo v2 será el mismo tipo de artefacto que v1 y que el base: un pipeline **InstructPix2Pix / Stable Diffusion 1.5** estándar (~4 GB en disco, safetensors, se carga con diffusers). Eso define dónde corre. Te lo desgloso por clase de máquina:

## Objetivo de diseño (garantizado)

- **Tu iMac** (i7-10700K, 32 GB RAM, Radeon 5500 XT 8 GB, MPS): es la restricción del plan. Ya validado: 256 px en MPS (~12 s/20 pasos) y 512 px en CPU (2 min 4 s). En MPS a 512 irá con `attention_slicing` — la verificación final es la tarea C5 del runbook.

## Otras máquinas donde correrá

| Tipo de máquina | Cómo corre | Expectativa |
|---|---|---|
| Mac Apple Silicon (M1/M2/M3/M4) ≥ 16 GB | MPS, fp32 | Mejor que en tu Radeon (memoria unificada); 512 cómodo |
| PC con NVIDIA ≥ 8 GB VRAM (3060, 3070, 4060…) | CUDA, **fp16** (~5 GB VRAM a 512) | El caso ideal: segundos por imagen |
| PC con NVIDIA 6 GB | CUDA fp16 + attention slicing | 512 justo; 256/384 cómodo |
| PC/Mac solo CPU, ≥ 16 GB RAM | CPU fp32 | Funciona pero lento (~2 min a 512 en tu i7) — sirve para demo, no interactivo |
| Nube (Vast.ai, RunPod, HF Spaces) | Cualquier GPU moderna | Trivial; es donde se entrena |

## Donde NO correrá (realistamente)

- **GPUs con < 6 GB de VRAM** para 512 px (las de 4 GB quedan descartadas salvo a 256 con slicing agresivo).
- **Móviles / edge**: no es un modelo para eso.
- Nada que exija compilar torch raro: basta Python 3.10+, torch, diffusers, transformers.

## La ventaja clave del formato

Al ser un pipeline IP2P/SD 1.5 estándar (no un formato exótico nuestro), el checkpoint también lo puede cargar **ComfyUI, AUTOMATIC1111** y cualquier herramienta del ecosistema que soporte InstructPix2Pix — útil si algún día lo muestras o lo usas fuera de nuestro `demo.py`.

Nota honesta: la única celda de la tabla aún no verificada al 100% es **MPS a 512 px en tu iMac** (la puerta 0.1 la corrí en CPU). Si ahí apareciera OOM, el fallback del plan es trabajar a 384 en MPS y 512 en CPU — sin cambiar el modelo.