# Guía de entrenamiento

## Requisitos

- Python >= 3.10
- GPU con al menos 24 GB de VRAM (RTX 3090/4090, A10, A100)
- Entorno virtual con dependencias instaladas

### Versiones de dependencias

Para reproducibilidad en macOS x86_64 con torch 2.2.2, el proyecto fija:

```text
torch==2.2.2
diffusers==0.27.2
transformers==4.39.3
accelerate==0.28.0
datasets==2.18.0
peft==0.10.0
huggingface_hub==0.23.5
numpy<2
```

En el entorno de entrenamiento (Linux/CUDA) pueden usarse versiones más recientes, pero se recomienda mantener `diffusers` y `transformers` compatibles.

## Preparar datos

```bash
python src/scripts/prepare_magicbrush.py --resolution 256
```

El script:

1. Descarga `osunlp/MagicBrush`.
2. Renombra las columnas a `original_image`, `edited_image` y `edit_prompt`.
3. Filtra ejemplos con imágenes corruptas o prompts vacíos.
4. Redimensiona el lado mayor de cada imagen a `--resolution` manteniendo el aspect ratio.
5. Guarda el dataset como un `DatasetDict` de Arrow en `data/processed/magicbrush/`.
6. Escribe `stats.json` con el número de ejemplos válidos por split.

Esto genera `data/processed/magicbrush/` con los splits `train` y `validation`, listo para cargar desde disco local.

Para acelerar el preprocesamiento en máquinas con varios núcleos:

```bash
python src/scripts/prepare_magicbrush.py --resolution 256 --num_proc 4
```

Si la descarga directa desde HuggingFace falla por problemas de red, descarga primero los parquet localmente:

```bash
huggingface-cli download osunlp/MagicBrush --repo-type dataset --local-dir data/raw/magicbrush
```

`prepare_magicbrush.py` detectará automáticamente los archivos en `data/raw/magicbrush/data` y los usará.

## Script de entrenamiento

El proyecto incluye el script oficial de Diffusers en `src/scripts/train_instruct_pix2pix.py` con un parche mínimo: detecta si `--dataset_name` apunta a un directorio local con `dataset_dict.json` y lo carga con `load_from_disk`, de forma que el dataset procesado por `prepare_magicbrush.py` sea directamente usable.

Si prefieres usar el script original de Diffusers, clónalo desde el repositorio oficial y aplica el mismo parche, o guarda el dataset en otro formato compatible con `--train_data_dir`.

## Comando de entrenamiento base

```bash
export MODEL_NAME="timbrooks/instruct-pix2pix"
export DATASET_NAME="data/processed/magicbrush"
export OUTPUT_DIR="models/checkpoints/grafito-v1"

accelerate launch --mixed_precision="fp16" src/scripts/train_instruct_pix2pix.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --original_image_column="original_image" \
  --edited_image_column="edited_image" \
  --edit_prompt_column="edit_prompt" \
  --resolution=256 \
  --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=5000 \
  --checkpointing_steps=1000 \
  --checkpoints_total_limit=2 \
  --learning_rate=5e-5 \
  --max_grad_norm=1 \
  --lr_warmup_steps=0 \
  --conditioning_dropout_prob=0.05 \
  --mixed_precision=fp16 \
  --seed=42 \
  --output_dir=$OUTPUT_DIR
```

## Subir a 512 px

Una vez validado el entrenamiento a 256 px, repetir con `--resolution=512` y posiblemente más pasos.

## Monitoreo

Se recomienda añadir:

```bash
--report_to=wandb \
--val_image_url="https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png" \
--validation_prompt="make the mountains snowy"
```

## Notas

- El script oficial entrena todo el UNet. No es LoRA.
- Si la VRAM es insuficiente, reducir `--resolution` o `--train_batch_size` y aumentar `--gradient_accumulation_steps`.
