#!/bin/bash
# Limpia artefactos descargados una vez que el dataset está procesado y el entrenamiento corre.
# Revisa cada comando antes de ejecutar; descomenta los que quieras aplicar.

# Backup antiguo del entorno virtual (seguro de borrar si .venv actual funciona)
rm -rf /workspace/Grafito/.venv-backups

# Dataset raw de MagicBrush (~26 GB). Solo borrar si ya no necesitas reprocesar.
# rm -rf /workspace/Grafito/data/raw/magicbrush
