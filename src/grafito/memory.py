"""Gestión de dispositivo y memoria para inferencia local."""

import torch


def get_device() -> str:
    """Selecciona el mejor dispositivo disponible: cuda > mps > cpu.

    Algunos entornos reportan MPS disponible pero no funcional (por ejemplo,
    macOS x86_64 virtualizado). Se hace una prueba rápida con un tensor antes
    de decidirse por MPS.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        try:
            _ = torch.zeros(1, device="mps")
            return "mps"
        except Exception:
            pass
    return "cpu"


def get_torch_dtype(device: str, prefer_float16: bool = True):
    """Devuelve el dtype adecuado para el dispositivo."""
    if device == "cuda" and prefer_float16:
        return torch.float16
    return torch.float32


def get_offload_strategy(device: str, memory_gb: float | None = None) -> str | None:
    """Decide si usar CPU offload según el dispositivo y memoria disponible.

    Nota: en MPS el model_cpu_offload puede forzar dispositivos CUDA internamente
    en algunas versiones de diffusers, así que por defecto no se usa.
    """
    if device == "cpu":
        return "sequential_cpu_offload"
    if device == "cuda" and memory_gb is not None and memory_gb < 10:
        return "model_cpu_offload"
    return None
