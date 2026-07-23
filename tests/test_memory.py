"""Tests para memory."""

from grafito.memory import get_device, get_offload_strategy, get_torch_dtype


def test_get_device_returns_valid_device():
    device = get_device()
    assert device in ("cuda", "mps", "cpu")


def test_get_torch_dtype_for_cpu_is_float32():
    dtype = get_torch_dtype("cpu")
    assert str(dtype) == "torch.float32"


def test_get_offload_strategy_for_cpu():
    assert get_offload_strategy("cpu") == "sequential_cpu_offload"


def test_get_offload_strategy_for_mps_is_none():
    assert get_offload_strategy("mps") is None
