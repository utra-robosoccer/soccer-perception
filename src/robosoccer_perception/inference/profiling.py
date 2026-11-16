"""Latency and memory measurement shared by the evaluation scripts.

Numbers depend on the machine, on what else is using the GPU, and on power settings; always
report them together with `hardware_summary()`.
"""
from __future__ import annotations

import os
import platform
import time
from typing import Any, Callable, Iterable

import numpy as np


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def cuda_available() -> bool:
    torch = _torch()
    return bool(torch is not None and torch.cuda.is_available())


def resolve_device(device: str | int | None = "auto") -> str:
    """``auto`` -> ``cuda:0`` when a GPU is usable, else ``cpu``. Other values pass through."""
    if device in (None, "auto"):
        return "cuda:0" if cuda_available() else "cpu"
    return f"cuda:{device}" if isinstance(device, int) or str(device).isdigit() else str(device)


def synchronize(device: str) -> None:
    """Wait for queued GPU work, so that wall-clock timings include it."""
    if device.startswith("cuda"):
        _torch().cuda.synchronize()


def hardware_summary(device: str = "auto") -> dict[str, Any]:
    device = resolve_device(device)
    summary: dict[str, Any] = {"device": device, "platform": platform.platform(),
                               "cpu": platform.processor() or platform.machine(),
                               "cpu_count": os.cpu_count()}
    torch = _torch()
    if torch is not None:
        summary["torch"] = torch.__version__
        if device.startswith("cuda"):
            index = torch.device(device).index or 0
            properties = torch.cuda.get_device_properties(index)
            summary["gpu"] = properties.name
            summary["gpu_memory_total_mb"] = round(properties.total_memory / 2**20)
    return summary


def process_memory_mb() -> float | None:
    """Resident memory of this process, MiB (None without psutil)."""
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / 2**20


def reset_gpu_peak(device: str) -> None:
    if device.startswith("cuda"):
        _torch().cuda.reset_peak_memory_stats()


def gpu_memory_mb(device: str) -> dict[str, float] | None:
    """Peak GPU memory since `reset_gpu_peak`: what PyTorch tensors used (allocated) and what
    PyTorch held from the driver (reserved). The CUDA context itself (a few hundred MiB) is not
    included; `nvidia-smi` shows the total."""
    if not device.startswith("cuda"):
        return None
    torch = _torch()
    return {"peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20}


def latency_stats(samples_ms: Iterable[float]) -> dict[str, float]:
    samples = np.asarray(list(samples_ms), dtype=float)
    if samples.size == 0:
        return {"count": 0}
    return {"count": int(samples.size), "mean_ms": float(samples.mean()),
            "median_ms": float(np.median(samples)), "p95_ms": float(np.percentile(samples, 95)),
            "min_ms": float(samples.min()), "max_ms": float(samples.max()),
            "fps_at_mean": float(1000.0 / samples.mean())}


def measure_latency(function: Callable[[Any], Any], inputs: list, device: str = "cpu",
                    warmup: int = 5, repeats: int = 50) -> dict[str, float]:
    """Time ``function(x)`` one input at a time, cycling through ``inputs``.

    The first ``warmup`` calls are discarded (lazy initialisation, cuDNN autotuning). Each
    timed call is bracketed by a GPU synchronisation.
    """
    device = resolve_device(device)
    for index in range(warmup):
        function(inputs[index % len(inputs)])
    samples = []
    for index in range(repeats):
        item = inputs[index % len(inputs)]
        synchronize(device)
        started = time.perf_counter()
        function(item)
        synchronize(device)
        samples.append((time.perf_counter() - started) * 1000.0)
    return latency_stats(samples)
