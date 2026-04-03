from __future__ import annotations

from dataclasses import dataclass
import os

import torch


HEAVY_ARCHES = {"vit_b16", "swin_tiny", "convnext_tiny", "efficientnet_b3"}


@dataclass
class RuntimeProfile:
    device: torch.device
    gpu_name: str | None
    vram_gb: float | None
    cpu_count: int


def detect_runtime() -> RuntimeProfile:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return RuntimeProfile(
            device=torch.device("cuda"),
            gpu_name=torch.cuda.get_device_name(0),
            vram_gb=float(props.total_memory / 1e9),
            cpu_count=os.cpu_count() or 4,
        )

    return RuntimeProfile(
        device=torch.device("cpu"),
        gpu_name=None,
        vram_gb=None,
        cpu_count=os.cpu_count() or 4,
    )


def suggest_batch_size(vram_gb: float | None, arches: list[str]) -> int:
    # choose a stable default for mixed architecture runs.
    if vram_gb is None:
        base = 8
    elif vram_gb >= 20:
        base = 64
    elif vram_gb >= 12:
        base = 48
    elif vram_gb >= 8:
        base = 32
    elif vram_gb >= 6:
        base = 24
    elif vram_gb >= 4:
        base = 16
    else:
        base = 8

    if any(arch in HEAVY_ARCHES for arch in arches):
        base = max(8, base // 2)

    return int(base)


def suggest_num_workers(cpu_count: int, on_gpu: bool) -> int:
    if cpu_count <= 2:
        return 0
    if on_gpu:
        return min(8, max(2, cpu_count - 1))
    return min(4, max(1, cpu_count - 1))
