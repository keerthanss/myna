"""Decide how much of a GGUF model to put on the GPU.

Models larger than VRAM are explicitly supported: llama.cpp keeps the layers
that do not fit in system RAM and runs them on the CPU. That is slower per
token but lets a 14B model produce far better summaries than a 3B that fits
entirely on a 6 GB card.

Picking the split up front matters. The alternative - try a full offload,
fail, retry with fewer layers - costs a complete load attempt each time, and a
load attempt on a 9 GB model means reading 9 GB from disk before finding out.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .gguf_meta import read_info

# Held back from the layer budget: the CUDA context, the KV cache for the
# chosen context length, and compute buffers. Measured generously - running a
# few layers short on the GPU costs some speed, running out of VRAM costs the
# whole load.
VRAM_RESERVE_GB = 1.1

# Windows plus background services never give this back, so it is excluded
# when judging whether a model can fit in RAM at all.
OS_RAM_RESERVE_GB = 4.0


def free_vram_gb() -> float | None:
    """Free VRAM in GB, or None when it cannot be determined."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        if result.returncode != 0:
            return None
        first = result.stdout.strip().splitlines()[0]
        return float(first.strip()) / 1024.0
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def total_ram_gb() -> float | None:
    """Total physical RAM in GB, or None when it cannot be determined."""
    return _ram_gb(total=True)


def free_ram_gb() -> float | None:
    """Free physical RAM in GB, or None when it cannot be determined.

    Uses the Win32 API directly rather than adding psutil for one number.
    """
    return _ram_gb(total=False)


def _ram_gb(total: bool) -> float | None:
    if sys.platform == "win32":
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            raw = status.ullTotalPhys if total else status.ullAvailPhys
            return raw / (1024 ** 3)
        except Exception:  # noqa: BLE001 - advisory only
            return None

    try:
        pages = "SC_PHYS_PAGES" if total else "SC_AVPHYS_PAGES"
        return os.sysconf(pages) * os.sysconf("SC_PAGE_SIZE") / (1024 ** 3)
    except (AttributeError, ValueError, OSError):
        return None


def plan_offload(
    model_path: Path,
    declared_layers: int = 0,
    free_gb: float | None = None,
) -> tuple[int, str]:
    """How many layers to place on the GPU, and a one-line explanation.

    Returns (n_gpu_layers, reason). -1 means "all layers", which is what
    llama.cpp expects when the whole model fits.
    """
    try:
        size_gb = model_path.stat().st_size / (1024 ** 3)
    except OSError:
        return -1, "could not size the model; attempting a full GPU offload"

    if free_gb is None:
        free_gb = free_vram_gb()
    if free_gb is None:
        return 0, "no GPU detected; running on the CPU"

    budget_gb = free_gb - VRAM_RESERVE_GB
    if budget_gb <= 0:
        return 0, f"only {free_gb:.1f} GB VRAM free; running on the CPU"

    # Comfortably fits: let llama.cpp place everything.
    if size_gb <= budget_gb:
        return -1, f"fits in VRAM ({size_gb:.1f} GB of {free_gb:.1f} GB free)"

    layers = declared_layers or read_info(model_path).layers
    if layers <= 0:
        # Without a layer count there is nothing to proportion, and a full
        # offload would fail. The loader's fallback ladder takes it from here.
        return 0, "layer count unknown; starting on the CPU"

    per_layer_gb = size_gb / layers
    fitting = int(budget_gb / per_layer_gb)
    fitting = max(0, min(layers, fitting))
    return (
        fitting,
        f"{size_gb:.1f} GB model, {free_gb:.1f} GB VRAM free — "
        f"{fitting} of {layers} layers on the GPU, the rest on the CPU",
    )


def estimated_ram_need_gb(model_path: Path, n_gpu_layers: int, declared_layers: int = 0) -> float:
    """Roughly how much system RAM the CPU-resident part will occupy."""
    try:
        size_gb = model_path.stat().st_size / (1024 ** 3)
    except OSError:
        return 0.0
    if n_gpu_layers == -1:
        return 0.0
    layers = declared_layers or read_info(model_path).layers
    if layers <= 0:
        return size_gb
    on_cpu = max(0, layers - n_gpu_layers)
    return size_gb * (on_cpu / layers)
