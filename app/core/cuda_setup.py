"""Make the NVIDIA CUDA/cuDNN DLLs shipped in pip wheels visible to CTranslate2.

faster-whisper -> CTranslate2 needs cuBLAS and cuDNN 9 at load time. We install
those as pip packages (nvidia-cublas-cu12, nvidia-cudnn-cu12) so the user never
has to install the CUDA Toolkit. On Windows the DLLs live in
site-packages/nvidia/<lib>/bin, which is not on the default search path, so we
register the directories before anything imports ctranslate2.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_done = False


def _nvidia_dll_dirs() -> list[Path]:
    dirs: list[Path] = []
    for entry in sys.path:
        nvidia_root = Path(entry) / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for lib_dir in sorted(nvidia_root.iterdir()):
            for sub in ("bin", "lib"):
                candidate = lib_dir / sub
                if candidate.is_dir():
                    dirs.append(candidate)
    return dirs


def ensure_cuda_dlls() -> list[str]:
    """Register bundled CUDA DLL directories. Safe to call repeatedly."""
    global _done
    if _done:
        return []
    _done = True

    registered: list[str] = []
    for path in _nvidia_dll_dirs():
        text = str(path)
        if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(text)
                registered.append(text)
            except OSError:
                continue
        os.environ["PATH"] = text + os.pathsep + os.environ.get("PATH", "")
    return registered


def cuda_available() -> bool:
    """True when CTranslate2 can actually see a usable CUDA device."""
    ensure_cuda_dlls()
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False
