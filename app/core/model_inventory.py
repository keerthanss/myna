"""What Myna has downloaded, and safe removal of it.

Backs the "Manage downloaded models" dialog. Kept free of Qt so it can be
tested headlessly, and every function takes the models root as a parameter so
tests run against a temporary folder rather than the real installation.

Deletion is confined to the models folder: any path that resolves outside it -
through a symlink, a junction, or a bug - is refused rather than removed.
"""

from __future__ import annotations

import os
import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path

from . import llm_models, whisper_models
from .paths import models_root

KIND_WHISPER = "Transcription"
KIND_LLM = "Summary"
KIND_DIARIZATION = "Speakers"
KIND_LEFTOVER = "Leftovers"

LEFTOVER_MIN_BYTES = 1024 * 1024


@dataclass
class InstalledModel:
    kind: str
    key: str                  # catalogue key, or a descriptive id
    label: str
    paths: list[Path] = field(default_factory=list)
    size_bytes: int = 0
    note: str = ""            # "Incomplete download", "Not in Myna's list", ...


def _size(path: Path) -> int:
    """Bytes actually occupied on disk, counting each file once.

    Symlinks are skipped, not followed. The Hugging Face cache stores each
    file once under blobs/ and points to it from snapshots/ with a link;
    following those links counts every Whisper model twice - a 2.9 GB model
    reported as 5.8 GB, and a deletion promising to free double what it does.
    """
    try:
        if path.is_symlink():
            return 0
        if path.is_file():
            return path.lstat().st_size
    except OSError:
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_symlink():
                continue
            if item.is_file():
                total += item.lstat().st_size
        except OSError:
            continue
    return total


def _whisper_entries(root: Path) -> list[InstalledModel]:
    folder = root / "whisper"
    if not folder.is_dir():
        return []

    by_folder = {
        whisper_models.cache_dir_name(m.key): m for m in whisper_models.WHISPER_MODELS
    }
    entries: list[InstalledModel] = []
    for path in sorted(folder.iterdir()):
        if not path.is_dir() or not path.name.startswith("models--"):
            continue
        size = _size(path)
        choice = by_folder.get(path.name)
        if choice is None:
            repo = path.name[len("models--"):].replace("--", "/")
            entries.append(
                InstalledModel(
                    KIND_WHISPER, path.name, f"Whisper ({repo})", [path], size,
                    note="Not in Myna's list",
                )
            )
            continue
        complete = whisper_models.is_downloaded(choice.key, folder)
        entries.append(
            InstalledModel(
                KIND_WHISPER, choice.key, f"Whisper {choice.key}", [path], size,
                note="" if complete else "Incomplete download",
            )
        )
    return entries


def _llm_entries(root: Path) -> list[InstalledModel]:
    folder = root / "llm"
    if not folder.is_dir():
        return []

    by_file = {c.filename: c for c in llm_models.LLM_MODELS}
    entries: list[InstalledModel] = []
    for path in sorted(folder.glob("*.gguf")):
        choice = by_file.get(path.name)
        if choice is None:
            entries.append(
                InstalledModel(
                    KIND_LLM, path.name, path.stem, [path], _size(path),
                    note="Not in Myna's list",
                )
            )
        else:
            label = choice.label.split(" — ")[0]
            entries.append(
                InstalledModel(KIND_LLM, choice.key, label, [path], _size(path))
            )
    return entries


def _diarization_entry(root: Path) -> list[InstalledModel]:
    folder = root / "diarization"
    if not folder.is_dir():
        return []
    size = _size(folder)
    if size == 0:
        return []
    return [
        InstalledModel(
            KIND_DIARIZATION, "diarization", "Speaker identification models",
            [folder], size,
        )
    ]


def _leftover_entry(root: Path) -> list[InstalledModel]:
    """Download staging and abandoned partial files - never needed afterwards.

    The hf-cache folder itself is kept (it is created at startup and pointed
    at by environment variables); only what is inside it goes.
    """
    paths: list[Path] = []
    hf_cache = root / "hf-cache"
    if hf_cache.is_dir():
        paths.extend(sorted(hf_cache.iterdir()))
    for candidate in (root / "llm" / ".cache", root / "whisper" / ".locks"):
        if candidate.exists():
            paths.append(candidate)

    size = sum(_size(p) for p in paths)
    # Below this it is only stale lock files, which every run already sweeps
    # (model_cache.clear_stale_locks). A "0 MB" row would only confuse.
    if size < LEFTOVER_MIN_BYTES:
        return []
    return [
        InstalledModel(
            KIND_LEFTOVER, "leftovers", "Download staging and partial files",
            paths, size,
        )
    ]


def list_installed(root: Path | None = None) -> list[InstalledModel]:
    """Everything Myna has downloaded under the models folder, largest first
    within each kind."""
    root = Path(root) if root is not None else models_root()
    if not root.is_dir():
        return []
    groups = [
        _whisper_entries(root),
        _llm_entries(root),
        _diarization_entry(root),
        _leftover_entry(root),
    ]
    result: list[InstalledModel] = []
    for group in groups:
        result.extend(sorted(group, key=lambda m: m.size_bytes, reverse=True))
    return result


def _clear_readonly_and_retry(function, path, _exc_info) -> None:
    """rmtree error hook: Windows refuses to delete read-only files."""
    try:
        os.chmod(path, stat.S_IWRITE)
        function(path)
    except OSError:
        raise


def _inside(path: Path, root: Path) -> bool:
    return path != root and root in path.parents


def delete(
    items: list[InstalledModel], root: Path | None = None
) -> tuple[int, list[str]]:
    """Remove the given models. Returns (bytes freed, error messages).

    Never raises for an individual failure: one file in use should not stop
    the rest from being removed, and the caller reports what could not go.
    """
    root = (Path(root) if root is not None else models_root()).resolve()
    freed = 0
    errors: list[str] = []

    for item in items:
        for path in item.paths:
            try:
                resolved = path.resolve()
            except OSError as exc:
                errors.append(f"{path.name}: {exc}")
                continue
            if not _inside(resolved, root):
                errors.append(
                    f"Refused to delete {path} — it is outside the model folder."
                )
                continue
            if not resolved.exists():
                continue

            before = _size(resolved)
            try:
                if resolved.is_dir():
                    shutil.rmtree(resolved, onerror=_clear_readonly_and_retry)
                else:
                    try:
                        resolved.unlink()
                    except PermissionError:
                        os.chmod(resolved, stat.S_IWRITE)
                        resolved.unlink()
            except OSError as exc:
                errors.append(f"{item.label}: {exc}")
            remaining = _size(resolved) if resolved.exists() else 0
            freed += max(0, before - remaining)

    return freed, errors


def human_size(size_bytes: int) -> str:
    gb = size_bytes / (1024 ** 3)
    if gb >= 0.1:
        return f"{gb:.2f} GB"
    mb = size_bytes / (1024 ** 2)
    if mb >= 1:
        return f"{mb:.0f} MB"
    return f"{max(1, round(size_bytes / 1024))} KB"
