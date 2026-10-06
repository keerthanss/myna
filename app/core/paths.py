"""Where Myna keeps its data, models, settings and logs.

Everything lives inside the installation folder by default, so the whole app is
one self-contained directory that can sit on any drive and be deleted in one go.
Nothing is written to the system drive unless the app was installed there.

That includes the Hugging Face cache. Its default location is under the user
profile on C:, and hf-xet stages multi-gigabyte downloads there, so HF_HOME is
redirected here before huggingface_hub is ever imported.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .branding import DATA_DIR_NAME, ICON_FILENAME

APP_NAME = DATA_DIR_NAME

# .../<install dir>/app/core/paths.py  ->  <install dir>
INSTALL_DIR = Path(__file__).resolve().parents[2]


def icon_path() -> Path | None:
    """Where the app icon lives, whether installed or run from source.

    Shared by main.py (window/taskbar icon) and ui/main_window.py (the header
    logo mark) so both always agree on the same file without importing each
    other - this module has no Qt dependency, so neither side risks a
    circular import.
    """
    for candidate in (
        INSTALL_DIR / "assets" / ICON_FILENAME,
        Path(__file__).resolve().parent.parent.parent / "assets" / ICON_FILENAME,
    ):
        if candidate.is_file():
            return candidate
    return None


def _writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def data_dir() -> Path:
    """The app's own data folder, normally <install dir>/data."""
    override = os.environ.get("WHISPER_STUDIO_DATA")
    if override:
        path = Path(override)
        if _writable(path):
            return path

    beside_app = INSTALL_DIR / "data"
    if _writable(beside_app):
        return beside_app

    # Only reached if the app was installed somewhere read-only.
    fallback = Path(
        os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.local/share")
    ) / APP_NAME
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def config_file() -> Path:
    return data_dir() / "config.json"


def settings_file() -> Path:
    """UI preferences as an INI file rather than the Windows registry."""
    return data_dir() / "settings.ini"


def _read_config() -> dict:
    try:
        return json.loads(config_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_config(config: dict) -> None:
    config_file().write_text(json.dumps(config, indent=2), encoding="utf-8")


def default_models_root() -> Path:
    return data_dir() / "models"


def models_root() -> Path:
    configured = _read_config().get("models_root")
    if configured:
        return Path(configured)
    return default_models_root()


def set_models_root(path: Path | str | None) -> None:
    """Point the model folder somewhere else, or None to restore the default."""
    config = _read_config()
    if path is None:
        config.pop("models_root", None)
    else:
        config["models_root"] = str(Path(path))
    _write_config(config)
    _redirect_hf_cache()


def whisper_models_dir() -> Path:
    path = models_root() / "whisper"
    path.mkdir(parents=True, exist_ok=True)
    return path


def llm_models_dir() -> Path:
    path = models_root() / "llm"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _redirect_hf_cache() -> None:
    """Keep Hugging Face's caches next to the models, not under C:\\Users.

    huggingface_hub reads these at import time, and hf-xet stages entire files
    in its cache before moving them into place - several GB per model - so
    leaving it on the default path silently fills the system drive.
    """
    cache = models_root() / "hf-cache"
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    os.environ["HF_HOME"] = str(cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache / "hub")
    os.environ["HF_XET_CACHE"] = str(cache / "xet")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def free_space_gb(path: Path | str) -> float:
    """Free gigabytes on the volume holding path, walking up if it is missing."""
    candidate = Path(path)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    try:
        return shutil.disk_usage(candidate).free / (1024 ** 3)
    except OSError:
        return 0.0


def log_file() -> Path:
    return data_dir() / "whisper-studio.log"


_redirect_hf_cache()
