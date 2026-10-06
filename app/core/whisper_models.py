"""Whisper model catalogue, sized for a 6 GB GPU."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WhisperChoice:
    key: str
    label: str
    fp16_gb: float      # weights at float16; other precisions scale from this
    download_gb: float  # size on disk of the CTranslate2 conversion


# Ordered worst-to-best quality. These are the CTranslate2 conversions hosted by
# Systran, which faster-whisper fetches automatically on first use.
WHISPER_MODELS: list[WhisperChoice] = [
    WhisperChoice("tiny", "tiny — fastest, rough accuracy", 0.15, 0.08),
    WhisperChoice("base", "base — very fast", 0.25, 0.15),
    WhisperChoice("small", "small — fast, decent accuracy", 0.8, 0.5),
    WhisperChoice("medium", "medium — good accuracy", 1.6, 1.6),
    WhisperChoice("large-v3", "large-v3 — best accuracy", 3.1, 3.1),
    WhisperChoice(
        "distil-large-v3",
        "distil-large-v3 — near-large quality, ~2x faster (English only)",
        1.9,
        1.6,
    ),
]

# Rough multipliers against the float16 weight size.
_PRECISION_SCALE = {
    "float32": 2.0,
    "float16": 1.0,
    "int8_float16": 0.55,
    "int8_float32": 0.55,
    "int8": 0.5,
}


def find_model(key: str) -> WhisperChoice | None:
    return next((m for m in WHISPER_MODELS if m.key == key), None)


def repo_id(key: str) -> str:
    """The Hugging Face repo faster-whisper pulls a given size from."""
    if key.startswith("distil-"):
        return f"Systran/faster-distil-whisper-{key[len('distil-'):]}"
    return f"Systran/faster-whisper-{key}"


def cache_dir_name(key: str) -> str:
    """The folder the Hugging Face cache stores that repo in."""
    return "models--" + repo_id(key).replace("/", "--")


def is_downloaded(key: str, root) -> bool:
    """True when the weights for this size are already on disk under root."""
    from pathlib import Path

    folder = Path(root) / cache_dir_name(key)
    return folder.is_dir() and any(folder.rglob("model.bin"))


def estimate_vram_gb(model_key: str, compute_type: str) -> float | None:
    """Approximate VRAM for the weights, before activations and audio buffers."""
    choice = find_model(model_key)
    if choice is None:
        return None
    return choice.fp16_gb * _PRECISION_SCALE.get(compute_type, 1.0)

DEFAULT_MODEL = "large-v3"

COMPUTE_TYPES = {
    "cuda": [
        ("int8_float16", "int8_float16 — fastest, lowest VRAM (recommended)"),
        ("float16", "float16 — highest fidelity"),
        ("int8", "int8 — smallest footprint"),
    ],
    "cpu": [
        ("int8", "int8 — recommended on CPU"),
        ("float32", "float32 — slow, highest fidelity"),
    ],
}

# Whisper's own language list, trimmed to the ones people actually pick. The UI
# also offers auto-detect, which is the default.
LANGUAGES: list[tuple[str, str]] = [
    ("", "Auto-detect"),
    ("en", "English"),
    ("hi", "Hindi"),
    ("ta", "Tamil"),
    ("te", "Telugu"),
    ("kn", "Kannada"),
    ("ml", "Malayalam"),
    ("mr", "Marathi"),
    ("bn", "Bengali"),
    ("gu", "Gujarati"),
    ("pa", "Punjabi"),
    ("ur", "Urdu"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("de", "German"),
    ("it", "Italian"),
    ("pt", "Portuguese"),
    ("nl", "Dutch"),
    ("ru", "Russian"),
    ("uk", "Ukrainian"),
    ("pl", "Polish"),
    ("tr", "Turkish"),
    ("ar", "Arabic"),
    ("fa", "Persian"),
    ("he", "Hebrew"),
    ("zh", "Chinese"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("vi", "Vietnamese"),
    ("th", "Thai"),
    ("id", "Indonesian"),
]
