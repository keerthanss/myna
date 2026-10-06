"""faster-whisper wrapper with progress reporting and cancellation."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from . import whisper_models
from .cuda_setup import cuda_available, ensure_cuda_dlls
from .errors import Cancelled
from .model_cache import download_with_watchdog
from .paths import whisper_models_dir
from .subtitles import Segment, Word

__all__ = [
    "Cancelled",
    "TranscribeOptions",
    "TranscribeResult",
    "Transcriber",
    "MEDIA_EXTENSIONS",
    "is_media_file",
    "resolve_device",
    "resolve_compute_type",
]


@dataclass
class TranscribeOptions:
    model: str = "large-v3"
    device: str = "auto"          # auto | cuda | cpu
    compute_type: str = "auto"
    language: str = ""            # "" means auto-detect
    task: str = "transcribe"      # transcribe | translate
    beam_size: int = 5
    vad_filter: bool = True
    initial_prompt: str = ""


@dataclass
class TranscribeResult:
    segments: list[Segment]
    language: str
    language_probability: float
    duration: float


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda"
    return "cuda" if cuda_available() else "cpu"


def resolve_compute_type(device: str, requested: str) -> str:
    if requested and requested != "auto":
        return requested
    return "int8_float16" if device == "cuda" else "int8"


class Transcriber:
    """Holds one loaded model so consecutive files do not pay the load cost twice."""

    def __init__(self) -> None:
        self._model = None
        self._signature: tuple[str, str, str] | None = None
        self._lock = threading.Lock()

    def load(
        self,
        options: TranscribeOptions,
        log: Callable[[str], None] = lambda _m: None,
        on_download: Callable[[float, str], None] | None = None,
        should_cancel: Callable[[], bool] = lambda: False,
    ) -> tuple[str, str]:
        ensure_cuda_dlls()
        device = resolve_device(options.device)
        compute_type = resolve_compute_type(device, options.compute_type)
        signature = (options.model, device, compute_type)

        with self._lock:
            if self._signature == signature and self._model is not None:
                return device, compute_type

            from faster_whisper import WhisperModel

            self.unload()
            root = whisper_models_dir()
            choice = whisper_models.find_model(options.model)
            needs_download = choice is not None and not whisper_models.is_downloaded(
                options.model, root
            )
            if needs_download:
                # Downloaded first, as its own killable step: a single stuck
                # CDN connection has been seen to hang forever with no error,
                # and nothing inside a plain WhisperModel(...) call can
                # recover from that. See model_cache.download_with_watchdog.
                log(
                    f"Downloading Whisper '{options.model}' "
                    f"(~{choice.download_gb:.1f} GB). This happens once."
                )
                download_with_watchdog(
                    "whisper",
                    options.model,
                    root,
                    int(choice.download_gb * 1024 ** 3),
                    on_progress=on_download or (lambda _f, _m: None),
                    should_cancel=should_cancel,
                    log=log,
                )
            log(f"Loading Whisper '{options.model}' on {device.upper()} ({compute_type}).")

            def build() -> None:
                self._model = WhisperModel(
                    options.model,
                    device=device,
                    compute_type=compute_type,
                    download_root=str(root),
                )

            try:
                build()
            except Cancelled:
                raise
            except Exception as exc:
                if device != "cuda":
                    raise
                log(f"CUDA load failed ({exc}). Falling back to CPU.")
                device, compute_type = "cpu", resolve_compute_type("cpu", "auto")
                self._model = WhisperModel(
                    options.model,
                    device=device,
                    compute_type=compute_type,
                    download_root=str(root),
                )
                signature = (options.model, device, compute_type)
            self._signature = signature
        return device, compute_type

    def unload(self) -> None:
        """Free VRAM — important before the summariser loads its own model."""
        if self._model is not None:
            del self._model
            self._model = None
            self._signature = None
            import gc

            gc.collect()

    def transcribe(
        self,
        media_path: Path,
        options: TranscribeOptions,
        on_progress: Callable[[float], None] = lambda _p: None,
        on_segment: Callable[[Segment], None] = lambda _s: None,
        should_cancel: Callable[[], bool] = lambda: False,
        log: Callable[[str], None] = lambda _m: None,
        on_download: Callable[[float, str], None] | None = None,
    ) -> TranscribeResult:
        self.load(options, log=log, on_download=on_download, should_cancel=should_cancel)
        assert self._model is not None

        segment_iter, info = self._model.transcribe(
            str(media_path),
            language=options.language or None,
            task=options.task,
            beam_size=options.beam_size,
            vad_filter=options.vad_filter,
            vad_parameters={"min_silence_duration_ms": 500} if options.vad_filter else None,
            word_timestamps=True,
            initial_prompt=options.initial_prompt or None,
            condition_on_previous_text=False,
        )

        duration = float(getattr(info, "duration", 0.0) or 0.0)
        log(
            f"Language: {info.language} "
            f"({info.language_probability:.0%} confidence) · "
            f"{duration / 60:.1f} min of audio"
        )

        collected: list[Segment] = []
        for raw in segment_iter:
            if should_cancel():
                raise Cancelled()
            words = [
                Word(start=w.start, end=w.end, text=w.word)
                for w in (raw.words or [])
                if w.start is not None and w.end is not None
            ]
            segment = Segment(start=raw.start, end=raw.end, text=raw.text, words=words)
            collected.append(segment)
            on_segment(segment)
            if duration > 0:
                on_progress(min(0.99, raw.end / duration))

        on_progress(1.0)
        return TranscribeResult(
            segments=collected,
            language=info.language,
            language_probability=float(info.language_probability or 0.0),
            duration=duration,
        )


VIDEO_EXTENSIONS: Iterable[str] = (
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg", ".ts",
)
AUDIO_EXTENSIONS: Iterable[str] = (
    ".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma",
)
MEDIA_EXTENSIONS = tuple(VIDEO_EXTENSIONS) + tuple(AUDIO_EXTENSIONS)


def is_media_file(path: Path) -> bool:
    return path.suffix.lower() in MEDIA_EXTENSIONS
