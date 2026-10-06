"""Background worker that runs transcription (and optional summarising) for a queue."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from . import diarization, llm_models, subtitles, whisper_models
from .model_cache import clear_stale_locks, download_with_watchdog
from .gpu_budget import OS_RAM_RESERVE_GB, free_ram_gb, free_vram_gb, total_ram_gb
from .paths import free_space_gb, llm_models_dir, models_root, whisper_models_dir
from .pdf_report import ReportMeta, build_chronological_pdf, build_pdf
from .summarizer import Summarizer, SummaryResult
from .transcriber import Cancelled, TranscribeOptions, Transcriber


@dataclass
class JobSettings:
    transcribe: TranscribeOptions = field(default_factory=TranscribeOptions)
    output_dir: Path | None = None       # None means "next to the video"
    write_srt: bool = True
    write_vtt: bool = False
    write_txt: bool = False

    # Speaker diarisation
    diarize: bool = False
    expected_speakers: int = 0           # 0 = detect automatically
    diarization_threshold: float = 0.5

    # Summaries
    make_summary: bool = False           # the general briefing PDF
    make_chronological: bool = False     # the in-order timeline PDF
    llm_key: str = llm_models.DEFAULT_LLM
    llm_custom_path: str = ""
    llm_on_gpu: bool = True
    include_transcript_in_pdf: bool = False
    overwrite: bool = True

    @property
    def needs_llm(self) -> bool:
        return self.make_summary or self.make_chronological


class PipelineWorker(QObject):
    """Lives on its own QThread; the UI only ever touches it through signals."""

    log = Signal(str)
    file_started = Signal(int, str)          # index, name
    file_progress = Signal(int, float, str)  # index, 0..1, stage label
    file_finished = Signal(int, bool, str)   # index, ok, message
    overall_progress = Signal(float)
    outputs_ready = Signal(int, list)        # index, list of written paths
    finished = Signal()

    def __init__(self, files: list[Path], settings: JobSettings) -> None:
        super().__init__()
        self._files = files
        self._settings = settings
        self._cancel = False
        self._transcriber = Transcriber()
        self._summarizer = Summarizer()

    @Slot()
    def cancel(self) -> None:
        """Ask the run to stop. Checked between segments, chunks and tokens.

        A model that is part-way through loading cannot be interrupted, so a
        Stop pressed during "Loading model" takes effect once the load returns.
        Everything else aborts within a fraction of a second.
        """
        self._cancel = True
        self.log.emit("Stopping…")

    def _cancelled(self) -> bool:
        return self._cancel

    def _output_dir_for(self, media: Path) -> Path:
        target = self._settings.output_dir or media.parent
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _resolve_llm(self) -> tuple[Path | None, str, llm_models.LlmChoice | None]:
        if self._settings.llm_custom_path:
            path = Path(self._settings.llm_custom_path)
            return (path if path.is_file() else None), path.name, None
        choice = llm_models.find_choice(self._settings.llm_key)
        if choice is None:
            return None, "", None
        if not llm_models.is_downloaded(choice):
            self.log.emit(
                f"Fetching {choice.filename} (~{choice.approx_gb:.1f} GB). "
                "This happens once."
            )
            # A killable subprocess, not a plain call: the same stuck-CDN-
            # connection failure mode that affects Whisper downloads applies
            # here too. See model_cache.download_with_watchdog.
            download_with_watchdog(
                "llm",
                choice.key,
                llm_models_dir(),
                int(choice.approx_gb * 1024 ** 3),
                on_progress=lambda fraction, text: self.file_progress.emit(
                    0, 0.0, text
                ),
                should_cancel=self._cancelled,
                log=self.log.emit,
            )
        return llm_models.local_path(choice), choice.label, choice

    def _warn_if_ram_tight(self, llm_choice: llm_models.LlmChoice) -> None:
        """Flag a model that will not fit in VRAM + free RAM before it is run.

        A model larger than VRAM is supported on purpose - the overflow runs on
        the CPU - but if it does not fit in free system RAM either, Windows
        pages it to disk and generation slows to a crawl rather than failing
        outright. Better to say so up front than to let it look like a hang.
        """
        if llm_choice.fits_6gb_vram:
            return

        free_vram = free_vram_gb() or 0.0
        usable_vram = max(0.0, free_vram - 1.1)
        overflow_gb = max(0.0, llm_choice.approx_gb - usable_vram)
        if overflow_gb <= 0:
            return

        free_ram = free_ram_gb()
        total_ram = total_ram_gb()
        if free_ram is None or total_ram is None:
            return

        # Judge against the machine's capacity, not the momentary free figure:
        # whatever a browser happens to be holding right now can be closed, and
        # the question that matters is whether the model can ever fit. Windows
        # and background services need a few GB that never come back.
        capacity_ram = max(0.0, total_ram - OS_RAM_RESERVE_GB)

        self.log.emit(
            f"{llm_choice.key}: about {overflow_gb:.1f} GB will run on the CPU "
            f"({free_vram:.1f} GB VRAM free; {free_ram:.1f} of {total_ram:.1f} GB "
            f"RAM free)."
        )
        if overflow_gb > capacity_ram:
            self.log.emit(
                f"WARNING: that exceeds this machine's usable RAM "
                f"(~{capacity_ram:.1f} GB of {total_ram:.1f} GB). It will page to "
                "disk however many applications are closed. Choose a smaller "
                "summary model."
            )
        elif overflow_gb > free_ram:
            self.log.emit(
                f"Note: only {free_ram:.1f} GB RAM is free right now, so close "
                "other applications before starting or this will page to disk "
                "and crawl. It fits once they are closed."
            )

    def _preflight_disk_check(self) -> None:
        """Fail early and clearly rather than part-way through a 3 GB download."""
        needed = 0.0
        missing: list[str] = []

        choice = whisper_models.find_model(self._settings.transcribe.model)
        if choice is not None:
            if not whisper_models.is_downloaded(choice.key, whisper_models_dir()):
                needed += choice.download_gb
                missing.append(f"Whisper {choice.key} (~{choice.download_gb:.1f} GB)")

        if self._settings.diarize and not diarization.models_present():
            needed += diarization.TOTAL_DOWNLOAD_MB / 1024
            missing.append(
                f"speaker models (~{diarization.TOTAL_DOWNLOAD_MB} MB)"
            )

        if self._settings.needs_llm and not self._settings.llm_custom_path:
            llm_choice = llm_models.find_choice(self._settings.llm_key)
            if llm_choice is not None:
                if not llm_models.is_downloaded(llm_choice):
                    needed += llm_choice.approx_gb
                    missing.append(f"{llm_choice.key} (~{llm_choice.approx_gb:.1f} GB)")
                self._warn_if_ram_tight(llm_choice)

        if not missing:
            return

        root = models_root()
        free = free_space_gb(root)
        # Hugging Face stages the download before moving it into place, so leave
        # headroom rather than requiring exactly the model size.
        required = needed * 1.15
        self.log.emit(
            f"Need to download: {', '.join(missing)}. "
            f"{free:.1f} GB free in {root}."
        )
        if free < required:
            raise RuntimeError(
                f"Not enough disk space. Downloading {', '.join(missing)} needs about "
                f"{required:.1f} GB, but only {free:.1f} GB is free on the drive "
                f"holding {root}.\n\n"
                "Either free up space, choose a smaller model, or move the model "
                "folder to another drive with File > Change model folder."
            )

    @Slot()
    def run(self) -> None:
        total = len(self._files)
        try:
            try:
                # A previous force quit can leave lock files behind that would
                # otherwise make the next download hang at 0% forever.
                for root in (whisper_models_dir(), llm_models_dir()):
                    clear_stale_locks(root, log=self.log.emit)
                self._preflight_disk_check()
            except Exception as exc:
                self.log.emit(f"ERROR: {exc}")
                for index in range(total):
                    self.file_finished.emit(index, False, str(exc))
                return
            for index, media in enumerate(self._files):
                if self._cancel:
                    break
                self.file_started.emit(index, media.name)
                try:
                    written = self._process(index, media)
                    self.outputs_ready.emit(index, [str(p) for p in written])
                    self.file_finished.emit(index, True, "Done")
                except Cancelled:
                    self.log.emit("Stopped. Partial outputs were not written.")
                    self.file_finished.emit(index, False, "Stopped")
                    break
                except Exception as exc:  # surfaced in the UI, logged in full
                    self.log.emit(f"ERROR on {media.name}: {exc}")
                    self.log.emit(traceback.format_exc())
                    self.file_finished.emit(index, False, str(exc))
                self.overall_progress.emit((index + 1) / total)
        finally:
            self._transcriber.unload()
            self._summarizer.unload()
            self.finished.emit()

    def _stage_weights(self) -> dict[str, tuple[float, float]]:
        """Split the per-file progress bar between the stages that will run."""
        settings = self._settings
        costs = {"transcribe": 10.0}
        if settings.diarize:
            costs["diarize"] = 3.0
        if settings.make_summary:
            costs["summary"] = 4.0
        if settings.make_chronological:
            costs["chronological"] = 4.0

        total = sum(costs.values())
        spans: dict[str, tuple[float, float]] = {}
        cursor = 0.0
        for stage, cost in costs.items():
            share = cost / total
            spans[stage] = (cursor, share)
            cursor += share
        return spans

    def _process(self, index: int, media: Path) -> list[Path]:
        settings = self._settings
        out_dir = self._output_dir_for(media)
        stem = media.stem
        written: list[Path] = []
        spans = self._stage_weights()

        def report(stage: str, fraction: float, label: str) -> None:
            base, span = spans[stage]
            self.file_progress.emit(index, base + fraction * span, label)

        # ------------------------------------------------------ transcription
        # A previous file in this queue may have left the summary LLM resident
        # in VRAM (it is only unloaded here, at the start of the next file, or
        # in run()'s finally at the very end of the queue). Free it before
        # Whisper tries to load, or the two overlap and combined VRAM can
        # exceed a 6 GB card, silently forcing a CPU fallback.
        self._summarizer.unload()
        report("transcribe", 0.0, "Loading model")
        result = self._transcriber.transcribe(
            media,
            settings.transcribe,
            on_progress=lambda p: report("transcribe", p, "Transcribing"),
            should_cancel=self._cancelled,
            log=self.log.emit,
            on_download=lambda fraction, text: self.file_progress.emit(
                index, 0.0, text
            ),
        )
        segments = result.segments
        self.log.emit(f"{media.name}: {len(segments)} segments transcribed.")

        # -------------------------------------------------------- diarisation
        if settings.diarize:
            # Whisper is unloaded first: diarisation runs on the CPU, and this
            # keeps the GPU free for the summary model that follows.
            self._transcriber.unload()
            report("diarize", 0.0, "Identifying speakers")
            try:
                turns = diarization.diarize(
                    media,
                    expected_speakers=settings.expected_speakers,
                    threshold=settings.diarization_threshold,
                    on_progress=lambda p: report("diarize", p, "Identifying speakers"),
                    should_cancel=self._cancelled,
                    log=self.log.emit,
                )
                segments = diarization.apply_speakers(segments, turns)
            except Cancelled:
                raise
            except Exception as exc:
                self.log.emit(f"Speaker identification failed ({exc}). Continuing without it.")

        show_speakers = settings.diarize and any(s.speaker for s in segments)

        # ------------------------------------------------------ text outputs
        if settings.write_srt:
            written.append(
                self._write(
                    out_dir / f"{stem}.srt", subtitles.to_srt(segments, show_speakers)
                )
            )
        if settings.write_vtt:
            written.append(
                self._write(
                    out_dir / f"{stem}.vtt", subtitles.to_vtt(segments, show_speakers)
                )
            )
        if settings.write_txt:
            written.append(
                self._write(
                    out_dir / f"{stem}.txt", subtitles.to_text(segments, show_speakers)
                )
            )

        if not settings.needs_llm:
            self.file_progress.emit(index, 1.0, "Done")
            return written

        # ----------------------------------------------------------- summaries
        # Free the Whisper weights before the LLM claims VRAM — 6 GB is tight.
        self._transcriber.unload()

        model_path, model_label, choice = self._resolve_llm()
        if model_path is None:
            self.log.emit("No summary model available; skipping the PDF(s).")
            self.file_progress.emit(index, 1.0, "Done (no summary)")
            return written

        self._summarizer.load(
            model_path,
            choice,
            n_gpu_layers=-1 if settings.llm_on_gpu else 0,
            log=self.log.emit,
        )

        meta = ReportMeta(
            source_name=media.name,
            duration_seconds=result.duration,
            language=result.language,
            whisper_model=settings.transcribe.model,
            device=self._device_label(settings),
            summary_model=model_label,
        )

        if settings.make_summary:
            summary: SummaryResult = self._summarizer.summarize(
                segments,
                title=media.stem,
                show_speakers=show_speakers,
                on_progress=lambda p: report("summary", p, "Summarising"),
                should_cancel=self._cancelled,
                log=self.log.emit,
            )
            summary.model_label = model_label
            pdf_path = out_dir / f"{stem} - summary.pdf"
            build_pdf(
                pdf_path,
                summary,
                meta,
                segments=segments,
                include_transcript=settings.include_transcript_in_pdf,
            )
            written.append(pdf_path)
            self.log.emit(f"Summary written to {pdf_path.name}")

        if settings.make_chronological:
            timeline = self._summarizer.summarize_chronological(
                segments,
                title=media.stem,
                show_speakers=show_speakers,
                on_progress=lambda p: report("chronological", p, "Building the timeline"),
                should_cancel=self._cancelled,
                log=self.log.emit,
            )
            timeline.model_label = model_label
            pdf_path = out_dir / f"{stem} - chronological summary.pdf"
            build_chronological_pdf(
                pdf_path,
                timeline,
                meta,
                segments=segments,
                include_transcript=settings.include_transcript_in_pdf,
            )
            written.append(pdf_path)
            self.log.emit(f"Chronological summary written to {pdf_path.name}")

        self.file_progress.emit(index, 1.0, "Done")
        return written

    @staticmethod
    def _device_label(settings: JobSettings) -> str:
        from .transcriber import resolve_device

        return resolve_device(settings.transcribe.device)

    def _write(self, path: Path, content: str) -> Path:
        if path.exists() and not self._settings.overwrite:
            counter = 2
            while True:
                candidate = path.with_name(f"{path.stem} ({counter}){path.suffix}")
                if not candidate.exists():
                    path = candidate
                    break
                counter += 1
        path.write_text(content, encoding="utf-8")
        return path
