"""Speaker diarisation with sherpa-onnx.

Chosen over pyannote/WhisperX because it is pure ONNX: no PyTorch download, no
Hugging Face token, no gated model licences to accept, and the two models total
about 47 MB. It runs on the CPU, so it never competes with Whisper for the 6 GB
of VRAM.

Speaker count is detected automatically. If the user states how many speakers to
expect, any extra clusters beyond that are folded into a single "Misc" label.
"""

from __future__ import annotations

import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .errors import Cancelled
from .paths import models_root
from .subtitles import Segment, Word

SEGMENTATION_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
EMBEDDING_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/nemo_en_titanet_small.onnx"
)
SEGMENTATION_FILENAME = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
EMBEDDING_FILENAME = "nemo_en_titanet_small.onnx"

MISC_LABEL = "Misc"
TOTAL_DOWNLOAD_MB = 48


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str


def diarization_dir() -> Path:
    path = models_root() / "diarization"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_present() -> bool:
    root = diarization_dir()
    return (root / SEGMENTATION_FILENAME).is_file() and (
        root / EMBEDDING_FILENAME
    ).is_file()


def _download(url: str, target: Path, log: Callable[[str], None]) -> None:
    log(f"Downloading {target.name}…")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response:
        with partial.open("wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
    partial.replace(target)


def _safe_extract(archive: Path, destination: Path) -> None:
    """Extract, refusing any member that would escape the destination."""
    destination = destination.resolve()
    with tarfile.open(archive, "r:bz2") as tar:
        for member in tar.getmembers():
            resolved = (destination / member.name).resolve()
            if not str(resolved).startswith(str(destination)):
                raise RuntimeError(f"Refusing unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Refusing link member: {member.name}")
        tar.extractall(destination)


def ensure_models(log: Callable[[str], None] = lambda _m: None) -> tuple[Path, Path]:
    """Download the segmentation and embedding models once. Returns their paths."""
    root = diarization_dir()
    segmentation = root / SEGMENTATION_FILENAME
    embedding = root / EMBEDDING_FILENAME

    if not segmentation.is_file():
        archive = root / "segmentation.tar.bz2"
        _download(SEGMENTATION_URL, archive, log)
        log("Extracting the segmentation model…")
        _safe_extract(archive, root)
        archive.unlink(missing_ok=True)

    if not embedding.is_file():
        _download(EMBEDDING_URL, embedding, log)

    if not segmentation.is_file() or not embedding.is_file():
        raise RuntimeError("Diarisation models could not be prepared.")
    return segmentation, embedding


def _build_recogniser(threshold: float, num_threads: int = 4):
    import sherpa_onnx

    segmentation, embedding = ensure_models()
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(segmentation)
            ),
            num_threads=num_threads,
            provider="cpu",
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(embedding),
            num_threads=num_threads,
            provider="cpu",
        ),
        # num_clusters=-1 lets the threshold decide how many speakers there are,
        # which is what makes automatic detection possible.
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=threshold),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError("Invalid diarisation configuration.")
    return sherpa_onnx.OfflineSpeakerDiarization(config)


def diarize(
    media_path: Path,
    expected_speakers: int = 0,
    threshold: float = 0.5,
    on_progress: Callable[[float], None] = lambda _p: None,
    should_cancel: Callable[[], bool] = lambda: False,
    log: Callable[[str], None] = lambda _m: None,
) -> list[SpeakerTurn]:
    """Return who spoke when. expected_speakers=0 means detect automatically."""
    from faster_whisper.audio import decode_audio

    log("Analysing voices…")
    recogniser = _build_recogniser(threshold)
    audio = decode_audio(str(media_path), sampling_rate=recogniser.sample_rate)

    def callback(processed: int, total: int) -> int:
        if total:
            on_progress(min(1.0, processed / total))
        # sherpa-onnx aborts when the callback returns a non-zero value.
        return 1 if should_cancel() else 0

    result = recogniser.process(audio, callback=callback)
    if should_cancel():
        raise Cancelled()

    raw = [
        SpeakerTurn(start=s.start, end=s.end, speaker=str(s.speaker))
        for s in result.sort_by_start_time()
    ]
    if not raw:
        log("No distinct speakers were found.")
        return []

    turns = _label_speakers(raw, expected_speakers)
    names = sorted({turn.speaker for turn in turns})
    log(f"Found {len(names)} speaker(s): {', '.join(names)}")
    on_progress(1.0)
    return turns


def _label_speakers(turns: list[SpeakerTurn], expected: int) -> list[SpeakerTurn]:
    """Rename raw cluster ids to Speaker 1..n in order of first appearance.

    When the caller said how many speakers to expect and more were detected, the
    extra clusters - usually brief interjections or noise - are merged into one
    "Misc" label rather than cluttering the transcript.
    """
    speaking_time: dict[str, float] = {}
    first_seen: dict[str, float] = {}
    for turn in turns:
        speaking_time[turn.speaker] = speaking_time.get(turn.speaker, 0.0) + (
            turn.end - turn.start
        )
        first_seen.setdefault(turn.speaker, turn.start)

    keep = set(speaking_time)
    if expected > 0 and len(speaking_time) > expected:
        ranked = sorted(speaking_time, key=lambda k: speaking_time[k], reverse=True)
        keep = set(ranked[:expected])

    ordered = sorted(keep, key=lambda k: first_seen[k])
    names = {cluster: f"Speaker {i + 1}" for i, cluster in enumerate(ordered)}

    return [
        SpeakerTurn(
            start=turn.start,
            end=turn.end,
            speaker=names.get(turn.speaker, MISC_LABEL),
        )
        for turn in turns
    ]


def _speaker_at(turns: list[SpeakerTurn], start: float, end: float) -> str:
    """The speaker overlapping [start, end] the most, or '' if none does."""
    best_speaker = ""
    best_overlap = 0.0
    for turn in turns:
        overlap = min(end, turn.end) - max(start, turn.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn.speaker
    return best_speaker


def apply_speakers(
    segments: list[Segment], turns: list[SpeakerTurn]
) -> list[Segment]:
    """Attach speaker labels, splitting segments where the speaker changes."""
    if not turns:
        return segments

    output: list[Segment] = []
    for segment in segments:
        if not segment.words:
            segment.speaker = _speaker_at(turns, segment.start, segment.end)
            output.append(segment)
            continue

        run: list[Word] = []
        run_speaker = ""
        for word in segment.words:
            speaker = _speaker_at(turns, word.start, word.end) or run_speaker
            if run and speaker != run_speaker:
                output.append(_segment_from_words(run, run_speaker))
                run = []
            run_speaker = speaker
            run.append(word)
        if run:
            output.append(_segment_from_words(run, run_speaker))

    return [s for s in output if s.text.strip()]


def _segment_from_words(words: list[Word], speaker: str) -> Segment:
    return Segment(
        start=words[0].start,
        end=words[-1].end,
        text="".join(w.text for w in words).strip(),
        words=list(words),
        speaker=speaker,
    )
