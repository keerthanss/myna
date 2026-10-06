"""Turn Whisper segments into well-formed SRT / VTT / plain text."""

from __future__ import annotations

from dataclasses import dataclass, field

MAX_CHARS_PER_LINE = 42
MAX_LINES = 2
MAX_CUE_CHARS = MAX_CHARS_PER_LINE * MAX_LINES
MAX_CUE_SECONDS = 7.0
MIN_CUE_SECONDS = 0.6


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    speaker: str = ""  # filled in by diarization when it is enabled


def format_timestamp(seconds: float, separator: str = ",") -> str:
    seconds = max(0.0, seconds)
    milliseconds = int(round(seconds * 1000.0))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{milliseconds:03d}"


def _wrap(text: str) -> str:
    """Balance a cue across at most MAX_LINES lines."""
    words = text.split()
    if not words:
        return text
    if len(text) <= MAX_CHARS_PER_LINE:
        return text

    target = len(text) / MAX_LINES
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > target and len(lines) < MAX_LINES - 1:
            lines.append(current)
            current = word
        else:
            current = candidate
    lines.append(current)
    return "\n".join(line for line in lines if line)


def _split_by_words(segment: Segment) -> list[Segment]:
    """Break an over-long segment on word boundaries, keeping real timings.

    Cues are sized evenly rather than greedily. Filling each cue to the maximum
    and letting the remainder fall into the last one produces orphans - a
    half-second cue reading "rates affect" - which are unpleasant to read.
    Working out how many cues are needed first and aiming for that average
    keeps them balanced.
    """
    words = segment.words
    if not words:
        return [segment]

    total_chars = len("".join(w.text for w in words).strip())
    total_duration = words[-1].end - words[0].start
    pieces = max(
        1,
        -(-total_chars // MAX_CUE_CHARS),              # ceil division
        -(-int(total_duration) // int(MAX_CUE_SECONDS)) if MAX_CUE_SECONDS else 1,
    )
    if pieces == 1:
        return [segment]
    target_chars = total_chars / pieces

    cues: list[Segment] = []
    bucket: list[Word] = []

    def bucket_text() -> str:
        return "".join(w.text for w in bucket).strip()

    def flush() -> None:
        text = bucket_text()
        if text:
            cues.append(
                Segment(
                    start=bucket[0].start,
                    end=bucket[-1].end,
                    text=text,
                    speaker=segment.speaker,
                )
            )
        bucket.clear()

    for word in words:
        bucket.append(word)
        length = len(bucket_text())
        duration = word.end - bucket[0].start
        at_sentence_end = word.text.strip().endswith((".", "!", "?", "…"))

        # Cut at a sentence boundary once the cue is already respectably long,
        # otherwise at the average size, and always before the hard limits.
        if (
            length >= MAX_CUE_CHARS
            or duration >= MAX_CUE_SECONDS
            or (at_sentence_end and length >= target_chars * 0.6)
            or length >= target_chars
        ):
            flush()
    flush()

    return _absorb_orphans(cues) or [segment]


MAX_MERGE_GAP_SECONDS = 1.0


def _absorb_orphans(cues: list[Segment]) -> list[Segment]:
    """Fold a stray very short cue into the previous one when it still fits.

    Whisper often emits a trailing fragment - "convergence." - as a segment of
    its own, which would otherwise flash on screen for half a second. Merging
    is refused across a speaker change or a real pause, so dialogue and
    separate thoughts stay apart.
    """
    if len(cues) < 2:
        return cues

    merged: list[Segment] = [cues[0]]
    for cue in cues[1:]:
        previous = merged[-1]
        combined = f"{previous.text} {cue.text}"
        short = len(cue.text) < MAX_CHARS_PER_LINE // 2
        # Only continuations are absorbed. If the previous cue ended a sentence,
        # this short cue is a new thought and deserves its own cue - otherwise
        # two consecutive short sentences would be glued together.
        continues_sentence = not previous.text.rstrip().endswith(
            (".", "!", "?", "…", ":", ";", "—")
        )
        fits = (
            len(combined) <= MAX_CUE_CHARS
            and (cue.end - previous.start) <= MAX_CUE_SECONDS
            and cue.speaker == previous.speaker
            and (cue.start - previous.end) <= MAX_MERGE_GAP_SECONDS
        )
        if short and continues_sentence and fits:
            previous.text = combined
            previous.end = cue.end
            continue
        merged.append(cue)
    return merged


def build_cues(segments: list[Segment]) -> list[Segment]:
    cues: list[Segment] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if len(text) <= MAX_CUE_CHARS and (segment.end - segment.start) <= MAX_CUE_SECONDS:
            cues.append(
                Segment(segment.start, segment.end, text, speaker=segment.speaker)
            )
        else:
            cues.extend(_split_by_words(segment))

    # Run across the whole file, not just within one segment: Whisper regularly
    # emits a trailing word as a segment of its own.
    cues = _absorb_orphans(cues)

    # Enforce a readable minimum duration and remove overlaps.
    for index, cue in enumerate(cues):
        if cue.end - cue.start < MIN_CUE_SECONDS:
            cue.end = cue.start + MIN_CUE_SECONDS
        if index + 1 < len(cues) and cue.end > cues[index + 1].start:
            cue.end = max(cue.start + 0.1, cues[index + 1].start - 0.001)
    return cues


def _cue_body(cue: Segment, show_speakers: bool) -> str:
    """Speaker labels go on their own line so they never eat into the wrap width."""
    body = _wrap(cue.text)
    if show_speakers and cue.speaker:
        return f"{cue.speaker}:\n{body}"
    return body


def to_srt(segments: list[Segment], show_speakers: bool = False) -> str:
    lines: list[str] = []
    for index, cue in enumerate(build_cues(segments), start=1):
        lines.append(str(index))
        lines.append(
            f"{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}"
        )
        lines.append(_cue_body(cue, show_speakers))
        lines.append("")
    return "\n".join(lines)


def to_vtt(segments: list[Segment], show_speakers: bool = False) -> str:
    lines = ["WEBVTT", ""]
    for cue in build_cues(segments):
        lines.append(
            f"{format_timestamp(cue.start, '.')} --> {format_timestamp(cue.end, '.')}"
        )
        lines.append(_cue_body(cue, show_speakers))
        lines.append("")
    return "\n".join(lines)


def to_text(segments: list[Segment], show_speakers: bool = False) -> str:
    """Plain transcript. With speakers, consecutive lines from one person merge."""
    if not show_speakers:
        return "\n".join(s.text.strip() for s in segments if s.text.strip())

    blocks: list[str] = []
    current_speaker: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        body = " ".join(buffer)
        blocks.append(f"{current_speaker}: {body}" if current_speaker else body)
        buffer.clear()

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if segment.speaker != current_speaker:
            flush()
            current_speaker = segment.speaker
        buffer.append(text)
    flush()
    return "\n\n".join(blocks)


def to_timestamped_text(segments: list[Segment], show_speakers: bool = False) -> str:
    lines = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        stamp = format_timestamp(segment.start, ".")[:-4]
        who = f"{segment.speaker}: " if show_speakers and segment.speaker else ""
        lines.append(f"[{stamp}] {who}{text}")
    return "\n".join(lines)
