"""Map-reduce summarisation of a transcript with a local GGUF model."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .cuda_setup import ensure_cuda_dlls
from .errors import Cancelled
from .gpu_budget import estimated_ram_need_gb, plan_offload
from .llm_models import LlmChoice
from .subtitles import Segment, format_timestamp

# Transcript chunking is derived from the loaded model's context window rather
# than fixed. A fixed 6000 characters is ~1.7k tokens, which used barely a
# fifth of an 8k context and shredded a long recording into many small chunks -
# each summarised in isolation and then merged, losing detail at every step.
# Fewer, larger chunks mean fewer lossy reductions.
#
# Half the context is given to transcript text; the rest covers the prompt, the
# model's own output, and slack for the tokeniser being less efficient than the
# estimate on names, numbers and timestamps.
CONTEXT_FRACTION_FOR_TEXT = 0.5
CHARS_PER_TOKEN = 3.6
CHUNK_CHARS = 6000          # fallback when the context is unknown
CHUNK_OVERLAP_CHARS = 400


def chunk_chars_for_context(context: int) -> int:
    """How many characters of transcript to put in one chunk."""
    if context <= 0:
        return CHUNK_CHARS
    tokens = context * CONTEXT_FRACTION_FOR_TEXT
    return max(CHUNK_CHARS, int(tokens * CHARS_PER_TOKEN))


@dataclass
class SummaryResult:
    overview: str = ""
    key_points: list[str] = field(default_factory=list)
    chapters: list[tuple[str, str]] = field(default_factory=list)  # (timestamp, title)
    action_items: list[str] = field(default_factory=list)
    model_label: str = ""


@dataclass
class TopicBlock:
    """One stretch of the recording that stays on a single subject."""

    start: float
    end: float
    title: str
    summary: str
    speakers: list[str] = field(default_factory=list)

    @property
    def time_range(self) -> str:
        return f"{_clock(self.start)} - {_clock(self.end)}"


@dataclass
class ChronologicalSummary:
    blocks: list[TopicBlock] = field(default_factory=list)
    model_label: str = ""


def _clock(seconds: float) -> str:
    """MM:SS under an hour, HH:MM:SS beyond it - matches how people read a scrubber."""
    stamp = format_timestamp(seconds, ".")[:8]
    return stamp[3:] if stamp.startswith("00:") else stamp


def _parse_clock(text: str) -> float | None:
    parts = text.strip().split(":")
    if not all(p.strip().isdigit() for p in parts) or not 2 <= len(parts) <= 3:
        return None
    numbers = [int(p) for p in parts]
    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]


def _timestamped_transcript(segments: list[Segment], show_speakers: bool = False) -> str:
    lines = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            stamp = format_timestamp(segment.start, ".")[:8]
            who = f" {segment.speaker}:" if show_speakers and segment.speaker else ""
            lines.append(f"[{stamp}]{who} {text}")
    return "\n".join(lines)


def _chunk(text: str, chunk_chars: int = CHUNK_CHARS) -> list[str]:
    lines = text.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in lines:
        if size + len(line) > chunk_chars and current:
            chunks.append("\n".join(current))
            # Keep a little overlap so context is not cut mid-thought.
            tail: list[str] = []
            tail_size = 0
            for previous in reversed(current):
                if tail_size + len(previous) > CHUNK_OVERLAP_CHARS:
                    break
                tail.insert(0, previous)
                tail_size += len(previous)
            current = tail
            size = tail_size
        current.append(line)
        size += len(line)
    if current:
        chunks.append("\n".join(current))
    return chunks


def _bullets(text: str) -> list[str]:
    items: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        line = re.sub(r"^[-*•]\s*", "", line)
        line = re.sub(r"^\d+[.)]\s*", "", line)
        line = line.strip("*").strip()
        if len(line) > 3 and not line.endswith(":"):
            items.append(line)
    return items


HEADINGS = ("OVERVIEW", "KEY POINTS", "CHAPTERS", "ACTION ITEMS")


def _is_heading(line: str, heading: str) -> bool:
    """Match a heading however the model dressed it up: ##, **, trailing colon."""
    stripped = line.strip().strip("#*_ ").rstrip(":").strip()
    return stripped.upper() == heading


def _section(text: str, heading: str) -> str:
    """Pull the body under one heading out of the model's reply."""
    collected: list[str] = []
    capturing = False
    for line in text.split("\n"):
        if _is_heading(line, heading):
            capturing = True
            continue
        if capturing and any(_is_heading(line, other) for other in HEADINGS):
            break
        if capturing:
            collected.append(line)
    return "\n".join(collected).strip()


def _chunk_bounds(chunk: str) -> tuple[float, float]:
    """First and last timestamp actually present in a chunk."""
    stamps = [
        value
        for value in (
            _parse_clock(m) for m in re.findall(r"^\[(\d{2}:\d{2}:\d{2})\]", chunk, re.M)
        )
        if value is not None
    ]
    return (min(stamps), max(stamps)) if stamps else (0.0, 0.0)


def _parse_topic_blocks(reply: str, lower: float, upper: float) -> list[TopicBlock]:
    """Read TOPIC/START/END/SUMMARY blocks, discarding anything malformed."""
    blocks: list[TopicBlock] = []
    current: dict[str, str] = {}

    def commit() -> None:
        title = current.get("topic", "").strip()
        summary = current.get("summary", "").strip()
        start = _parse_clock(current.get("start", ""))
        end = _parse_clock(current.get("end", ""))
        current.clear()
        if not title or start is None:
            return
        if end is None or end < start:
            end = start
        # The model occasionally invents a timestamp; keep it inside this chunk.
        start = min(max(start, lower), upper)
        end = min(max(end, start), upper)
        blocks.append(
            TopicBlock(start=start, end=end, title=title, summary=summary)
        )

    for raw in reply.split("\n"):
        line = raw.strip().lstrip("#*-• ").strip()
        match = re.match(r"^(TOPIC|START|END|SUMMARY)\s*:\s*(.*)$", line, re.IGNORECASE)
        if match:
            key = match.group(1).lower()
            if key == "topic" and current:
                commit()
            current[key] = match.group(2).strip().strip("*_ ")
        elif current.get("summary") is not None and line and "summary" in current:
            # Continuation of a wrapped SUMMARY line.
            current["summary"] = f"{current['summary']} {line}".strip()
    if current:
        commit()
    return blocks


def _tidy_blocks(blocks: list[TopicBlock], media_end: float) -> list[TopicBlock]:
    """Sort, merge repeats across chunk seams, and close gaps in the timeline."""
    if not blocks:
        return []

    blocks.sort(key=lambda b: b.start)

    merged: list[TopicBlock] = []
    for block in blocks:
        if not block.title:
            continue
        previous = merged[-1] if merged else None
        same_topic = (
            previous is not None
            and previous.title.strip().lower() == block.title.strip().lower()
        )
        if same_topic:
            previous.end = max(previous.end, block.end)
            if block.summary and block.summary not in previous.summary:
                previous.summary = f"{previous.summary} {block.summary}".strip()
            continue
        merged.append(block)

    # Each section should run up to the next one so the timeline has no holes.
    for index, block in enumerate(merged):
        if index + 1 < len(merged):
            block.end = max(block.start, merged[index + 1].start)
        else:
            block.end = max(block.end, media_end)
    return merged


def _attach_speakers(blocks: list[TopicBlock], segments: list[Segment]) -> None:
    """Record who spoke during each section, most talkative first."""
    for block in blocks:
        talk_time: dict[str, float] = {}
        for segment in segments:
            overlap = min(block.end, segment.end) - max(block.start, segment.start)
            if overlap > 0 and segment.speaker:
                talk_time[segment.speaker] = talk_time.get(segment.speaker, 0.0) + overlap
        block.speakers = sorted(talk_time, key=lambda k: talk_time[k], reverse=True)


class Summarizer:
    """Loads a GGUF once and reuses it across files."""

    def __init__(self) -> None:
        self._llm = None
        self._path: Path | None = None
        self._context = 0

    def load(
        self,
        model_path: Path,
        choice: LlmChoice | None = None,
        n_gpu_layers: int = -1,
        log: Callable[[str], None] = lambda _m: None,
    ) -> None:
        if self._llm is not None and self._path == model_path:
            return

        # The CUDA build of llama.cpp links cudart, cuBLAS and friends, which
        # ship as pip packages rather than on PATH. Register them before the
        # import or loading llama.dll fails with a bare "module not found".
        ensure_cuda_dlls()
        from llama_cpp import Llama

        self.unload()
        context = choice.context if choice else 8192
        log(f"Loading summary model {model_path.name}...")

        # Work out the GPU/CPU split before loading rather than discovering it
        # by failing: a failed attempt on a 9 GB model means reading 9 GB off
        # disk first. n_gpu_layers == 0 is an explicit "CPU only" from the
        # caller and is respected as-is.
        if n_gpu_layers != 0:
            planned, reason = plan_offload(
                model_path, declared_layers=choice.layers if choice else 0
            )
            log(f"GPU offload: {reason}.")
            if planned != -1:
                ram_gb = estimated_ram_need_gb(
                    model_path, planned, choice.layers if choice else 0
                )
                if ram_gb > 0.5:
                    log(f"About {ram_gb:.1f} GB of this model will sit in system RAM.")
            n_gpu_layers = planned

        # Safety net: if the planned split still will not load, step down.
        # The planned value is tried first, so this rarely costs an attempt.
        if n_gpu_layers == 0:
            attempts = [0]
        elif n_gpu_layers == -1:
            attempts = [-1, 24, 12, 0]
        else:
            attempts = [n_gpu_layers, max(0, n_gpu_layers // 2), 0]
        last_error: Exception | None = None

        for index, layers in enumerate(attempts):
            try:
                self._llm = Llama(
                    model_path=str(model_path),
                    n_ctx=context,
                    n_gpu_layers=layers,
                    verbose=False,
                )
                if index:
                    where = f"{layers} layers on the GPU" if layers else "the CPU"
                    log(f"Not enough VRAM for a full offload; running with {where}.")
                self._path = model_path
                self._context = context
                return
            except OSError as exc:
                raise self._translate_os_error(exc) from exc  # not retryable
            except Exception as exc:  # noqa: BLE001 - retried with fewer layers
                last_error = exc
                self.unload()
                if index + 1 < len(attempts):
                    log(f"Load with n_gpu_layers={layers} failed ({exc}). Retrying.")

        raise RuntimeError(
            f"Could not load the summary model: {last_error}"
        ) from last_error

    @staticmethod
    def _translate_os_error(exc: OSError) -> Exception:
        """Turn an opaque Windows fault into something a user can act on."""
        text = str(exc).lower()
        # 0xc000001d is STATUS_ILLEGAL_INSTRUCTION: the llama.cpp build uses SIMD
        # instructions this CPU lacks (usually AVX-512 on an AMD Zen or an older
        # Intel chip). Retrying on the CPU cannot help - it is the same binary.
        if "c000001d" in text or "-1073741795" in text:
            return RuntimeError(
                "The installed llama-cpp-python build uses CPU instructions this "
                "processor does not support (illegal instruction). Re-run "
                "Install.bat, which pins a build compiled for AVX2. Transcription "
                "and speaker labels are unaffected."
            )
        if "llama.dll" in text or "shared library" in text:
            return RuntimeError(
                "The summarisation engine could not load its CUDA libraries. "
                "Re-run Install.bat to repair the installation. Transcription "
                "and speaker labels are unaffected."
            )
        return exc

    def unload(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
            self._path = None
            import gc

            gc.collect()

    def _chat(
        self,
        system: str,
        user: str,
        max_tokens: int = 900,
        should_cancel: Callable[[], bool] = lambda: False,
    ) -> str:
        assert self._llm is not None

        # Streamed rather than awaited in one go purely so Stop can take effect
        # between tokens. create_chat_completion has no stopping_criteria
        # parameter, and a non-streamed call cannot be interrupted at all - a
        # long summary would keep the UI waiting for a minute after Stop.
        stream = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
            stream=True,
        )

        pieces: list[str] = []
        try:
            for chunk in stream:
                if should_cancel():
                    raise Cancelled()
                delta = chunk["choices"][0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    pieces.append(piece)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

        return "".join(pieces).strip()

    def summarize(
        self,
        segments: list[Segment],
        title: str,
        show_speakers: bool = False,
        on_progress: Callable[[float], None] = lambda _p: None,
        should_cancel: Callable[[], bool] = lambda: False,
        log: Callable[[str], None] = lambda _m: None,
    ) -> SummaryResult:
        transcript = _timestamped_transcript(segments, show_speakers=show_speakers)
        chunks = _chunk(transcript, chunk_chars_for_context(self._context))
        log(f"Summarising {len(chunks)} transcript chunk(s)...")

        notes: list[str] = []
        for index, chunk in enumerate(chunks):
            if should_cancel():
                raise Cancelled()
            note = self._chat(
                system=(
                    "You summarise transcript excerpts precisely. Never invent facts "
                    "that are not in the excerpt."
                ),
                user=(
                    f"Below is part {index + 1} of {len(chunks)} of a transcript of "
                    f"{title!r}. Each line starts with its timestamp.\n\n"
                    "Write concise notes covering: what is discussed, the specific "
                    "claims or decisions made, and any tasks or follow-ups mentioned. "
                    "Keep the leading timestamp for each note.\n\n"
                    f"TRANSCRIPT:\n{chunk}"
                ),
                max_tokens=700,
                should_cancel=should_cancel,
            )
            notes.append(note)
            on_progress((index + 1) / (len(chunks) + 1))

        combined = "\n\n".join(notes)
        if should_cancel():
            raise Cancelled()

        log("Composing the final summary...")
        final = self._chat(
            system=(
                "You write clear, factual briefing documents from transcript notes. "
                "Use only what the notes contain."
            ),
            user=(
                f"Here are ordered notes from the full transcript of {title!r}.\n\n"
                f"{combined}\n\n"
                "Produce a briefing using exactly these headings and nothing else:\n\n"
                "OVERVIEW\n"
                "Two to four paragraphs describing what this recording covers.\n\n"
                "KEY POINTS\n"
                "Six to ten bullet points, each a single specific statement.\n\n"
                "CHAPTERS\n"
                "Bullets of the form HH:MM:SS - Section title, in time order.\n\n"
                "ACTION ITEMS\n"
                "Bullets for concrete tasks or follow-ups. Write 'None identified.' "
                "if the recording contains none."
            ),
            max_tokens=1400,
            should_cancel=should_cancel,
        )
        on_progress(1.0)

        chapters: list[tuple[str, str]] = []
        for line in _bullets(_section(final, "CHAPTERS")):
            match = re.match(r"^\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?\s*[-–—:]?\s*(.+)$", line)
            if match:
                chapters.append((match.group(1), match.group(2).strip()))

        actions = _bullets(_section(final, "ACTION ITEMS"))
        if len(actions) == 1 and actions[0].lower().startswith("none"):
            actions = []

        overview = _section(final, "OVERVIEW") or final
        return SummaryResult(
            overview=overview,
            key_points=_bullets(_section(final, "KEY POINTS")),
            chapters=chapters,
            action_items=actions,
        )

    # ------------------------------------------------------------ chronological

    def summarize_chronological(
        self,
        segments: list[Segment],
        title: str,
        show_speakers: bool = False,
        on_progress: Callable[[float], None] = lambda _p: None,
        should_cancel: Callable[[], bool] = lambda: False,
        log: Callable[[str], None] = lambda _m: None,
    ) -> ChronologicalSummary:
        """Walk the recording in order, describing each topic and when it ran.

        Unlike the general summary this never reorders or merges across the whole
        transcript: the output follows the video from start to finish, so a
        reader can scrub straight to the part they care about.
        """
        transcript = _timestamped_transcript(segments, show_speakers=show_speakers)
        chunks = _chunk(transcript, chunk_chars_for_context(self._context))
        media_end = max((s.end for s in segments), default=0.0)
        log(f"Building the chronological summary from {len(chunks)} chunk(s)...")

        blocks: list[TopicBlock] = []
        for index, chunk in enumerate(chunks):
            if should_cancel():
                raise Cancelled()

            chunk_start, chunk_end = _chunk_bounds(chunk)
            reply = self._chat(
                system=(
                    "You divide a transcript into consecutive topic sections. You "
                    "never reorder the material and never invent content."
                ),
                user=(
                    f"This is part {index + 1} of {len(chunks)} of the transcript of "
                    f"{title!r}. Every line begins with its timestamp.\n\n"
                    "Split it into the topics discussed, in the order they occur. "
                    "For each topic output exactly this block and nothing else:\n\n"
                    "TOPIC: <a short title, at most eight words>\n"
                    "START: <timestamp of the first line of this topic>\n"
                    "END: <timestamp of the last line of this topic>\n"
                    "SUMMARY: <two to four sentences on what was said>\n\n"
                    "Use timestamps copied from the transcript lines. Prefer three "
                    "to six topics for this part; merge trivial asides into the "
                    "topic around them.\n\n"
                    f"TRANSCRIPT:\n{chunk}"
                ),
                max_tokens=1200,
                should_cancel=should_cancel,
            )
            blocks.extend(_parse_topic_blocks(reply, chunk_start, chunk_end))
            on_progress((index + 1) / len(chunks))

        blocks = _tidy_blocks(blocks, media_end)
        if show_speakers:
            _attach_speakers(blocks, segments)
        log(f"Chronological summary has {len(blocks)} section(s).")
        return ChronologicalSummary(blocks=blocks)
