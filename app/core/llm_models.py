"""GGUF summarisation models.

Models that exceed 6 GB of VRAM are deliberately offered. llama.cpp keeps the
layers that do not fit in system RAM and runs them on the CPU - slower per
token, but a 14B model writes markedly better summaries than a 3B that happens
to fit entirely on the card. See gpu_budget.plan_offload for how the split is
chosen.

Sizes below are the real Content-Length of each file, measured against Hugging
Face rather than estimated, because the pre-flight disk check depends on them.
`layers` is the GGUF block_count, used to proportion the GPU/CPU split without
having to load the model first; 0 means "read it from the file".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .paths import llm_models_dir


@dataclass(frozen=True)
class LlmChoice:
    key: str
    label: str
    repo_id: str
    filename: str
    approx_gb: float
    context: int
    layers: int = 0          # GGUF block_count; 0 = read from the file
    fits_6gb_vram: bool = True


LLM_MODELS: list[LlmChoice] = [
    # --- fits entirely on a 6 GB card -------------------------------------
    LlmChoice(
        key="qwen2.5-3b",
        label="Qwen2.5 3B Instruct (Q4_K_M) — fastest, ~2 GB, fits on the GPU",
        repo_id="bartowski/Qwen2.5-3B-Instruct-GGUF",
        filename="Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        approx_gb=2.0,
        context=8192,
        layers=36,
    ),
    LlmChoice(
        key="llama-3.2-3b",
        label="Llama 3.2 3B Instruct (Q4_K_M) — fastest, ~2 GB, fits on the GPU",
        repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        approx_gb=2.0,
        context=8192,
        layers=28,
    ),
    LlmChoice(
        key="mistral-7b",
        label="Mistral 7B Instruct v0.3 (Q4_K_M) — ~4.4 GB, fits on the GPU",
        repo_id="bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        filename="Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        approx_gb=4.4,
        context=8192,
        layers=32,
    ),
    LlmChoice(
        key="qwen2.5-7b",
        label="Qwen2.5 7B Instruct (Q4_K_M) — ~4.7 GB, fits on the GPU",
        repo_id="bartowski/Qwen2.5-7B-Instruct-GGUF",
        filename="Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        approx_gb=4.7,
        context=8192,
        layers=28,
    ),

    # --- larger than 6 GB: splits across GPU and CPU RAM, slower ----------
    LlmChoice(
        key="qwen2.5-14b-q3",
        label="Qwen2.5 14B Instruct (Q3_K_M) — 7.3 GB, part on CPU, much better summaries",
        repo_id="bartowski/Qwen2.5-14B-Instruct-GGUF",
        filename="Qwen2.5-14B-Instruct-Q3_K_M.gguf",
        approx_gb=7.34,
        context=16384,
        layers=48,
        fits_6gb_vram=False,
    ),
    LlmChoice(
        key="qwen2.5-14b",
        label="Qwen2.5 14B Instruct (Q4_K_M) — 9.0 GB, part on CPU, best quality here",
        repo_id="bartowski/Qwen2.5-14B-Instruct-GGUF",
        filename="Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        approx_gb=8.99,
        context=16384,
        layers=48,
        fits_6gb_vram=False,
    ),
    LlmChoice(
        key="qwen3-14b",
        label="Qwen3 14B (Q4_K_M) — 9.0 GB, part on CPU, newer than Qwen2.5",
        repo_id="unsloth/Qwen3-14B-GGUF",
        filename="Qwen3-14B-Q4_K_M.gguf",
        approx_gb=9.00,
        context=16384,
        layers=40,
        fits_6gb_vram=False,
    ),
    LlmChoice(
        key="mistral-small-24b-q3",
        label="Mistral Small 24B (Q3_K_M) — 11.5 GB, mostly on CPU, slow",
        repo_id="bartowski/Mistral-Small-24B-Instruct-2501-GGUF",
        filename="Mistral-Small-24B-Instruct-2501-Q3_K_M.gguf",
        approx_gb=11.47,
        context=16384,
        layers=40,
        fits_6gb_vram=False,
    ),
]

DEFAULT_LLM = "qwen2.5-7b"


def find_choice(key: str) -> LlmChoice | None:
    return next((m for m in LLM_MODELS if m.key == key), None)


def local_path(choice: LlmChoice) -> Path:
    return llm_models_dir() / choice.filename


def is_downloaded(choice: LlmChoice) -> bool:
    return local_path(choice).is_file()


def download(
    choice: LlmChoice,
    on_progress: Callable[[float, str], None] = lambda _f, _m: None,
) -> Path:
    """Fetch a GGUF from Hugging Face into the app's model folder."""
    target = local_path(choice)
    if target.is_file():
        return target

    from huggingface_hub import hf_hub_download

    from .model_cache import DownloadProgress

    root = llm_models_dir()
    expected = int(choice.approx_gb * 1024 ** 3)
    with DownloadProgress(root, expected, on_progress, label=f"Downloading {choice.key}"):
        downloaded = hf_hub_download(
            repo_id=choice.repo_id,
            filename=choice.filename,
            local_dir=str(root),
        )
    on_progress(1.0, "Download complete.")
    return Path(downloaded)
