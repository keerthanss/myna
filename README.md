# Myna

*Named after the hill myna, a bird celebrated for reproducing almost any sound it hears.*

A small Windows desktop app that transcribes video **and audio** to subtitles
using OpenAI's Whisper models, entirely on your own machine. It can also label
who is speaking, and write two kinds of summary PDF using a local open-source
language model.

Nothing is uploaded anywhere. No API keys, no accounts, no internet needed after
the models have downloaded once. That makes it a workable local notetaker for
recorded lectures and meetings.

## Install

Double-click **`Install.bat`** and leave the window open. It:

1. finds your Python (3.9–3.12),
2. builds a private environment in `D:\VibeCodedApps\Myna`,
3. installs everything it needs (about 2 GB the first time),
4. creates **Desktop** and **Start Menu** shortcuts named *Myna*,
5. registers an entry in **Add/Remove Programs**.

To install somewhere else, run it with a path:

```
Install.bat -InstallDir E:\Apps\Myna
```

No administrator rights are required, and nothing is installed system-wide.

### Everything stays in one folder

The app, its Python environment, the models, the Hugging Face download cache,
your preferences and the log all live inside the install folder:

```
D:\VibeCodedApps\Myna\
├─ app\                        the program
├─ tools\                      icon generator and self-test
├─ venv\                       its private Python environment
├─ Myna.cmd                    launcher
└─ data\
   ├─ models\                  Whisper, diarisation and summary models
   │  └─ hf-cache\             Hugging Face download staging
   ├─ settings.ini             your preferences
   ├─ config.json              model folder location
   └─ whisper-studio.log       crash logs
```

Nothing is written to `C:` — no `AppData`, no `%USERPROFILE%\.cache`, and
preferences go to an INI file rather than the registry. The one exception is the
**Add/Remove Programs** entry, which Windows requires to be a registry value; it
is a few hundred bytes and the uninstaller removes it.

This matters more than it sounds: Hugging Face normally stages downloads in
`%USERPROFILE%\.cache\huggingface`, and `hf-xet` writes the *entire* model
there before moving it into place. On a 3 GB model that silently costs 3 GB of
system-drive space. The app redirects that cache into its own folder.

If you don't have Python, get 3.12 from [python.org](https://www.python.org/downloads/windows/)
and tick *"Add python.exe to PATH"* during its setup.

To remove the app later, use *Add/Remove Programs*, or run `Uninstall.bat`.
Because everything lives in the one folder, deleting that folder also works —
the uninstaller only additionally clears the shortcuts and the registry entry.

## Using it

1. Launch **Myna** from the Desktop or Start Menu.
2. Drag files (or a whole folder) into the window.
3. Pick the accuracy level and tick the outputs you want.
4. Press **Start**.

Video (mp4, mkv, mov, avi, webm, …) and audio (**mp3**, wav, m4a, flac, opus, …)
are both accepted, so a recorded lecture goes in exactly like a video does.

Only one copy runs at a time. Launching it again brings the existing window
forward rather than starting a second process, because two copies would
compete for the same model files and stall each other's downloads.

Results are written next to the source file by default, or to a folder you
choose:

| Output | File |
| --- | --- |
| Subtitles | `MyLecture.srt` |
| WebVTT | `MyLecture.vtt` |
| Plain text | `MyLecture.txt` |
| General summary | `MyLecture - summary.pdf` |
| Chronological summary | `MyLecture - chronological summary.pdf` |

### The two summaries

They answer different questions, so you can tick either or both.

**General summary** — what the recording was about, as a briefing: an overview,
key points, chapter markers and any action items. Material is grouped by theme,
not by when it was said.

**Chronological summary** — a walk through the recording in order, so you can
scrub straight to the part you need. Each section is a topic with the time range
it occupied:

```
Learning rates and convergence        [00:00 - 05:12]
    Covers what happens when the learning rate is too large ...

Momentum                              [05:12 - 07:30]
    Momentum accumulates a moving average of past gradients ...
```

The sections join end to end with no gaps, and a contents strip at the top of
the PDF lists every section with its timestamp.

### Speaker labels

Tick **Label who is speaking** and the app tags subtitles, the transcript and
both summaries with `Speaker 1`, `Speaker 2`, and so on.

The number of speakers is **detected automatically** — you do not have to supply
it. If you do set a number and more voices turn up, the extra ones (usually a
brief interjection, or background noise) are grouped together as `Misc` rather
than cluttering the transcript with near-empty speakers.

Subtitles are split at the point the speaker changes, not just tagged, so a cue
never straddles two people.

### Where models are kept

The first run of any model downloads it automatically. After that it is instant.

Models are large — `large-v3` is 3.1 GB and a summary model another 2 GB. They
go in `data\models` inside the install folder by default. The **Model storage**
box shows the path and how much room is left there; **Change…** moves them to
another drive, taking the download cache with them.

The app checks free space before it starts and refuses with a clear message
rather than failing part-way through a 3 GB download.

## Choosing settings on a 6 GB GPU

The app was tuned for a 6 GB card (GTX 1660 Ti and similar).

**Accuracy** — `large-v3` is the default and the most accurate. At the default
`int8_float16` precision it uses roughly 1.7 GB of VRAM, so it fits comfortably.
`distil-large-v3` is about twice as fast with almost the same quality, but it is
English-only. Drop to `medium` or `small` if you want speed above all.

**Precision** — `int8_float16` is the fastest option on this class of card and
uses the least memory. Switch to `float16` only if you want the last fraction of
a percent of accuracy and don't mind roughly 3.1 GB of VRAM.

**Summary model** — Qwen2.5 3B (about 2 GB) is the default and is quick. Qwen2.5
7B gives noticeably better summaries but needs about 4.7 GB, so it will be
slower on a 6 GB card once context is included; it still works.

Whisper is fully unloaded from the GPU before the summary model loads, so the
two never compete for VRAM even on a small card.

**Rough speed**, 1 hour of video on a GTX 1660 Ti: about 6–10 minutes with
`large-v3` at `int8_float16`, plus 2–5 minutes if you also generate a summary.

## How it works

```
video / audio
      │
      ▼
PyAV (bundled FFmpeg)
      │
      ├──▶ faster-whisper / CTranslate2 ──▶ segments + word timestamps
      │              (GPU)                          │
      │                                             ▼
      └──▶ sherpa-onnx diarisation ──────▶ speaker labels merged in
                     (CPU)                          │
                                    ┌───────────────┼───────────────┐
                                    ▼               ▼               ▼
                          SRT / VTT / TXT     llama.cpp + GGUF (GPU)
                                                    │
                                        ┌───────────┴───────────┐
                                        ▼                       ▼
                                 general summary        chronological
                                      PDF                summary PDF
```

- **faster-whisper** runs the same OpenAI Whisper weights on the CTranslate2
  runtime — roughly four times faster than the reference implementation and far
  lighter on VRAM, which is what makes `large-v3` practical on 6 GB.
- **PyAV** ships its own FFmpeg libraries, so there is no separate FFmpeg to
  install or put on PATH.
- **Word-level timestamps** are used to split long segments on real word
  boundaries. Cues are sized evenly rather than greedily: the number of cues is
  worked out first and each aims for the average, so you never get a full line
  followed by a half-second scrap reading "convergence." A short cue that
  continues the previous sentence is folded back into it, unless the speaker
  changed, a sentence ended, or there was a real pause.
- **The general summary** is map-reduce: the transcript is chunked with overlap,
  each chunk becomes notes, and the notes are composed into one briefing.
- **The chronological summary** works chunk by chunk in order, asking the model
  to divide each chunk into topics with start and end timestamps. Timestamps
  outside the chunk are clamped, a topic repeated across a chunk boundary is
  merged, and each section is stretched to meet the next so the timeline has no
  holes.
- **Diarisation** uses sherpa-onnx: ONNX segmentation and speaker-embedding
  models, about 48 MB in total. It was chosen over pyannote and WhisperX because
  it needs no PyTorch download, no Hugging Face token and no gated licence
  acceptance. It runs on the CPU, so it never competes with Whisper for VRAM.

## Project layout

```
myna/
├─ Install.bat / install.ps1      one-click installer
├─ Uninstall.bat / uninstall.ps1  removal
├─ requirements.txt
├─ app/
│  ├─ main.py                     entry point
│  ├─ ui/
│  │  ├─ main_window.py           the window
│  │  └─ models_dialog.py         "Manage models" dialog
│  └─ core/
│     ├─ transcriber.py           faster-whisper wrapper
│     ├─ diarization.py           sherpa-onnx speaker labelling
│     ├─ subtitles.py             SRT / VTT / TXT writers
│     ├─ summarizer.py            general + chronological summarisation
│     ├─ pdf_report.py            ReportLab layout for both PDFs
│     ├─ pipeline.py              background worker
│     ├─ llm_models.py            GGUF catalogue
│     ├─ whisper_models.py        Whisper catalogue
│     ├─ gpu_budget.py            splits a model between GPU and CPU RAM
│     ├─ gguf_meta.py             reads layer counts from GGUF headers
│     ├─ model_cache.py           lock recovery, download progress, watchdog
│     ├─ download_worker.py       downloads in a killable subprocess
│     ├─ model_inventory.py       lists and safely removes downloaded models
│     ├─ branding.py              app name and Windows identity
│     ├─ cuda_setup.py            finds the pip-installed CUDA DLLs
│     ├─ errors.py                shared exceptions
│     └─ paths.py                 model and log locations
├─ assets/                        app icon
└─ tools/
   ├─ make_icon.py                generates the app icon
   └─ make_launcher.py            builds Myna.exe with the icon embedded
```

## Troubleshooting

**It says it's running on CPU.** The app needs the CUDA DLLs that ship as pip
packages (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`). Re-run `Install.bat`; the
last step prints how many CUDA devices it can see. An NVIDIA driver of 527 or
newer is required.

**Out of memory.** Lower *Precision* to `int8`, or pick a smaller model. If you
enabled summaries with the 7B model, switch to the 3B one. Whisper, diarisation
and the summary model never run at the same time: Whisper is unloaded before the
summary model loads, and diarisation runs on the CPU.

**Not enough disk space.** Use *Model storage → Change…* to move the model
folder to a drive with more room. Models already downloaded stay where they
were; you can delete the old folder yourself.

**Speaker labels look wrong.** Diarisation is the least exact part of the
pipeline. If two people have similar voices they may be merged; if one person's
audio changes a lot they may be split. Setting the expected number of speakers
usually helps, since surplus clusters then collapse into `Misc`.

**The summary PDF didn't appear.** `llama-cpp-python` has no wheel on PyPI for
Windows, so the installer fetches it from the project's own wheel index. If that
step failed you'll have seen a warning during install — re-run `Install.bat`.
Transcription is unaffected either way.

**"Failed to load shared library ... llama.dll".** The DLL is there, but a CUDA
dependency is not. The `cu124` build of llama.cpp needs `cudart64_12.dll` from
`nvidia-cuda-runtime-cu12`, on top of the cuBLAS and cuDNN packages Whisper
needs. Re-running `Install.bat` installs it.

**"illegal instruction", or error `0xc000001d`, when a summary starts.** The
current `llama-cpp-python` wheels (0.3.35 and nearby) are compiled with AVX-512.
AMD Zen 1–3 and Intel chips before Skylake-X do not have it, so the process dies
at context creation — before the GPU is used at all, which is why it looks like a
GPU fault when it is not. `Install.bat` pins **0.3.4**, which is built for AVX2
and keeps CUDA offload. If you upgrade `llama-cpp-python` yourself, check this
first.

To see what your CPU actually supports, the app's own diagnosis was: AVX2 yes,
AVX-512 no. Anything without AVX-512 needs the pinned build.

**A download sits at 0% and never moves.** This is almost always a leftover
lock file. Hugging Face takes a lock on each file before downloading it; if the
process holding that lock dies - a force quit, a crash, or a second copy of the
app - the lock stays behind and the next attempt waits on it forever, with no
bytes, no timeout and no error.

Myna now clears abandoned locks automatically at the start of every run, and
refuses to start a second instance, which is what created the contention in the
first place. If you ever need to clear it by hand, close the app and delete
`data\models\whisper\.locks` and any zero-byte `*.incomplete` files.

Downloads now report real progress ("Downloading model - 420 of 3100 MB"), and
say so explicitly if no data arrives for 30 seconds.

**Something crashed.** The full traceback is in `data\whisper-studio.log`
inside the install folder.

## Licences

Whisper is MIT-licensed (OpenAI); faster-whisper and CTranslate2 are MIT and
Apache 2.0. The GGUF summary models carry their own licences — Qwen2.5 under the
Qwen licence, Llama 3.2 under the Llama Community Licence, Mistral under
Apache 2.0.
