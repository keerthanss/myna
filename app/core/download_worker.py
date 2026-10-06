"""Runs a single Hugging Face model download in its own process.

Why a subprocess: a download can stall on one unlucky CDN edge connection and
never recover - no exception, no timeout, zero bytes, forever. This has been
observed in practice (see HANDOFF.md). A background thread cannot forcibly
interrupt a blocking C-level socket read happening on another thread, so
there is no way to recover from this within a single process. A subprocess
can simply be killed: the parent watches for the stall and terminates it,
then retries in a *new* process, which gets a fresh DNS resolution and fresh
sockets - reliably avoiding whatever edge server was stuck.

Invoked as: <python> -m app.core.download_worker <whisper|llm> <key> <download_root>
Exit code 0 on success; non-zero with a message on stderr otherwise.
"""

from __future__ import annotations

import sys


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "usage: download_worker.py <whisper|llm> <key> <download_root>",
            file=sys.stderr,
        )
        return 2

    kind, key, download_root = sys.argv[1], sys.argv[2], sys.argv[3]

    try:
        if kind == "whisper":
            from faster_whisper.utils import download_model

            download_model(key, cache_dir=download_root)
        elif kind == "llm":
            from .llm_models import download as download_llm
            from .llm_models import find_choice

            choice = find_choice(key)
            if choice is None:
                print(f"Unknown summary model key: {key}", file=sys.stderr)
                return 3
            download_llm(choice)
        else:
            print(f"Unknown download kind: {kind}", file=sys.stderr)
            return 2
    except Exception as exc:  # noqa: BLE001 - surfaced via exit code + stderr
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
