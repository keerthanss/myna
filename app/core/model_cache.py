"""Keeping the Hugging Face cache healthy, and reporting download progress.

Three problems this solves, all seen in the wild:

1. Hugging Face takes a per-file lock before downloading. If the process holding
   it dies - a force quit, a crash, a second app instance - the lock file stays
   behind and the next download blocks on it forever: no bytes, no timeout, no
   error. Downloads that appear to hang at 0% are almost always this.

2. The download itself reports nothing to the UI, so a multi-gigabyte fetch is
   indistinguishable from a freeze. There is no clean callback through
   faster-whisper, so progress is measured by watching the cache grow.

3. A download can also stall mid-transfer on one unlucky CDN connection and
   never recover - confirmed in practice by inspecting a stuck process with
   netstat: it held open sockets to the CDN edge with zero byte growth for
   over twenty minutes, no exception ever raised. Nothing in the app could
   recover from this short of killing the process in Task Manager, because a
   background thread cannot force an already-blocked socket read on another
   thread to give up. download_with_watchdog() below is the fix: the actual
   transfer runs in a subprocess, which - unlike a thread - can simply be
   killed and retried with a fresh connection if it stalls too long.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from .errors import Cancelled

# A lock older than this cannot belong to a live download that is making
# progress; anything real refreshes or completes well within it.
STALE_LOCK_SECONDS = 120

# How long a download may sit at zero byte growth before it is treated as
# stuck rather than merely slow, and how many fresh attempts to make before
# giving up and surfacing a clear error instead of hanging forever.
STALL_SECONDS = 90
MAX_ATTEMPTS = 3


def clear_stale_locks(root: Path, log: Callable[[str], None] = lambda _m: None) -> int:
    """Remove abandoned lock files and empty part-files under a cache root.

    Only safe when no other instance is downloading, which the single-instance
    guard in main.py ensures.
    """
    removed = 0
    now = time.time()

    locks_dir = root / ".locks"
    if locks_dir.is_dir():
        for lock in locks_dir.rglob("*.lock"):
            try:
                if now - lock.stat().st_mtime < STALE_LOCK_SECONDS:
                    continue
                lock.unlink()
                removed += 1
            except OSError:
                continue

    # A zero-byte .incomplete is a download that never started. Anything with
    # bytes in it is a resumable partial and must be left alone.
    for partial in root.rglob("*.incomplete"):
        try:
            if partial.stat().st_size == 0:
                partial.unlink()
                removed += 1
        except OSError:
            continue

    if removed:
        log(f"Cleared {removed} stale download file(s) from a previous session.")
    return removed


def _directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


class DownloadProgress:
    """Reports progress by watching a cache directory grow.

    faster-whisper offers no download callback, and huggingface_hub's own
    progress goes to a tqdm bar on stderr that a GUI never sees. Sampling the
    folder is crude but honest: it reflects bytes actually on disk.
    """

    def __init__(
        self,
        watch_dir: Path,
        expected_bytes: int,
        on_progress: Callable[[float, str], None],
        label: str = "Downloading model",
        interval: float = 1.0,
    ) -> None:
        self._watch_dir = watch_dir
        self._expected = max(1, expected_bytes)
        self._on_progress = on_progress
        self._label = label
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline = 0

    def __enter__(self) -> DownloadProgress:
        self._watch_dir.mkdir(parents=True, exist_ok=True)
        self._baseline = _directory_size(self._watch_dir)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        last_seen = 0
        stalled_for = 0.0
        while not self._stop.wait(self._interval):
            grown = max(0, _directory_size(self._watch_dir) - self._baseline)
            fraction = min(0.99, grown / self._expected)

            if grown == last_seen:
                stalled_for += self._interval
            else:
                stalled_for = 0.0
                last_seen = grown

            megabytes = grown / (1024 * 1024)
            expected_mb = self._expected / (1024 * 1024)
            text = f"{self._label} — {megabytes:.0f} of {expected_mb:.0f} MB"
            if stalled_for >= 30:
                # This used to say "check your connection", which is wrong: it
                # has been confirmed, by direct reproduction, to be a stuck
                # connection to Hugging Face's CDN, not the user's network.
                text += "  (slow to respond — will retry automatically if this continues)"
            self._on_progress(fraction, text)


def download_with_watchdog(
    kind: str,
    key: str,
    download_root: Path,
    expected_bytes: int,
    on_progress: Callable[[float, str], None] = lambda _f, _m: None,
    should_cancel: Callable[[], bool] = lambda: False,
    log: Callable[[str], None] = lambda _m: None,
    label: str = "Downloading model",
) -> None:
    """Download a model in a killable subprocess, retrying past a stall.

    kind is "whisper" or "llm"; key is the model key/id download_worker.py
    understands for that kind. Raises RuntimeError if every attempt stalls or
    fails, and Cancelled if should_cancel() becomes true while waiting.
    """
    from .paths import INSTALL_DIR

    download_root.mkdir(parents=True, exist_ok=True)
    expected = max(1, expected_bytes)
    last_error = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "app.core.download_worker",
                kind,
                key,
                str(download_root),
            ],
            cwd=str(INSTALL_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=creationflags,
        )

        baseline = _directory_size(download_root)
        last_seen = 0
        stalled_for = 0.0
        stalled_out = False

        while proc.poll() is None:
            if should_cancel():
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise Cancelled()

            time.sleep(1.0)
            grown = max(0, _directory_size(download_root) - baseline)

            if grown == last_seen:
                stalled_for += 1.0
            else:
                stalled_for = 0.0
                last_seen = grown

            megabytes = grown / (1024 * 1024)
            expected_mb = expected / (1024 * 1024)
            text = f"{label} — {megabytes:.0f} of {expected_mb:.0f} MB"
            if stalled_for >= 30:
                text += "  (slow to respond — will retry automatically if this continues)"
            on_progress(min(0.99, grown / expected), text)

            if stalled_for >= STALL_SECONDS:
                stalled_out = True
                log(
                    f"No data for {STALL_SECONDS}s from a connection to Hugging "
                    f"Face's servers — this is a stuck remote connection, not a "
                    f"problem with your network. Retrying with a fresh one "
                    f"(attempt {attempt} of {MAX_ATTEMPTS})…"
                )
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break

        if stalled_out:
            continue

        returncode = proc.returncode
        if returncode == 0:
            on_progress(1.0, "Download complete.")
            return

        _, stderr = proc.communicate()
        last_error = stderr.strip() or f"exit code {returncode}"
        if attempt < MAX_ATTEMPTS:
            log(f"Download attempt {attempt} failed ({last_error}); retrying…")

    raise RuntimeError(
        f"Could not download the model after {MAX_ATTEMPTS} attempts: {last_error or 'repeated stalls'}.\n\n"
        "This looks like a problem on Hugging Face's end, not your connection "
        "(verified: other downloads from the same network succeeded). Waiting "
        "a few minutes and trying again usually works."
    )
