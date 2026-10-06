"""One place for the application's identity.

Named after the hill myna, a bird celebrated across South Asia for reproducing
almost any sound it hears - which is the job.
"""

from __future__ import annotations

APP_NAME = "Myna"
APP_TAGLINE = "Transcribe video and audio to subtitles and summaries, on this PC."
APP_VERSION = "1.0.0"

# Windows groups taskbar buttons and picks the taskbar icon by this id. Without
# it the app inherits Python's identity and shows Python's icon, however the
# window icon is set.
APP_USER_MODEL_ID = "Myna.Transcriber.1"

# The launcher executable. A venv's pythonw.exe redirects to the base
# interpreter, so Task Manager shows "pythonw.exe" (or the base install's path).
# The installer places a real interpreter under this name instead.
LAUNCHER_EXE = "Myna.exe"
ICON_FILENAME = "Myna.ico"

# Folder name for per-user data. Kept as the original so an existing install
# keeps finding its models after the rename.
DATA_DIR_NAME = "WhisperStudio"
