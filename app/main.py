"""Myna entry point."""

from __future__ import annotations

import logging
import sys
import time
import traceback
from pathlib import Path

# Register the bundled CUDA DLLs before anything can import ctranslate2.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.branding import (  # noqa: E402
    APP_NAME,
    APP_USER_MODEL_ID,
)
from app.core.cuda_setup import ensure_cuda_dlls  # noqa: E402
from app.core.paths import icon_path, log_file  # noqa: E402

ensure_cuda_dlls()

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication, QIcon  # noqa: E402
from PySide6.QtNetwork import QLocalServer, QLocalSocket  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from app.ui.main_window import MainWindow  # noqa: E402

SINGLE_INSTANCE_KEY = "Myna-single-instance"

# The QLocalServer must outlive this function or the socket closes with it.
_keep_alive: list = []


def _configure_logging() -> None:
    logging.basicConfig(
        filename=str(log_file()),
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
    )


def _set_windows_identity() -> None:
    """Tell Windows this is its own application, not Python.

    Without an explicit AppUserModelID the taskbar groups the window under the
    Python interpreter and shows Python's icon regardless of the window icon.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            APP_USER_MODEL_ID
        )
    except Exception:  # noqa: BLE001 - cosmetic only, never worth crashing over
        logging.debug("Could not set the AppUserModelID", exc_info=True)


def app_icon() -> QIcon:
    path = icon_path()
    return QIcon(str(path)) if path else QIcon()


def _another_instance_running() -> bool:
    """True when a copy is already running, having nudged it to the front.

    Two instances would fight over the same model cache: Hugging Face locks a
    file before downloading it, and a second process waits on that lock with no
    timeout, which looks exactly like a frozen download.
    """
    probe = QLocalSocket()
    probe.connectToServer(SINGLE_INSTANCE_KEY)
    if probe.waitForConnected(500):
        probe.write(b"raise")
        probe.waitForBytesWritten(500)
        probe.disconnectFromServer()
        return True

    # No listener: either nobody is running, or a previous crash left the pipe
    # name behind. removeServer clears a stale name; a kill can leave one around
    # briefly, so a failed listen is retried once rather than silently leaving
    # the app with no guard at all.
    server = QLocalServer()
    for attempt in range(2):
        QLocalServer.removeServer(SINGLE_INSTANCE_KEY)
        if server.listen(SINGLE_INSTANCE_KEY):
            _keep_alive.append(server)
            return False
        logging.warning(
            "Could not claim the single-instance socket (attempt %d): %s",
            attempt + 1,
            server.errorString(),
        )
        time.sleep(0.4)

    logging.error("Running without a single-instance guard.")
    return False


def _install_excepthook() -> None:
    def hook(exc_type, exc_value, exc_tb):
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logging.error("Unhandled exception:\n%s", text)
        QMessageBox.critical(
            None,
            f"{APP_NAME} hit an error",
            f"{exc_value}\n\nFull details were written to:\n{log_file()}",
        )

    sys.excepthook = hook


def main() -> int:
    _configure_logging()
    _set_windows_identity()
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    app.setDesktopFileName(APP_NAME)
    app.setStyle("Fusion")
    app.setWindowIcon(app_icon())
    _install_excepthook()

    if _another_instance_running():
        # The running copy has been asked to come forward. Showing a dialog here
        # would only add something to dismiss.
        logging.info("Another instance is running; raised it and exited.")
        return 0

    window = MainWindow()
    window.setWindowIcon(app_icon())
    window.show()

    # Bring the window forward when a second launch is attempted.
    if _keep_alive:
        server = _keep_alive[0]
        server.newConnection.connect(
            lambda: (window.showNormal(), window.raise_(), window.activateWindow())
        )

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
