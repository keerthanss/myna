"""Myna main window."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QThread, QUrl, Slot
from PySide6.QtGui import QAction, QDesktopServices, QFont, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ..core import llm_models, whisper_models
from ..core.branding import APP_NAME, APP_TAGLINE
from ..core.cuda_setup import cuda_available
from ..core.gpu_budget import free_ram_gb, free_vram_gb
from ..core.paths import (
    data_dir,
    free_space_gb,
    icon_path,
    models_root,
    set_models_root,
    settings_file,
)
from ..core.pipeline import JobSettings, PipelineWorker
from .models_dialog import ModelsDialog
from ..core.transcriber import (
    MEDIA_EXTENSIONS,
    TranscribeOptions,
    is_media_file,
    resolve_compute_type,
)

# A light, material palette: paper-white surfaces on an off-white background,
# with a mosaic-myna gold used for anything actionable or in progress. Matches
# tools/make_icon.py exactly, so the app icon and the in-app theme read as one
# design rather than two - GOLD is the icon's eye-patch/beak colour verbatim,
# GOLD_DEEP its leg colour.
PAPER = "#FFFFFF"
SAND = "#F8F6F1"
LINE = "#E8E0D5"
INK = "#2B2622"
MUTED_INK = "#8A7F73"
GOLD = "#C49129"
GOLD_BRIGHT = "#D6A83D"
GOLD_DEEP = "#8C6416"
GOLD_TINT = "#F7ECD3"

STYLE = f"""
QWidget {{
    font-size: 13px;
    color: {INK};
}}
QMainWindow, QDialog {{ background: {SAND}; }}
QMenuBar {{ background: {SAND}; border-bottom: 1px solid {LINE}; }}
QMenuBar::item:selected {{ background: {GOLD_TINT}; }}
QMenu {{ background: {PAPER}; border: 1px solid {LINE}; }}
QMenu::item:selected {{ background: {GOLD_TINT}; }}
QStatusBar {{ background: {SAND}; color: {MUTED_INK}; }}
QScrollArea, QSplitter {{ background: transparent; }}

QGroupBox {{
    font-weight: 600;
    background: {PAPER};
    border: 1px solid {LINE};
    border-radius: 10px;
    margin-top: 12px;
    padding: 12px 12px 10px 12px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {GOLD_DEEP};
}}

QPushButton {{
    padding: 6px 14px;
    border-radius: 8px;
    border: 1px solid {LINE};
    background: {PAPER};
    color: {INK};
}}
QPushButton:hover {{ background: {GOLD_TINT}; border-color: #E4C888; }}
QPushButton:pressed {{ background: #F3E4C0; }}
QPushButton:disabled {{ color: #BEB4A8; background: {SAND}; }}
QPushButton#primary {{
    background: {GOLD};
    color: {PAPER};
    border: 1px solid {GOLD_DEEP};
    font-weight: 600;
    padding: 8px 24px;
}}
QPushButton#primary:hover {{ background: {GOLD_BRIGHT}; }}
QPushButton#primary:pressed {{ background: {GOLD_DEEP}; }}
QPushButton#primary:disabled {{ background: #E9DBB9; border-color: #E9DBB9; color: {PAPER}; }}
QPushButton#danger {{ color: #B4331B; border-color: #EBC4B9; }}
QPushButton#danger:hover {{ background: #FBEDE9; }}

QListWidget, QTreeWidget {{
    background: {PAPER};
    border: 1px solid {LINE};
    border-radius: 8px;
    padding: 4px;
    alternate-background-color: {SAND};
    outline: none;
    selection-background-color: {GOLD_TINT};
    selection-color: {INK};
}}
QListWidget::item, QTreeWidget::item {{ padding: 5px 6px; border-radius: 5px; }}
QListWidget::item:selected, QTreeWidget::item:selected {{ background: {GOLD_TINT}; color: {INK}; }}
QHeaderView::section {{
    background: {SAND};
    color: {MUTED_INK};
    border: none;
    border-bottom: 1px solid {LINE};
    padding: 4px 6px;
}}

QComboBox, QSpinBox, QLineEdit {{
    background: {PAPER};
    border: 1px solid {LINE};
    border-radius: 7px;
    padding: 4px 8px;
    min-height: 20px;
}}
QComboBox:focus, QSpinBox:focus, QLineEdit:focus {{ border-color: {GOLD}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {PAPER};
    border: 1px solid {LINE};
    selection-background-color: {GOLD_TINT};
    selection-color: {INK};
    outline: none;
}}

QCheckBox {{ spacing: 8px; padding: 2px 0; }}
QCheckBox::indicator, QTreeWidget::indicator {{
    width: 16px; height: 16px;
    border: 1px solid #CDBFAE;
    border-radius: 4px;
    background: {PAPER};
}}
QCheckBox::indicator:hover, QTreeWidget::indicator:hover {{ border-color: {GOLD}; }}
QCheckBox::indicator:checked, QTreeWidget::indicator:checked {{
    background: {GOLD}; border-color: {GOLD_DEEP};
}}
QCheckBox:disabled {{ color: #BEB4A8; }}

QPlainTextEdit {{
    background: {PAPER};
    border: 1px solid {LINE};
    border-radius: 8px;
    font-family: Consolas, 'Courier New', monospace;
    font-size: 12px;
    color: #4A423B;
}}

QProgressBar {{
    background: {SAND};
    border: 1px solid {LINE};
    border-radius: 8px;
    height: 18px;
    text-align: center;
    color: {INK};
}}
QProgressBar::chunk {{ background: {GOLD}; border-radius: 7px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #DED3C4; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: #C9BBA8; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QLabel#hint {{ color: {MUTED_INK}; font-size: 12px; }}
QLabel#appTitle {{ color: {GOLD_DEEP}; }}
QLabel#dropHint {{
    color: {MUTED_INK};
    font-size: 13px;
    padding: 26px 18px;
    border: 2px dashed #E6D6A8;
    border-radius: 10px;
    background: {GOLD_TINT};
}}
"""


class MediaList(QListWidget):
    """File queue that accepts drops of videos, audio and folders."""

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setAlternatingRowColors(True)

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        paths: list[Path] = []
        for url in event.mimeData().urls():
            local = Path(url.toLocalFile())
            if local.is_dir():
                paths.extend(sorted(p for p in local.rglob("*") if is_media_file(p)))
            elif is_media_file(local):
                paths.append(local)
        if paths:
            self.window().add_files(paths)
            event.acceptProposedAction()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1020, 740)
        self.setStyleSheet(STYLE)

        # An INI beside the app rather than the Windows registry, so the whole
        # installation stays self-contained on whichever drive it lives on.
        self.settings = QSettings(str(settings_file()), QSettings.IniFormat)
        self._files: list[Path] = []
        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None
        self._last_output_dir: Path | None = None

        self._build_ui()
        self._restore_settings()
        self._refresh_device_hint()
        self._update_start_state()

    # ---------------------------------------------------------------- layout

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 12, 14, 10)
        root.setSpacing(10)

        header_row = QHBoxLayout()
        header_row.setSpacing(10)

        logo_path = icon_path()
        if logo_path is not None:
            logo = QLabel()
            logo.setPixmap(
                QPixmap(str(logo_path)).scaled(
                    36, 36, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
            header_row.addWidget(logo)

        title_column = QVBoxLayout()
        title_column.setSpacing(0)
        header = QLabel(APP_NAME)
        header_font = QFont()
        header_font.setPointSize(15)
        header_font.setBold(True)
        header.setFont(header_font)
        header.setObjectName("appTitle")
        subtitle = QLabel(
            APP_TAGLINE
        )
        subtitle.setObjectName("hint")
        title_column.addWidget(header)
        title_column.addWidget(subtitle)
        header_row.addLayout(title_column)
        header_row.addStretch(1)
        root.addLayout(header_row)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self._build_top_area())
        splitter.addWidget(self._build_log_area())
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

        root.addLayout(self._build_progress_area())
        root.addLayout(self._build_button_row())

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self._build_menu()

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        add_action = QAction("Add files…", self)
        add_action.setShortcut("Ctrl+O")
        add_action.triggered.connect(self.browse_files)
        file_menu.addAction(add_action)

        folder_action = QAction("Add folder…", self)
        folder_action.triggered.connect(self.browse_folder)
        file_menu.addAction(folder_action)

        file_menu.addSeparator()
        models_action = QAction("Open model folder", self)
        models_action.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(models_root())))
        )
        file_menu.addAction(models_action)

        change_models_action = QAction("Change model folder…", self)
        change_models_action.triggered.connect(self.choose_models_folder)
        file_menu.addAction(change_models_action)

        manage_models_action = QAction("Manage downloaded models…", self)
        manage_models_action.triggered.connect(self.manage_models)
        file_menu.addAction(manage_models_action)

        file_menu.addSeparator()
        quit_action = QAction("Exit", self)
        quit_action.setShortcut("Alt+F4")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        help_menu = self.menuBar().addMenu("&Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _build_top_area(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(self._build_files_panel(), 3)
        layout.addWidget(self._build_options_panel(), 2)
        return container

    def _build_files_panel(self) -> QWidget:
        box = QGroupBox("Files to process")
        layout = QVBoxLayout(box)

        self.drop_hint = QLabel(
            "Drag videos or a folder here — or use the buttons below.\n"
            "Supports mp4, mkv, mov, avi, webm and common audio files."
        )
        self.drop_hint.setObjectName("dropHint")
        self.drop_hint.setAlignment(Qt.AlignCenter)
        self.drop_hint.setWordWrap(True)
        layout.addWidget(self.drop_hint)

        self.file_list = MediaList()
        self.file_list.setVisible(False)
        layout.addWidget(self.file_list, 1)

        buttons = QHBoxLayout()
        self.add_button = QPushButton("Add files…")
        self.add_button.clicked.connect(self.browse_files)
        self.folder_button = QPushButton("Add folder…")
        self.folder_button.clicked.connect(self.browse_folder)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_selected)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_files)
        for button in (
            self.add_button, self.folder_button, self.remove_button, self.clear_button
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        return box

    def _build_options_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # --- transcription -------------------------------------------------
        transcription = QGroupBox("Transcription")
        form = QFormLayout(transcription)
        form.setLabelAlignment(Qt.AlignRight)

        self.model_combo = QComboBox()
        for choice in whisper_models.WHISPER_MODELS:
            self.model_combo.addItem(choice.label, choice.key)
        self.model_combo.currentIndexChanged.connect(self._refresh_device_hint)
        form.addRow("Accuracy", self.model_combo)

        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto (use GPU if available)", "auto")
        self.device_combo.addItem("GPU (CUDA)", "cuda")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        form.addRow("Run on", self.device_combo)

        self.compute_combo = QComboBox()
        self.compute_combo.currentIndexChanged.connect(self._refresh_device_hint)
        form.addRow("Precision", self.compute_combo)

        self.language_combo = QComboBox()
        for code, label in whisper_models.LANGUAGES:
            self.language_combo.addItem(label, code)
        form.addRow("Language", self.language_combo)

        self.task_combo = QComboBox()
        self.task_combo.addItem("Transcribe (keep original language)", "transcribe")
        self.task_combo.addItem("Translate to English", "translate")
        form.addRow("Mode", self.task_combo)

        self.beam_spin = QSpinBox()
        self.beam_spin.setRange(1, 10)
        self.beam_spin.setValue(5)
        self.beam_spin.setToolTip("Higher is slightly more accurate and slower.")
        form.addRow("Beam size", self.beam_spin)

        self.vad_check = QCheckBox("Skip silence (recommended)")
        self.vad_check.setChecked(True)
        form.addRow("", self.vad_check)

        self.device_hint = QLabel()
        self.device_hint.setObjectName("hint")
        self.device_hint.setWordWrap(True)
        form.addRow("", self.device_hint)
        layout.addWidget(transcription)

        # --- outputs -------------------------------------------------------
        outputs = QGroupBox("Output files")
        out_layout = QVBoxLayout(outputs)

        self.srt_check = QCheckBox("Subtitles (.srt)")
        self.srt_check.setChecked(True)
        self.srt_check.stateChanged.connect(self._update_start_state)
        self.vtt_check = QCheckBox("WebVTT (.vtt)")
        self.vtt_check.stateChanged.connect(self._update_start_state)
        self.txt_check = QCheckBox("Plain text (.txt)")
        self.txt_check.stateChanged.connect(self._update_start_state)
        for check in (self.srt_check, self.vtt_check, self.txt_check):
            out_layout.addWidget(check)

        folder_row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Same folder as the video")
        self.output_edit.setReadOnly(True)
        browse = QPushButton("Choose…")
        browse.clicked.connect(self.choose_output_folder)
        reset = QPushButton("Reset")
        reset.clicked.connect(lambda: self.output_edit.clear())
        folder_row.addWidget(QLabel("Save to"))
        folder_row.addWidget(self.output_edit, 1)
        folder_row.addWidget(browse)
        folder_row.addWidget(reset)
        out_layout.addLayout(folder_row)
        layout.addWidget(outputs)

        # --- speakers ------------------------------------------------------
        speakers = QGroupBox("Speakers (optional)")
        speakers_layout = QVBoxLayout(speakers)

        self.diarize_check = QCheckBox("Label who is speaking")
        self.diarize_check.stateChanged.connect(self._on_diarize_toggled)
        speakers_layout.addWidget(self.diarize_check)

        speakers_form = QFormLayout()
        self.speaker_count_spin = QSpinBox()
        self.speaker_count_spin.setRange(0, 20)
        self.speaker_count_spin.setSpecialValueText("Detect automatically")
        self.speaker_count_spin.setValue(0)
        self.speaker_count_spin.setToolTip(
            "Leave on automatic unless you know the number. If you set a number "
            "and more voices are found, the extras are grouped as 'Misc'."
        )
        speakers_form.addRow("How many", self.speaker_count_spin)
        speakers_layout.addLayout(speakers_form)

        self.speaker_hint = QLabel(
            "Detects speakers automatically and tags subtitles and summaries with "
            "Speaker 1, Speaker 2, and so on. Runs on the CPU (~48 MB of models, "
            "downloaded once)."
        )
        self.speaker_hint.setObjectName("hint")
        self.speaker_hint.setWordWrap(True)
        speakers_layout.addWidget(self.speaker_hint)
        layout.addWidget(speakers)

        # --- summary -------------------------------------------------------
        summary = QGroupBox("Summary PDFs (optional)")
        summary_layout = QVBoxLayout(summary)

        self.summary_check = QCheckBox("General summary — overview, key points, actions")
        self.summary_check.stateChanged.connect(self._on_summary_toggled)
        summary_layout.addWidget(self.summary_check)

        self.chronological_check = QCheckBox(
            "Chronological summary — each topic with its time range"
        )
        self.chronological_check.setToolTip(
            "Walks the recording in order: 'Topic X [00:00 - 05:00]' followed by "
            "'Topic Y [05:00 - 07:30]', each with a short summary."
        )
        self.chronological_check.stateChanged.connect(self._on_summary_toggled)
        summary_layout.addWidget(self.chronological_check)

        summary_form = QFormLayout()
        self.llm_combo = QComboBox()
        for choice in llm_models.LLM_MODELS:
            self.llm_combo.addItem(choice.label, choice.key)
        self.llm_combo.addItem("Custom GGUF file…", "custom")
        self.llm_combo.currentIndexChanged.connect(self._on_llm_changed)
        summary_form.addRow("Model", self.llm_combo)

        custom_row = QHBoxLayout()
        self.llm_path_edit = QLineEdit()
        self.llm_path_edit.setPlaceholderText("Path to a .gguf file")
        self.llm_path_edit.setReadOnly(True)
        self.llm_browse = QPushButton("Browse…")
        self.llm_browse.clicked.connect(self.choose_gguf)
        custom_row.addWidget(self.llm_path_edit, 1)
        custom_row.addWidget(self.llm_browse)
        self.custom_row_widget = QWidget()
        self.custom_row_widget.setLayout(custom_row)
        self.custom_row_widget.setVisible(False)
        summary_form.addRow("", self.custom_row_widget)
        summary_layout.addLayout(summary_form)

        self.llm_gpu_check = QCheckBox("Use the GPU for summarising")
        self.llm_gpu_check.setChecked(True)
        self.llm_gpu_check.setToolTip(
            "On: as many layers as fit go on the GPU and the rest run on the "
            "CPU, which is how models larger than 6 GB are supported.\n"
            "Off: everything runs on the CPU — much slower, but it frees the "
            "GPU entirely."
        )
        self.transcript_check = QCheckBox("Append the full transcript to the PDF")
        for check in (self.llm_gpu_check, self.transcript_check):
            summary_layout.addWidget(check)

        self.llm_size_hint = QLabel()
        self.llm_size_hint.setObjectName("hint")
        self.llm_size_hint.setWordWrap(True)
        summary_layout.addWidget(self.llm_size_hint)

        self.summary_hint = QLabel(
            "The model downloads once on first use. Whisper is unloaded first, so the "
            "two never compete for VRAM."
        )
        self.summary_hint.setObjectName("hint")
        self.summary_hint.setWordWrap(True)
        summary_layout.addWidget(self.summary_hint)
        layout.addWidget(summary)

        self._summary_widgets = [
            self.llm_combo, self.llm_gpu_check, self.transcript_check,
            self.custom_row_widget,
        ]
        self._on_summary_toggled()
        self._on_diarize_toggled()

        # --- storage -------------------------------------------------------
        storage = QGroupBox("Model storage")
        storage_layout = QVBoxLayout(storage)
        storage_row = QHBoxLayout()
        self.models_path_label = QLabel()
        self.models_path_label.setObjectName("hint")
        self.models_path_label.setWordWrap(True)
        change_models = QPushButton("Change…")
        change_models.clicked.connect(self.choose_models_folder)
        storage_row.addWidget(self.models_path_label, 1)
        storage_row.addWidget(change_models)
        storage_layout.addLayout(storage_row)
        layout.addWidget(storage)
        self._refresh_models_path()

        layout.addStretch(1)

        # The options column is long; scrolling keeps the window usable on
        # smaller screens without shrinking the file queue.
        scroll = QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        scroll.setMinimumWidth(380)
        return scroll

    def _build_log_area(self) -> QWidget:
        box = QGroupBox("Activity")
        layout = QVBoxLayout(box)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(4000)
        layout.addWidget(self.log_view)
        return box

    def _build_progress_area(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        self.current_label = QLabel("Ready.")
        self.current_label.setObjectName("hint")
        self.file_bar = QProgressBar()
        self.file_bar.setRange(0, 1000)
        self.overall_bar = QProgressBar()
        self.overall_bar.setRange(0, 1000)
        self.overall_bar.setFormat("Overall — %p%")
        layout.addWidget(self.current_label)
        layout.addWidget(self.file_bar)
        layout.addWidget(self.overall_bar)
        return layout

    def _build_button_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.open_output_button = QPushButton("Open output folder")
        self.open_output_button.clicked.connect(self.open_output_folder)
        self.open_output_button.setEnabled(False)
        row.addWidget(self.open_output_button)

        # Lives in the always-visible bottom bar on purpose: the options column
        # scrolls, and at a normal window size anything near its end (the
        # Model storage box included) starts well below the visible area.
        self.manage_models_button = QPushButton("Manage models…")
        self.manage_models_button.setToolTip(
            "See what Myna has downloaded and delete models you no longer need."
        )
        self.manage_models_button.clicked.connect(self.manage_models)
        row.addWidget(self.manage_models_button)
        row.addStretch(1)

        self.cancel_button = QPushButton("Stop")
        self.cancel_button.setObjectName("danger")
        self.cancel_button.clicked.connect(self.cancel_run)
        self.cancel_button.setEnabled(False)
        row.addWidget(self.cancel_button)

        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self.start_run)
        row.addWidget(self.start_button)
        return row

    # ------------------------------------------------------------ file queue

    def add_files(self, paths: list[Path]) -> None:
        added = 0
        existing = set(self._files)
        for path in paths:
            if path in existing or not is_media_file(path):
                continue
            self._files.append(path)
            existing.add(path)
            item = QListWidgetItem(f"{path.name}   —   queued")
            item.setToolTip(str(path))
            self.file_list.addItem(item)
            added += 1
        if added:
            self.append_log(f"Added {added} file(s).")
        self._sync_list_visibility()
        self._update_start_state()

    @Slot()
    def browse_files(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in MEDIA_EXTENSIONS)
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Choose video or audio files",
            self.settings.value("last_input_dir", "", str),
            f"Media files ({patterns});;All files (*.*)",
        )
        if files:
            self.settings.setValue("last_input_dir", str(Path(files[0]).parent))
            self.add_files([Path(f) for f in files])

    @Slot()
    def browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Choose a folder", self.settings.value("last_input_dir", "", str)
        )
        if folder:
            self.settings.setValue("last_input_dir", folder)
            found = sorted(p for p in Path(folder).rglob("*") if is_media_file(p))
            if not found:
                QMessageBox.information(
                    self, "Nothing found", "That folder has no video or audio files."
                )
                return
            self.add_files(found)

    @Slot()
    def remove_selected(self) -> None:
        for item in self.file_list.selectedItems():
            row = self.file_list.row(item)
            self.file_list.takeItem(row)
            del self._files[row]
        self._sync_list_visibility()
        self._update_start_state()

    @Slot()
    def clear_files(self) -> None:
        self._files.clear()
        self.file_list.clear()
        self._sync_list_visibility()
        self._update_start_state()

    def _sync_list_visibility(self) -> None:
        has_files = bool(self._files)
        self.file_list.setVisible(has_files)
        self.drop_hint.setVisible(not has_files)

    # -------------------------------------------------------------- settings

    @Slot()
    def choose_output_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Where should the results go?", self.output_edit.text()
        )
        if folder:
            self.output_edit.setText(folder)

    def _refresh_models_path(self) -> None:
        root = models_root()
        free = free_space_gb(root)
        warning = "  ⚠ low space" if free < 6 else ""
        self.models_path_label.setText(f"{root}\n{free:.1f} GB free{warning}")

    @Slot()
    def choose_models_folder(self) -> None:
        """Models are several GB; the system drive is often not where they fit."""
        folder = QFileDialog.getExistingDirectory(
            self, "Where should downloaded models be kept?", str(models_root())
        )
        if not folder:
            return
        set_models_root(Path(folder))
        self._refresh_models_path()
        self.append_log(f"Models will be stored in {folder}")
        QMessageBox.information(
            self,
            "Model folder changed",
            f"New downloads will go to:\n{folder}\n\n"
            f"Restart {APP_NAME} before downloading anything new, so the "
            "download cache moves there too.\n\n"
            "Models already downloaded stay where they are. You can move or "
            "delete the old folder yourself if you want the space back.",
        )

    @Slot()
    def manage_models(self) -> None:
        """Open the downloaded-models manager."""
        if self._thread is not None:
            # A running job holds models open (memory-mapped, on Windows that
            # blocks deletion) and may be mid-download into the same folders.
            QMessageBox.information(
                self,
                "Busy",
                "Models cannot be removed while a job is running. "
                "Wait for it to finish, or press Stop first.",
            )
            return

        llm_key = self.llm_combo.currentData()
        dialog = ModelsDialog(
            self,
            models_root(),
            selected_whisper=self.model_combo.currentData() or "",
            selected_llm="" if llm_key == "custom" else (llm_key or ""),
        )
        dialog.exec()
        if dialog.changed:
            self._refresh_models_path()
            self._refresh_llm_size_hint()
            self.append_log("Removed downloaded models; they re-download when needed.")

    @Slot()
    def choose_gguf(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a GGUF model", str(data_dir()), "GGUF models (*.gguf)"
        )
        if path:
            self.llm_path_edit.setText(path)

    @Slot()
    def _on_device_changed(self) -> None:
        device = self.device_combo.currentData()
        effective = device if device != "auto" else ("cuda" if cuda_available() else "cpu")
        current = self.compute_combo.currentData()
        self.compute_combo.clear()
        self.compute_combo.addItem("Auto (recommended)", "auto")
        for key, label in whisper_models.COMPUTE_TYPES[effective]:
            self.compute_combo.addItem(label, key)
        index = self.compute_combo.findData(current)
        self.compute_combo.setCurrentIndex(max(0, index))
        self._refresh_device_hint()

    @Slot()
    def _on_summary_toggled(self) -> None:
        enabled = self._wants_summary()
        for widget in self._summary_widgets:
            widget.setEnabled(enabled)
        self.summary_hint.setEnabled(enabled)
        if enabled:
            self._on_llm_changed()
        self._update_start_state()

    def _wants_summary(self) -> bool:
        return self.summary_check.isChecked() or self.chronological_check.isChecked()

    @Slot()
    def _on_diarize_toggled(self) -> None:
        enabled = self.diarize_check.isChecked()
        self.speaker_count_spin.setEnabled(enabled)
        self.speaker_hint.setEnabled(enabled)

    @Slot()
    def _on_llm_changed(self) -> None:
        is_custom = self.llm_combo.currentData() == "custom"
        self.custom_row_widget.setVisible(is_custom)
        self.custom_row_widget.setEnabled(self._wants_summary())
        self._refresh_llm_size_hint()

    def _refresh_llm_size_hint(self) -> None:
        """Say plainly what the chosen model will do on this machine.

        Model size alone does not tell the user much; what matters is whether
        it fits the card, how much spills to CPU RAM, and whether there is
        enough free RAM for that spill not to page to disk.
        """
        if not hasattr(self, "llm_size_hint"):
            return

        key = self.llm_combo.currentData()
        if key == "custom":
            self.llm_size_hint.setText(
                "Custom GGUF: the GPU/CPU split is worked out from the file itself."
            )
            return

        choice = llm_models.find_choice(key)
        if choice is None:
            self.llm_size_hint.clear()
            return

        free_vram = free_vram_gb()
        free_ram = free_ram_gb()
        on_disk = " (already downloaded)" if llm_models.is_downloaded(choice) else ""

        if choice.fits_6gb_vram:
            self.llm_size_hint.setText(
                f"{choice.approx_gb:.1f} GB{on_disk} — runs entirely on the GPU."
            )
            return

        usable = max(0.0, (free_vram or 0.0) - 1.1)
        overflow = max(0.0, choice.approx_gb - usable)
        parts = [
            f"{choice.approx_gb:.1f} GB{on_disk} — larger than VRAM, so about "
            f"{overflow:.1f} GB runs on the CPU. Slower, better summaries."
        ]
        if free_ram is not None and overflow > free_ram:
            parts.append(
                f"⚠ Only {free_ram:.1f} GB RAM free — close other apps first, "
                "or this will page to disk and crawl."
            )
        free_disk = free_space_gb(models_root())
        if not llm_models.is_downloaded(choice) and choice.approx_gb * 1.15 > free_disk:
            parts.append(
                f"⚠ Needs ~{choice.approx_gb * 1.15:.1f} GB to download but only "
                f"{free_disk:.1f} GB is free in the model folder."
            )
        self.llm_size_hint.setText("  ".join(parts))

    def _refresh_device_hint(self) -> None:
        if not hasattr(self, "device_hint"):
            return  # called while the options panel is still being built
        device = self.device_combo.currentData()
        has_cuda = cuda_available()
        if device == "cpu" or (device == "auto" and not has_cuda):
            self.device_hint.setText(
                "Running on CPU — expect roughly real-time speed or slower."
            )
            return
        if device == "cuda" and not has_cuda:
            self.device_hint.setText(
                "No CUDA device detected. The app will fall back to CPU at run time."
            )
            return

        compute = self.compute_combo.currentData() or "auto"
        if compute == "auto":
            compute = resolve_compute_type("cuda", "auto")
        estimate = whisper_models.estimate_vram_gb(
            self.model_combo.currentData(), compute
        )
        if estimate is None:
            self.device_hint.setText("GPU ready.")
            return
        self.device_hint.setText(
            f"GPU ready. This model at {compute} needs roughly "
            f"{estimate:.1f} GB of VRAM."
        )

    def _update_start_state(self) -> None:
        if not hasattr(self, "start_button"):
            return  # called while the panels are still being built
        wants_output = (
            self.srt_check.isChecked()
            or self.vtt_check.isChecked()
            or self.txt_check.isChecked()
            or self._wants_summary()
        )
        running = self._thread is not None
        self.start_button.setEnabled(bool(self._files) and wants_output and not running)
        self.manage_models_button.setEnabled(not running)

    # ------------------------------------------------------------------- run

    def _collect_settings(self) -> JobSettings:
        options = TranscribeOptions(
            model=self.model_combo.currentData(),
            device=self.device_combo.currentData(),
            compute_type=self.compute_combo.currentData() or "auto",
            language=self.language_combo.currentData() or "",
            task=self.task_combo.currentData(),
            beam_size=self.beam_spin.value(),
            vad_filter=self.vad_check.isChecked(),
        )
        output_dir = self.output_edit.text().strip()
        llm_key = self.llm_combo.currentData()
        return JobSettings(
            transcribe=options,
            output_dir=Path(output_dir) if output_dir else None,
            write_srt=self.srt_check.isChecked(),
            write_vtt=self.vtt_check.isChecked(),
            write_txt=self.txt_check.isChecked(),
            diarize=self.diarize_check.isChecked(),
            expected_speakers=self.speaker_count_spin.value(),
            make_summary=self.summary_check.isChecked(),
            make_chronological=self.chronological_check.isChecked(),
            llm_key="" if llm_key == "custom" else llm_key,
            llm_custom_path=(
                self.llm_path_edit.text().strip() if llm_key == "custom" else ""
            ),
            llm_on_gpu=self.llm_gpu_check.isChecked(),
            include_transcript_in_pdf=self.transcript_check.isChecked(),
        )

    @Slot()
    def start_run(self) -> None:
        if self._thread is not None:
            return
        settings = self._collect_settings()
        if settings.needs_llm and self.llm_combo.currentData() == "custom":
            if not Path(settings.llm_custom_path or "").is_file():
                QMessageBox.warning(
                    self, "Pick a model", "Choose a .gguf file for the summary model."
                )
                return

        self._save_settings()
        self.log_view.clear()
        self.append_log(f"Starting {len(self._files)} file(s).")
        for index in range(self.file_list.count()):
            self.file_list.item(index).setText(
                f"{self._files[index].name}   —   queued"
            )

        self._thread = QThread(self)
        self._worker = PipelineWorker(list(self._files), settings)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self.append_log)
        self._worker.file_started.connect(self._on_file_started)
        self._worker.file_progress.connect(self._on_file_progress)
        self._worker.file_finished.connect(self._on_file_finished)
        self._worker.overall_progress.connect(
            lambda value: self.overall_bar.setValue(int(value * 1000))
        )
        self._worker.outputs_ready.connect(self._on_outputs_ready)
        self._worker.finished.connect(self._on_run_finished)

        self.start_button.setEnabled(False)
        self.manage_models_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setText("Stop")
        self.file_bar.setValue(0)
        self.overall_bar.setValue(0)
        self._thread.start()

    @Slot()
    def cancel_run(self) -> None:
        if self._worker is None:
            return
        self._worker.cancel()
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Stopping…")
        self.current_label.setText(
            "Stopping — finishing the current step, then releasing the models."
        )
        self.file_bar.setFormat("Stopping… %p%")
        # Anything not started will not be started now.
        for index in range(self.file_list.count()):
            if "queued" in self.file_list.item(index).text():
                self._set_item_status(index, "skipped")

    @Slot(int, str)
    def _on_file_started(self, index: int, name: str) -> None:
        self.current_label.setText(f"Working on {name}")
        self.file_bar.setValue(0)
        self.file_bar.setFormat(f"{name} — %p%")
        self._set_item_status(index, "working…")

    @Slot(int, float, str)
    def _on_file_progress(self, index: int, value: float, stage: str) -> None:
        self.file_bar.setValue(int(value * 1000))
        self.file_bar.setFormat(f"{stage} — %p%")
        self._set_item_status(index, f"{stage} {value:.0%}")

    @Slot(int, bool, str)
    def _on_file_finished(self, index: int, ok: bool, message: str) -> None:
        self._set_item_status(index, "done" if ok else f"failed — {message}")

    @Slot(int, list)
    def _on_outputs_ready(self, index: int, paths: list) -> None:
        for path in paths:
            self.append_log(f"  wrote {path}")
        if paths:
            self._last_output_dir = Path(paths[0]).parent
            self.open_output_button.setEnabled(True)

    @Slot()
    def _on_run_finished(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Stop")
        self.current_label.setText("Finished.")
        self.file_bar.setFormat("%p%")
        self.statusBar().showMessage("All done.", 6000)
        self._update_start_state()

    def _set_item_status(self, index: int, status: str) -> None:
        if 0 <= index < self.file_list.count():
            self.file_list.item(index).setText(
                f"{self._files[index].name}   —   {status}"
            )

    @Slot(str)
    def append_log(self, message: str) -> None:
        self.log_view.appendPlainText(message)

    @Slot()
    def open_output_folder(self) -> None:
        target = self._last_output_dir
        if target and target.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            f"About {APP_NAME}",
            f"<b>{APP_NAME}</b><br><br>"
            "Named after the hill myna, a bird that reproduces almost any "
            "sound it hears.<br><br>"
            "Transcribes video to subtitles with OpenAI Whisper (via faster-whisper) "
            "and writes summary PDFs with a local GGUF language model.<br><br>"
            "Everything runs on this machine — nothing is uploaded.<br><br>"
            f"Models and logs live in:<br><code>{data_dir()}</code>",
        )

    # ------------------------------------------------------------ persistence

    def _save_settings(self) -> None:
        store = {
            "model": self.model_combo.currentData(),
            "device": self.device_combo.currentData(),
            "compute": self.compute_combo.currentData(),
            "language": self.language_combo.currentData(),
            "task": self.task_combo.currentData(),
            "beam": self.beam_spin.value(),
            "vad": self.vad_check.isChecked(),
            "srt": self.srt_check.isChecked(),
            "vtt": self.vtt_check.isChecked(),
            "txt": self.txt_check.isChecked(),
            "output_dir": self.output_edit.text(),
            "diarize": self.diarize_check.isChecked(),
            "speaker_count": self.speaker_count_spin.value(),
            "summary": self.summary_check.isChecked(),
            "chronological": self.chronological_check.isChecked(),
            "llm": self.llm_combo.currentData(),
            "llm_path": self.llm_path_edit.text(),
            "llm_gpu": self.llm_gpu_check.isChecked(),
            "pdf_transcript": self.transcript_check.isChecked(),
        }
        for key, value in store.items():
            self.settings.setValue(key, value)

    def _restore_settings(self) -> None:
        get = self.settings.value

        def select(combo: QComboBox, value, fallback_index: int = 0) -> None:
            index = combo.findData(value)
            combo.setCurrentIndex(index if index >= 0 else fallback_index)

        select(
            self.model_combo,
            get("model", whisper_models.DEFAULT_MODEL, str),
            self.model_combo.findData(whisper_models.DEFAULT_MODEL),
        )
        select(self.device_combo, get("device", "auto", str))
        self._on_device_changed()
        select(self.compute_combo, get("compute", "auto", str))
        select(self.language_combo, get("language", "", str))
        select(self.task_combo, get("task", "transcribe", str))
        self.beam_spin.setValue(int(get("beam", 5)))
        self.vad_check.setChecked(get("vad", True, bool))
        self.srt_check.setChecked(get("srt", True, bool))
        self.vtt_check.setChecked(get("vtt", False, bool))
        self.txt_check.setChecked(get("txt", False, bool))
        self.output_edit.setText(get("output_dir", "", str))
        self.diarize_check.setChecked(get("diarize", False, bool))
        self.speaker_count_spin.setValue(int(get("speaker_count", 0)))
        self.summary_check.setChecked(get("summary", False, bool))
        self.chronological_check.setChecked(get("chronological", False, bool))
        select(self.llm_combo, get("llm", llm_models.DEFAULT_LLM, str))
        self.llm_path_edit.setText(get("llm_path", "", str))
        self.llm_gpu_check.setChecked(get("llm_gpu", True, bool))
        self.transcript_check.setChecked(get("pdf_transcript", False, bool))
        self._on_summary_toggled()
        # Populate on startup too: selecting the already-selected model fires
        # no change event, so the hint would otherwise stay blank until the
        # user touched the dropdown.
        self._refresh_llm_size_hint()
        self._on_diarize_toggled()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._thread is not None:
            answer = QMessageBox.question(
                self,
                "Still working",
                "A transcription is in progress. Stop it and quit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            if self._worker is not None:
                self._worker.cancel()
            if self._thread is not None:
                self._thread.quit()
                self._thread.wait(15000)
        self._save_settings()
        event.accept()
