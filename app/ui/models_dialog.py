"""The "Manage downloaded models" dialog."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from ..core import llm_models, whisper_models
from ..core.model_inventory import (
    KIND_LLM,
    KIND_WHISPER,
    InstalledModel,
    delete,
    human_size,
    list_installed,
)
from ..core.paths import free_space_gb

_ROLE_MODEL = Qt.UserRole + 1


class ModelsDialog(QDialog):
    """Lists what Myna has downloaded and lets the user remove any of it.

    `changed` is set when anything was deleted, so the main window knows to
    refresh its storage and model hints afterwards.
    """

    def __init__(
        self,
        parent,
        root: Path,
        selected_whisper: str = "",
        selected_llm: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Downloaded models")
        self.resize(720, 440)
        self.changed = False
        self._root = root
        self._selected_whisper = selected_whisper
        self._selected_llm = selected_llm

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(8)

        title = QLabel("Downloaded models")
        title_font = QFont()
        title_font.setPointSize(13)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setObjectName("appTitle")
        layout.addWidget(title)

        explainer = QLabel(
            "Remove models you no longer need. Anything removed downloads again "
            "automatically the next time it is needed."
        )
        explainer.setObjectName("hint")
        explainer.setWordWrap(True)
        layout.addWidget(explainer)

        self.folder_label = QLabel()
        self.folder_label.setObjectName("hint")
        self.folder_label.setWordWrap(True)
        layout.addWidget(self.folder_label)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Model", "Used for", "Size", "Status"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.NoSelection)
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in (1, 2, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self.tree.itemChanged.connect(self._update_selection_total)
        layout.addWidget(self.tree, 1)

        self.empty_label = QLabel(
            "Nothing downloaded yet. Models download automatically the first "
            "time they are needed."
        )
        self.empty_label.setObjectName("dropHint")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setWordWrap(True)
        layout.addWidget(self.empty_label, 1)

        buttons = QHBoxLayout()
        self.select_all_button = QPushButton("Select all")
        self.select_all_button.clicked.connect(self._toggle_all)
        buttons.addWidget(self.select_all_button)
        buttons.addStretch(1)
        self.selection_label = QLabel()
        self.selection_label.setObjectName("hint")
        buttons.addWidget(self.selection_label)
        self.delete_button = QPushButton("Delete selected…")
        self.delete_button.setObjectName("danger")
        self.delete_button.clicked.connect(self._delete_selected)
        buttons.addWidget(self.delete_button)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

        self.refresh()

    # ----------------------------------------------------------------- state

    def _status_for(self, model: InstalledModel) -> str:
        if model.note:
            return model.note
        if model.kind == KIND_WHISPER and model.key == self._selected_whisper:
            return "Selected in Myna"
        if model.kind == KIND_LLM and model.key == self._selected_llm:
            return "Selected in Myna"
        return ""

    def refresh(self) -> None:
        self.tree.blockSignals(True)
        self.tree.clear()
        models = list_installed(self._root)
        for model in models:
            row = QTreeWidgetItem(
                [model.label, model.kind, human_size(model.size_bytes), self._status_for(model)]
            )
            row.setFlags(row.flags() | Qt.ItemIsUserCheckable)
            row.setCheckState(0, Qt.Unchecked)
            row.setTextAlignment(2, Qt.AlignRight | Qt.AlignVCenter)
            row.setData(0, _ROLE_MODEL, model)
            self.tree.addTopLevelItem(row)
        self.tree.blockSignals(False)

        has_models = bool(models)
        self.tree.setVisible(has_models)
        self.empty_label.setVisible(not has_models)
        self.select_all_button.setEnabled(has_models)

        used = sum(m.size_bytes for m in models)
        free = free_space_gb(self._root)
        self.folder_label.setText(
            f"Model folder: {self._root}\n"
            f"{human_size(used)} used by Myna · {free:.1f} GB free on this drive"
        )
        self._update_selection_total()

    def _checked(self) -> list[InstalledModel]:
        chosen: list[InstalledModel] = []
        for index in range(self.tree.topLevelItemCount()):
            row = self.tree.topLevelItem(index)
            if row.checkState(0) == Qt.Checked:
                chosen.append(row.data(0, _ROLE_MODEL))
        return chosen

    def _update_selection_total(self, *_args) -> None:
        chosen = self._checked()
        total = sum(m.size_bytes for m in chosen)
        self.delete_button.setEnabled(bool(chosen))
        self.selection_label.setText(
            f"Selected: {human_size(total)}" if chosen else ""
        )
        all_checked = bool(chosen) and len(chosen) == self.tree.topLevelItemCount()
        self.select_all_button.setText("Select none" if all_checked else "Select all")

    def _toggle_all(self) -> None:
        count = self.tree.topLevelItemCount()
        target = Qt.Unchecked if len(self._checked()) == count else Qt.Checked
        self.tree.blockSignals(True)
        for index in range(count):
            self.tree.topLevelItem(index).setCheckState(0, target)
        self.tree.blockSignals(False)
        self._update_selection_total()

    # ---------------------------------------------------------------- delete

    def _redownload_warning(self, model: InstalledModel) -> str | None:
        """A line for the confirmation when a model the user relies on goes."""
        if model.kind == KIND_WHISPER and model.key == self._selected_whisper:
            choice = whisper_models.find_model(model.key)
            size = f" (~{choice.download_gb:.1f} GB)" if choice else ""
            return (
                f"{model.label} is your current transcription model — it will "
                f"download again{size} the next time you press Start."
            )
        if model.kind == KIND_LLM and model.key == self._selected_llm:
            choice = llm_models.find_choice(model.key)
            size = f" (~{choice.approx_gb:.1f} GB)" if choice else ""
            return (
                f"{model.label} is your current summary model — it will download "
                f"again{size} the next time you make a summary."
            )
        return None

    def _delete_selected(self) -> None:
        chosen = self._checked()
        if not chosen:
            return

        total = sum(m.size_bytes for m in chosen)
        lines = [f"•  {m.label} — {human_size(m.size_bytes)}" for m in chosen]
        warnings = [w for w in (self._redownload_warning(m) for m in chosen) if w]

        message = "Delete these from disk?\n\n" + "\n".join(lines)
        message += f"\n\nThis frees about {human_size(total)}."
        if warnings:
            message += "\n\n" + "\n\n".join(warnings)

        answer = QMessageBox.warning(
            self,
            "Delete downloaded models",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        freed, errors = delete(chosen, self._root)
        self.changed = self.changed or freed > 0
        self.refresh()

        if errors:
            QMessageBox.warning(
                self,
                "Some files could not be deleted",
                f"Freed {human_size(freed)}, but these could not be removed:\n\n"
                + "\n".join(errors)
                + "\n\nA file may be in use. Close anything using it and try again.",
            )
        else:
            self.selection_label.setText(f"Freed {human_size(freed)}.")
