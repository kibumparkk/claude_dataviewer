"""
File management panel for loading and managing data files.
"""

from typing import Optional, List
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QListWidgetItem, QFileDialog, QGroupBox,
    QMessageBox
)
from PyQt6.QtCore import pyqtSignal, Qt


class FilePanel(QWidget):
    """
    Panel for managing loaded data files.
    """

    fileAdded = pyqtSignal(str)  # Emits file path
    fileRemoved = pyqtSignal(str)  # Emits file path
    fileSelected = pyqtSignal(str)  # Emits file path when selection changes

    SUPPORTED_FILTERS = (
        "All Supported (*.csv *.xlsx *.parquet *.ftr *.feather *.zip *.gz);;"
        "CSV Files (*.csv);;"
        "Excel Files (*.xlsx);;"
        "Parquet Files (*.parquet);;"
        "Feather Files (*.ftr *.feather);;"
        "ZIP Archives (*.zip);;"
        "GZipped CSV (*.gz);;"
        "All Files (*)"
    )

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._file_paths: dict[str, str] = {}  # display_name -> full_path

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Group box
        group = QGroupBox("Files")
        group_layout = QVBoxLayout(group)

        # File list
        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(80)
        self.file_list.setMaximumHeight(150)
        self.file_list.itemSelectionChanged.connect(self._on_selection_changed)
        group_layout.addWidget(self.file_list)

        # Buttons
        button_layout = QHBoxLayout()

        self.add_button = QPushButton("Add...")
        self.add_button.clicked.connect(self._on_add_clicked)
        button_layout.addWidget(self.add_button)

        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self._on_remove_clicked)
        self.remove_button.setEnabled(False)
        button_layout.addWidget(self.remove_button)

        self.clear_button = QPushButton("Clear All")
        self.clear_button.clicked.connect(self._on_clear_clicked)
        self.clear_button.setEnabled(False)
        button_layout.addWidget(self.clear_button)

        group_layout.addLayout(button_layout)
        layout.addWidget(group)

    def _on_add_clicked(self):
        """Handle add file button click."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open Data Files",
            "",
            self.SUPPORTED_FILTERS
        )

        for file_path in file_paths:
            self.add_file(file_path)

    def _on_remove_clicked(self):
        """Handle remove file button click."""
        current_item = self.file_list.currentItem()
        if current_item:
            display_name = current_item.text()
            file_path = self._file_paths.get(display_name)
            if file_path:
                self.remove_file(file_path)

    def _on_clear_clicked(self):
        """Handle clear all button click."""
        reply = QMessageBox.question(
            self,
            "Clear All Files",
            "Are you sure you want to remove all files?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.clear_all()

    def _on_selection_changed(self):
        """Handle file selection change."""
        current_item = self.file_list.currentItem()
        self.remove_button.setEnabled(current_item is not None)

        if current_item:
            display_name = current_item.text()
            file_path = self._file_paths.get(display_name)
            if file_path:
                self.fileSelected.emit(file_path)

    def add_file(self, file_path: str) -> bool:
        """
        Add a file to the panel.

        Args:
            file_path: Path to the file

        Returns:
            True if file was added successfully
        """
        # Check if already added
        if file_path in self._file_paths.values():
            return False

        display_name = Path(file_path).name
        # Handle duplicate names
        base_name = display_name
        counter = 1
        while display_name in self._file_paths:
            display_name = f"{base_name} ({counter})"
            counter += 1

        self._file_paths[display_name] = file_path

        item = QListWidgetItem(display_name)
        item.setToolTip(file_path)
        self.file_list.addItem(item)
        self.file_list.setCurrentItem(item)

        self.clear_button.setEnabled(True)
        self.fileAdded.emit(file_path)

        return True

    def remove_file(self, file_path: str):
        """
        Remove a file from the panel.

        Args:
            file_path: Path to the file to remove
        """
        # Find display name
        display_name = None
        for name, path in self._file_paths.items():
            if path == file_path:
                display_name = name
                break

        if display_name is None:
            return

        # Remove from dict
        del self._file_paths[display_name]

        # Remove from list widget
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item and item.text() == display_name:
                self.file_list.takeItem(i)
                break

        self.clear_button.setEnabled(self.file_list.count() > 0)
        self.fileRemoved.emit(file_path)

    def clear_all(self):
        """Remove all files."""
        paths = list(self._file_paths.values())
        for path in paths:
            self.remove_file(path)

    def get_selected_file(self) -> Optional[str]:
        """Get the currently selected file path."""
        current_item = self.file_list.currentItem()
        if current_item:
            display_name = current_item.text()
            return self._file_paths.get(display_name)
        return None

    def get_all_files(self) -> List[str]:
        """Get all loaded file paths."""
        return list(self._file_paths.values())

    def select_file(self, file_path: str):
        """
        Select a file in the list.

        Args:
            file_path: Path to the file to select
        """
        for name, path in self._file_paths.items():
            if path == file_path:
                for i in range(self.file_list.count()):
                    item = self.file_list.item(i)
                    if item and item.text() == name:
                        self.file_list.setCurrentItem(item)
                        break
                break
