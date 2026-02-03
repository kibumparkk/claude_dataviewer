"""
Signal selection widgets for X and Y axis selection.
"""

from typing import List, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QLineEdit, QComboBox,
    QGroupBox, QGridLayout, QAbstractItemView
)
from PyQt6.QtCore import pyqtSignal, Qt


EMPTY_SIGNAL = ""  # Empty signal placeholder


class SignalSelector(QWidget):
    """
    Base signal selector with search filter and list widget.
    """

    selectionChanged = pyqtSignal(list)  # Emits list of selected signal names

    def __init__(
        self,
        title: str = "Signals",
        multi_select: bool = True,
        visible_items: int = 6,
        add_empty_option: bool = False,
        show_search: bool = True,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self._all_signals: List[str] = []
        self._multi_select = multi_select
        self._add_empty_option = add_empty_option

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Title label
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.title_label)

        # Search filter (optional)
        self.search_edit = None
        if show_search:
            self.search_edit = QLineEdit()
            self.search_edit.setPlaceholderText("Filter...")
            self.search_edit.textChanged.connect(self._filter_signals)
            layout.addWidget(self.search_edit)

        # List widget
        self.list_widget = QListWidget()
        if multi_select:
            self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        else:
            self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        # Set fixed height based on visible items
        item_height = 20  # Approximate height per item
        self.list_widget.setMinimumHeight(item_height * visible_items)
        self.list_widget.setMaximumHeight(item_height * (visible_items + 1))

        self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.list_widget)

    def set_signals(self, signals: List[str]):
        """
        Set the available signals.

        Args:
            signals: List of signal names
        """
        self._all_signals = signals.copy()
        self._populate_list(signals)

    def _populate_list(self, signals: List[str]):
        """Populate the list widget with signals."""
        self.list_widget.clear()

        # Add empty option at top if enabled
        if self._add_empty_option:
            item = QListWidgetItem("(None)")
            item.setData(Qt.ItemDataRole.UserRole, EMPTY_SIGNAL)
            self.list_widget.addItem(item)

        for signal in signals:
            item = QListWidgetItem(signal)
            item.setData(Qt.ItemDataRole.UserRole, signal)
            self.list_widget.addItem(item)

    def _filter_signals(self, text: str):
        """Filter signals based on search text."""
        text = text.lower()
        filtered = [s for s in self._all_signals if text in s.lower()]
        self._populate_list(filtered)

    def _on_selection_changed(self):
        """Handle selection change."""
        selected = self.get_selected()
        self.selectionChanged.emit(selected)

    def get_selected(self) -> List[str]:
        """Get list of selected signal names."""
        result = []
        for item in self.list_widget.selectedItems():
            signal = item.data(Qt.ItemDataRole.UserRole)
            if signal is not None and signal != EMPTY_SIGNAL:
                result.append(signal)
        return result

    def clear_selection(self):
        """Clear all selections."""
        self.list_widget.clearSelection()

    def select_signals(self, signals: List[str]):
        """
        Programmatically select signals.

        Args:
            signals: List of signal names to select
        """
        self.list_widget.clearSelection()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item:
                signal = item.data(Qt.ItemDataRole.UserRole)
                if signal and signal in signals:
                    item.setSelected(True)


class XAxisSelector(QWidget):
    """
    X-axis selector using a combo box.
    Supports shared mode (single selector) or independent mode (4 selectors).
    """

    selectionChanged = pyqtSignal(str)  # Emits selected signal name (shared mode)
    selectionChangedMulti = pyqtSignal(int, str)  # Emits (subplot_index, signal) for independent mode

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._signals: List[str] = []
        self._share_x = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Group box
        self.group = QGroupBox("X Axis")
        self.group_layout = QVBoxLayout(self.group)
        self.group_layout.setSpacing(4)

        # Shared mode: single combo box
        self.shared_combo = QComboBox()
        self.shared_combo.currentTextChanged.connect(self._on_shared_selection_changed)
        self.group_layout.addWidget(self.shared_combo)

        # Independent mode: 4 combo boxes (hidden by default)
        # Layout: n by 2 (n rows, 2 columns: label + combo)
        self.independent_widget = QWidget()
        self.independent_layout = QGridLayout(self.independent_widget)
        self.independent_layout.setContentsMargins(0, 0, 0, 0)
        self.independent_layout.setSpacing(2)

        self.independent_combos: List[QComboBox] = []
        titles = ["Plot 1:", "Plot 2:", "Plot 3:", "Plot 4:"]

        # n by 2 layout: each row has label (col 0) + combo (col 1)
        for i, title in enumerate(titles):
            label = QLabel(title)
            label.setStyleSheet("font-size: 10px;")
            label.setFixedWidth(45)

            combo = QComboBox()
            combo.currentTextChanged.connect(lambda text, idx=i: self._on_independent_selection_changed(idx, text))

            self.independent_layout.addWidget(label, i, 0)
            self.independent_layout.addWidget(combo, i, 1)
            self.independent_combos.append(combo)

        # Set column stretch for combo box column to expand
        self.independent_layout.setColumnStretch(1, 1)

        self.independent_widget.hide()
        self.group_layout.addWidget(self.independent_widget)

        layout.addWidget(self.group)

    def set_share_x(self, share: bool):
        """
        Set whether X axis is shared across subplots.

        Args:
            share: True for shared mode, False for independent mode
        """
        self._share_x = share
        self.shared_combo.setVisible(share)
        self.independent_widget.setVisible(not share)

        if share:
            self.group.setTitle("X Axis (Shared)")
        else:
            self.group.setTitle("X Axis (Per Subplot)")

    def set_signals(self, signals: List[str]):
        """
        Set available signals for X axis.

        Args:
            signals: List of signal names
        """
        self._signals = signals.copy()

        # Update shared combo
        current_shared = self.shared_combo.currentText()
        self.shared_combo.clear()
        self.shared_combo.addItems(signals)
        if current_shared and current_shared in signals:
            self.shared_combo.setCurrentText(current_shared)

        # Update independent combos
        for combo in self.independent_combos:
            current = combo.currentText()
            combo.clear()
            combo.addItems(signals)
            if current and current in signals:
                combo.setCurrentText(current)

    def _on_shared_selection_changed(self, text: str):
        """Handle shared mode selection change."""
        self.selectionChanged.emit(text)

    def _on_independent_selection_changed(self, index: int, text: str):
        """Handle independent mode selection change."""
        self.selectionChangedMulti.emit(index, text)

    def get_selected(self) -> str:
        """Get the selected X axis signal (shared mode)."""
        return self.shared_combo.currentText()

    def get_selected_multi(self) -> List[str]:
        """Get selected X axis signals for each subplot (independent mode)."""
        return [combo.currentText() for combo in self.independent_combos]

    def is_share_x(self) -> bool:
        """Check if in shared X mode."""
        return self._share_x

    def select_signal(self, signal: str):
        """
        Programmatically select a signal (shared mode).

        Args:
            signal: Signal name to select
        """
        index = self.shared_combo.findText(signal)
        if index >= 0:
            self.shared_combo.setCurrentIndex(index)

    def select_signal_multi(self, index: int, signal: str):
        """
        Programmatically select a signal for specific subplot.

        Args:
            index: Subplot index (0-3)
            signal: Signal name to select
        """
        if 0 <= index < len(self.independent_combos):
            combo = self.independent_combos[index]
            idx = combo.findText(signal)
            if idx >= 0:
                combo.setCurrentIndex(idx)


class YAxisSelectorGrid(QWidget):
    """
    Grid of Y-axis selectors for 2x2 subplot layout.
    """

    selectionChanged = pyqtSignal(int, list)  # Emits (subplot_index, selected_signals)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._all_signals: List[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Group box
        group = QGroupBox("Y Axis (2x2 Subplots)")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(4)

        # Shared search filter at top
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Filter Y signals...")
        self.search_edit.textChanged.connect(self._filter_all_signals)
        group_layout.addWidget(self.search_edit)

        # Grid for 4 selectors
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(4)

        # Create 4 signal selectors in 2x2 grid (without individual search)
        self.selectors: List[SignalSelector] = []
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        titles = ["Plot 1 (Top-Left)", "Plot 2 (Top-Right)",
                  "Plot 3 (Bottom-Left)", "Plot 4 (Bottom-Right)"]

        for i, (row, col) in enumerate(positions):
            selector = SignalSelector(
                title=titles[i],
                multi_select=True,
                visible_items=5,
                add_empty_option=True,
                show_search=False  # No individual search
            )
            selector.selectionChanged.connect(lambda selected, idx=i: self._on_selector_changed(idx, selected))
            grid_layout.addWidget(selector, row, col)
            self.selectors.append(selector)

        group_layout.addWidget(grid_widget)
        layout.addWidget(group)

    def set_signals(self, signals: List[str]):
        """
        Set available signals for all Y axis selectors.

        Args:
            signals: List of signal names
        """
        self._all_signals = signals.copy()
        for selector in self.selectors:
            selector.set_signals(signals)

    def _filter_all_signals(self, text: str):
        """Filter signals in all selectors based on search text."""
        text = text.lower()
        filtered = [s for s in self._all_signals if text in s.lower()]
        for selector in self.selectors:
            selector._populate_list(filtered)

    def _on_selector_changed(self, index: int, selected: List[str]):
        """Handle selection change in a selector."""
        self.selectionChanged.emit(index, selected)

    def get_all_selected(self) -> List[List[str]]:
        """
        Get selected signals for all subplots.

        Returns:
            List of 4 lists, each containing selected signal names
        """
        return [selector.get_selected() for selector in self.selectors]

    def clear_all_selections(self):
        """Clear selections in all selectors."""
        for selector in self.selectors:
            selector.clear_selection()
