"""
Filter panel for data filtering functionality.
"""

from typing import Optional, List, Dict, Any, Callable
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QLineEdit, QPushButton, QGroupBox,
    QScrollArea, QFrame, QDoubleSpinBox, QCheckBox,
    QMessageBox
)
from PyQt6.QtCore import pyqtSignal, Qt
import numpy as np


class FilterCondition(QFrame):
    """Single filter condition widget."""

    removed = pyqtSignal(object)  # Emits self when remove is clicked
    changed = pyqtSignal()  # Emits when condition changes

    OPERATORS = [
        ("==", "Equal to"),
        ("!=", "Not equal to"),
        (">", "Greater than"),
        (">=", "Greater or equal"),
        ("<", "Less than"),
        ("<=", "Less or equal"),
        ("between", "Between"),
        ("contains", "Contains (text)"),
    ]

    def __init__(self, columns: List[str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._columns = columns

        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.setLineWidth(1)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Column selector
        self.column_combo = QComboBox()
        self.column_combo.addItems(columns)
        self.column_combo.setMinimumWidth(120)
        self.column_combo.currentTextChanged.connect(self._on_changed)
        layout.addWidget(self.column_combo)

        # Operator selector
        self.operator_combo = QComboBox()
        for op, desc in self.OPERATORS:
            self.operator_combo.addItem(desc, op)
        self.operator_combo.currentIndexChanged.connect(self._on_operator_changed)
        layout.addWidget(self.operator_combo)

        # Value input (single)
        self.value_edit = QLineEdit()
        self.value_edit.setPlaceholderText("Value")
        self.value_edit.textChanged.connect(self._on_changed)
        layout.addWidget(self.value_edit)

        # Value input 2 (for between)
        self.value_edit2 = QLineEdit()
        self.value_edit2.setPlaceholderText("Max value")
        self.value_edit2.textChanged.connect(self._on_changed)
        self.value_edit2.hide()
        layout.addWidget(self.value_edit2)

        # Remove button
        self.remove_btn = QPushButton("X")
        self.remove_btn.setFixedWidth(30)
        self.remove_btn.clicked.connect(lambda: self.removed.emit(self))
        layout.addWidget(self.remove_btn)

    def _on_operator_changed(self, index: int):
        """Handle operator change."""
        op = self.operator_combo.currentData()
        self.value_edit2.setVisible(op == "between")
        if op == "between":
            self.value_edit.setPlaceholderText("Min value")
        else:
            self.value_edit.setPlaceholderText("Value")
        self._on_changed()

    def _on_changed(self):
        """Emit changed signal."""
        self.changed.emit()

    def set_columns(self, columns: List[str]):
        """Update available columns."""
        current = self.column_combo.currentText()
        self._columns = columns
        self.column_combo.clear()
        self.column_combo.addItems(columns)
        if current in columns:
            self.column_combo.setCurrentText(current)

    def get_condition(self) -> Dict[str, Any]:
        """Get the filter condition as a dictionary."""
        return {
            "column": self.column_combo.currentText(),
            "operator": self.operator_combo.currentData(),
            "value": self.value_edit.text(),
            "value2": self.value_edit2.text() if self.value_edit2.isVisible() else None,
        }

    def is_valid(self) -> bool:
        """Check if the condition is valid."""
        cond = self.get_condition()
        if not cond["column"]:
            return False
        if not cond["value"]:
            return False
        if cond["operator"] == "between" and not cond["value2"]:
            return False
        return True


class FilterPanel(QWidget):
    """Panel for managing filter conditions."""

    filterChanged = pyqtSignal()  # Emitted when filters change
    applyRequested = pyqtSignal()  # Emitted when Apply is clicked

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._columns: List[str] = []
        self._conditions: List[FilterCondition] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Filter conditions group
        group = QGroupBox("Filter Conditions")
        group_layout = QVBoxLayout(group)

        # Scroll area for conditions
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(150)
        scroll.setMaximumHeight(250)

        self.conditions_widget = QWidget()
        self.conditions_layout = QVBoxLayout(self.conditions_widget)
        self.conditions_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.conditions_layout.setSpacing(4)

        scroll.setWidget(self.conditions_widget)
        group_layout.addWidget(scroll)

        # Add condition button
        self.add_btn = QPushButton("+ Add Filter Condition")
        self.add_btn.clicked.connect(self._add_condition)
        group_layout.addWidget(self.add_btn)

        layout.addWidget(group)

        # Logic selector (AND/OR)
        logic_layout = QHBoxLayout()
        logic_layout.addWidget(QLabel("Combine conditions with:"))
        self.logic_combo = QComboBox()
        self.logic_combo.addItems(["AND (all must match)", "OR (any must match)"])
        self.logic_combo.currentIndexChanged.connect(lambda: self.filterChanged.emit())
        logic_layout.addWidget(self.logic_combo)
        logic_layout.addStretch()
        layout.addLayout(logic_layout)

        # Buttons
        btn_layout = QHBoxLayout()

        self.apply_btn = QPushButton("Apply Filter")
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.apply_btn.clicked.connect(self._on_apply)
        btn_layout.addWidget(self.apply_btn)

        self.clear_btn = QPushButton("Clear All Filters")
        self.clear_btn.clicked.connect(self._clear_all)
        btn_layout.addWidget(self.clear_btn)

        layout.addLayout(btn_layout)

        # Status label
        self.status_label = QLabel("No filters applied")
        self.status_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.status_label)

        layout.addStretch()

    def set_columns(self, columns: List[str]):
        """Set available columns for filtering."""
        self._columns = columns
        for cond in self._conditions:
            cond.set_columns(columns)

    def _add_condition(self):
        """Add a new filter condition."""
        if not self._columns:
            QMessageBox.warning(self, "Warning", "Please load a file first.")
            return

        cond = FilterCondition(self._columns)
        cond.removed.connect(self._remove_condition)
        cond.changed.connect(lambda: self.filterChanged.emit())

        self._conditions.append(cond)
        self.conditions_layout.addWidget(cond)
        self.filterChanged.emit()

    def _remove_condition(self, cond: FilterCondition):
        """Remove a filter condition."""
        if cond in self._conditions:
            self._conditions.remove(cond)
            self.conditions_layout.removeWidget(cond)
            cond.deleteLater()
            self.filterChanged.emit()

    def _clear_all(self):
        """Clear all filter conditions."""
        for cond in self._conditions[:]:
            self._remove_condition(cond)
        self.status_label.setText("No filters applied")
        self.filterChanged.emit()

    def _on_apply(self):
        """Handle apply button click."""
        self.applyRequested.emit()

    def get_filter_conditions(self) -> List[Dict[str, Any]]:
        """Get all valid filter conditions."""
        return [c.get_condition() for c in self._conditions if c.is_valid()]

    def get_logic(self) -> str:
        """Get the combination logic (and/or)."""
        return "and" if self.logic_combo.currentIndex() == 0 else "or"

    def set_status(self, text: str):
        """Set the status label text."""
        self.status_label.setText(text)

    def apply_filter(self, df) -> Any:
        """
        Apply filter conditions to a DataFrame.

        Args:
            df: pandas DataFrame to filter

        Returns:
            Filtered DataFrame
        """
        import pandas as pd

        conditions = self.get_filter_conditions()
        if not conditions:
            return df

        masks = []
        for cond in conditions:
            col = cond["column"]
            op = cond["operator"]
            val = cond["value"]
            val2 = cond["value2"]

            if col not in df.columns:
                continue

            col_data = df[col]

            try:
                # Try to convert value to numeric if column is numeric
                if pd.api.types.is_numeric_dtype(col_data):
                    val = float(val)
                    if val2:
                        val2 = float(val2)

                if op == "==":
                    mask = col_data == val
                elif op == "!=":
                    mask = col_data != val
                elif op == ">":
                    mask = col_data > val
                elif op == ">=":
                    mask = col_data >= val
                elif op == "<":
                    mask = col_data < val
                elif op == "<=":
                    mask = col_data <= val
                elif op == "between":
                    mask = (col_data >= val) & (col_data <= val2)
                elif op == "contains":
                    mask = col_data.astype(str).str.contains(str(val), case=False, na=False)
                else:
                    continue

                masks.append(mask)

            except (ValueError, TypeError):
                # If conversion fails, try string comparison
                if op == "==":
                    mask = col_data.astype(str) == str(val)
                elif op == "!=":
                    mask = col_data.astype(str) != str(val)
                elif op == "contains":
                    mask = col_data.astype(str).str.contains(str(val), case=False, na=False)
                else:
                    continue
                masks.append(mask)

        if not masks:
            return df

        # Combine masks
        logic = self.get_logic()
        if logic == "and":
            final_mask = masks[0]
            for m in masks[1:]:
                final_mask = final_mask & m
        else:
            final_mask = masks[0]
            for m in masks[1:]:
                final_mask = final_mask | m

        return df[final_mask]
