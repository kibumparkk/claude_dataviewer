"""
Calculated signal panel for creating new signals using Python expressions.
"""

import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGroupBox, QLineEdit, QTextEdit,
    QListWidget, QListWidgetItem, QMessageBox,
    QComboBox, QSplitter, QTabWidget
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont
import pandas as pd
import numpy as np


# Config file path
CONFIG_DIR = Path.home() / ".claude_dataviewer"
CALC_SIGNALS_FILE = CONFIG_DIR / "calculated_signals.json"


# Safe built-ins for expression evaluation
SAFE_BUILTINS = {
    'abs': abs,
    'min': min,
    'max': max,
    'sum': sum,
    'len': len,
    'round': round,
    'int': int,
    'float': float,
    'str': str,
    'bool': bool,
    'list': list,
    'range': range,
    'enumerate': enumerate,
    'zip': zip,
    'map': map,
    'filter': filter,
    'sorted': sorted,
    'reversed': reversed,
    'any': any,
    'all': all,
}

# Safe numpy/pandas functions
SAFE_MODULES = {
    'np': np,
    'pd': pd,
    'numpy': np,
    'pandas': pd,
}


class CalculatedSignalPanel(QWidget):
    """
    Panel for creating calculated signals using Python expressions.
    """

    signalCreated = pyqtSignal(str)  # Emits new signal name when created
    signalRemoved = pyqtSignal(str)  # Emits signal name when removed
    columnsChanged = pyqtSignal()  # Emitted when calculated columns change

    # Example templates
    TEMPLATES = [
        ("Simple Math", "col_a + col_b"),
        ("Percentage Change", "(col_a - col_b) / col_b * 100"),
        ("Moving Average", "col_a.rolling(window=20).mean()"),
        ("Cumulative Sum", "col_a.cumsum()"),
        ("Difference", "col_a.diff()"),
        ("Z-Score", "(col_a - col_a.mean()) / col_a.std()"),
        ("Log Transform", "np.log(col_a)"),
        ("DateTime Parse", "pd.to_datetime(col_a)"),
        ("DateTime to Timestamp", "pd.to_datetime(col_a).astype(np.int64) // 10**9"),
        ("Extract Hour", "pd.to_datetime(col_a).dt.hour"),
        ("Extract Date", "pd.to_datetime(col_a).dt.date"),
        ("String to Numeric", "pd.to_numeric(col_a, errors='coerce')"),
        ("Fill NaN", "col_a.fillna(0)"),
        ("Clip Values", "col_a.clip(lower=0, upper=100)"),
        ("Shift", "col_a.shift(1)"),
        ("Rank", "col_a.rank()"),
    ]

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._available_columns: List[str] = []
        self._calculated_signals: Dict[str, str] = {}  # name -> expression
        self._data_callback: Optional[Callable] = None
        self._results: Dict[str, Any] = {}

        self._setup_ui()
        self._load_saved_signals()

    def _setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Main group
        group = QGroupBox("Calculated Signals")
        group_layout = QVBoxLayout(group)

        # Tab widget for Simple / Script modes
        self.tab_widget = QTabWidget()

        # Simple expression tab
        simple_tab = self._create_simple_tab()
        self.tab_widget.addTab(simple_tab, "Expression")

        # Script tab
        script_tab = self._create_script_tab()
        self.tab_widget.addTab(script_tab, "Script")

        group_layout.addWidget(self.tab_widget)

        # Created signals list
        signals_group = QGroupBox("Created Signals")
        signals_layout = QVBoxLayout(signals_group)

        self.signals_list = QListWidget()
        self.signals_list.setMaximumHeight(100)
        signals_layout.addWidget(self.signals_list)

        # Remove button
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._on_remove_signal)
        signals_layout.addWidget(remove_btn)

        group_layout.addWidget(signals_group)

        layout.addWidget(group)

    def _create_simple_tab(self) -> QWidget:
        """Create simple expression tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(4)

        # Signal name input
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("new_signal_name")
        name_layout.addWidget(self.name_edit)
        layout.addLayout(name_layout)

        # Template selector
        template_layout = QHBoxLayout()
        template_layout.addWidget(QLabel("Template:"))
        self.template_combo = QComboBox()
        self.template_combo.addItem("-- Select Template --", "")
        for name, expr in self.TEMPLATES:
            self.template_combo.addItem(name, expr)
        self.template_combo.currentIndexChanged.connect(self._on_template_selected)
        template_layout.addWidget(self.template_combo)
        layout.addLayout(template_layout)

        # Expression input
        layout.addWidget(QLabel("Expression:"))
        self.expr_edit = QLineEdit()
        self.expr_edit.setPlaceholderText("e.g., col_a + col_b * 2")
        layout.addWidget(self.expr_edit)

        # Available columns
        layout.addWidget(QLabel("Available Columns (double-click to insert):"))
        self.columns_list = QListWidget()
        self.columns_list.setMaximumHeight(80)
        self.columns_list.itemDoubleClicked.connect(self._on_column_double_clicked)
        layout.addWidget(self.columns_list)

        # Create button
        self.create_btn = QPushButton("Create Signal")
        self.create_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 6px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #45a049; }
        """)
        self.create_btn.clicked.connect(self._on_create_simple)
        layout.addWidget(self.create_btn)

        layout.addStretch()
        return tab

    def _create_script_tab(self) -> QWidget:
        """Create Python script tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(4)

        # Instructions
        info_label = QLabel(
            "Write Python code. Use 'df' to access data.\n"
            "Return result by assigning to 'result' variable."
        )
        info_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(info_label)

        # Signal name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self.script_name_edit = QLineEdit()
        self.script_name_edit.setPlaceholderText("new_signal_name")
        name_layout.addWidget(self.script_name_edit)
        layout.addLayout(name_layout)

        # Script editor
        layout.addWidget(QLabel("Python Script:"))
        self.script_edit = QTextEdit()
        self.script_edit.setFont(QFont("Consolas", 10))
        self.script_edit.setPlaceholderText(
            "# Example:\n"
            "# result = pd.to_datetime(df['timestamp'])\n"
            "# result = df['close'].rolling(20).mean()\n"
            "# result = df['high'] - df['low']\n"
        )
        self.script_edit.setMinimumHeight(120)
        layout.addWidget(self.script_edit)

        # Run button
        self.run_script_btn = QPushButton("Run Script & Create Signal")
        self.run_script_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 6px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #1976D2; }
        """)
        self.run_script_btn.clicked.connect(self._on_run_script)
        layout.addWidget(self.run_script_btn)

        layout.addStretch()
        return tab

    def set_columns(self, columns: List[str]):
        """Set available columns."""
        self._available_columns = columns.copy()
        self.columns_list.clear()
        for col in columns:
            self.columns_list.addItem(col)

    def set_data_callback(self, callback: Callable):
        """Set callback to get current DataFrame."""
        self._data_callback = callback

    def _on_template_selected(self, index: int):
        """Handle template selection."""
        expr = self.template_combo.currentData()
        if expr:
            self.expr_edit.setText(expr)

    def _on_column_double_clicked(self, item: QListWidgetItem):
        """Insert column name into expression."""
        col_name = item.text()
        # Insert at cursor or append
        current = self.expr_edit.text()
        cursor_pos = self.expr_edit.cursorPosition()

        # Use df['col'] syntax for safety
        col_ref = f"df['{col_name}']"
        new_text = current[:cursor_pos] + col_ref + current[cursor_pos:]
        self.expr_edit.setText(new_text)
        self.expr_edit.setFocus()

    def _on_create_simple(self):
        """Create signal from simple expression."""
        name = self.name_edit.text().strip()
        expr = self.expr_edit.text().strip()

        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a signal name.")
            return

        if not expr:
            QMessageBox.warning(self, "Warning", "Please enter an expression.")
            return

        # Validate name
        if not name.isidentifier():
            QMessageBox.warning(self, "Warning", "Invalid signal name. Use letters, numbers, and underscores.")
            return

        if name in self._available_columns:
            QMessageBox.warning(self, "Warning", f"Column '{name}' already exists.")
            return

        # Try to evaluate
        try:
            result = self._evaluate_expression(expr)
            if result is None:
                return

            self._add_calculated_signal(name, expr, result)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to evaluate expression:\n{e}")

    def _on_run_script(self):
        """Run Python script and create signal."""
        name = self.script_name_edit.text().strip()
        script = self.script_edit.toPlainText().strip()

        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a signal name.")
            return

        if not script:
            QMessageBox.warning(self, "Warning", "Please enter a script.")
            return

        if not name.isidentifier():
            QMessageBox.warning(self, "Warning", "Invalid signal name.")
            return

        if name in self._available_columns:
            QMessageBox.warning(self, "Warning", f"Column '{name}' already exists.")
            return

        try:
            result = self._execute_script(script)
            if result is None:
                QMessageBox.warning(self, "Warning", "Script must assign result to 'result' variable.")
                return

            self._add_calculated_signal(name, f"[script]\n{script}", result)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Script error:\n{e}")

    def _evaluate_expression(self, expr: str):
        """Evaluate a simple expression."""
        if not self._data_callback:
            QMessageBox.warning(self, "Warning", "No data loaded.")
            return None

        df = self._data_callback()
        if df is None or df.empty:
            QMessageBox.warning(self, "Warning", "No data available.")
            return None

        # Build evaluation context
        eval_globals = {
            '__builtins__': SAFE_BUILTINS,
            **SAFE_MODULES,
            'df': df,
        }

        # Add columns as variables for convenience (col_name style)
        for col in df.columns:
            safe_name = col.replace(' ', '_').replace('-', '_')
            eval_globals[safe_name] = df[col]

        result = eval(expr, eval_globals)
        return result

    def _execute_script(self, script: str):
        """Execute a Python script."""
        if not self._data_callback:
            QMessageBox.warning(self, "Warning", "No data loaded.")
            return None

        df = self._data_callback()
        if df is None or df.empty:
            QMessageBox.warning(self, "Warning", "No data available.")
            return None

        # Build execution context
        exec_globals = {
            '__builtins__': SAFE_BUILTINS,
            **SAFE_MODULES,
            'df': df.copy(),  # Copy to prevent modification
            'result': None,
        }

        exec(script, exec_globals)
        return exec_globals.get('result')

    def _add_calculated_signal(self, name: str, expr: str, result):
        """Add a calculated signal."""
        self._calculated_signals[name] = expr

        # Add to list widget
        item = QListWidgetItem(f"{name}")
        item.setData(Qt.ItemDataRole.UserRole, name)
        item.setToolTip(expr)
        self.signals_list.addItem(item)

        # Store result for retrieval
        self._results[name] = result

        # Update available columns
        if name not in self._available_columns:
            self._available_columns.append(name)
            self.columns_list.addItem(name)

        # Auto-save
        self._save_signals()

        self.signalCreated.emit(name)
        self.columnsChanged.emit()

        QMessageBox.information(self, "Success", f"Signal '{name}' created successfully!")

        # Clear inputs
        self.name_edit.clear()
        self.script_name_edit.clear()

    def _on_remove_signal(self):
        """Remove selected calculated signal."""
        item = self.signals_list.currentItem()
        if not item:
            return

        name = item.data(Qt.ItemDataRole.UserRole)
        if name in self._calculated_signals:
            del self._calculated_signals[name]

        if name in self._results:
            del self._results[name]

        if name in self._available_columns:
            self._available_columns.remove(name)

        # Remove from columns list
        for i in range(self.columns_list.count()):
            if self.columns_list.item(i).text() == name:
                self.columns_list.takeItem(i)
                break

        self.signals_list.takeItem(self.signals_list.row(item))

        # Auto-save
        self._save_signals()

        self.signalRemoved.emit(name)
        self.columnsChanged.emit()

    def get_calculated_columns(self) -> List[str]:
        """Get list of calculated column names."""
        return list(self._calculated_signals.keys())

    def get_calculated_data(self, name: str):
        """Get calculated data for a signal."""
        if hasattr(self, '_results'):
            return self._results.get(name)
        return None

    def apply_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all calculated signals to a DataFrame."""
        df = df.copy()

        # Re-evaluate all expressions with the new dataframe
        for name, expr in self._calculated_signals.items():
            try:
                if expr.startswith("[script]\n"):
                    # Script mode
                    script = expr[9:]  # Remove "[script]\n" prefix
                    result = self._execute_script_safe(script, df)
                else:
                    # Expression mode
                    result = self._evaluate_expression_safe(expr, df)

                if result is not None:
                    if isinstance(result, pd.Series):
                        if len(result) == len(df):
                            df[name] = result.values
                        else:
                            df[name] = result.reindex(df.index)
                    else:
                        df[name] = result
            except Exception:
                # Skip if evaluation fails (missing columns, etc.)
                pass

        return df

    def _evaluate_expression_safe(self, expr: str, df: pd.DataFrame):
        """Evaluate expression safely, returning None if columns are missing."""
        if df is None or df.empty:
            return None

        # Build evaluation context
        eval_globals = {
            '__builtins__': SAFE_BUILTINS,
            **SAFE_MODULES,
            'df': df,
        }

        # Add columns as variables for convenience
        for col in df.columns:
            safe_name = col.replace(' ', '_').replace('-', '_')
            eval_globals[safe_name] = df[col]

        try:
            result = eval(expr, eval_globals)
            return result
        except (KeyError, NameError):
            # Missing column - skip silently
            return None
        except Exception:
            return None

    def _execute_script_safe(self, script: str, df: pd.DataFrame):
        """Execute script safely, returning None if columns are missing."""
        if df is None or df.empty:
            return None

        exec_globals = {
            '__builtins__': SAFE_BUILTINS,
            **SAFE_MODULES,
            'df': df.copy(),
            'result': None,
        }

        try:
            exec(script, exec_globals)
            return exec_globals.get('result')
        except (KeyError, NameError):
            # Missing column - skip silently
            return None
        except Exception:
            return None

    def _save_signals(self):
        """Save calculated signals to config file."""
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)

            data = {
                'signals': self._calculated_signals
            }

            with open(CALC_SIGNALS_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save calculated signals: {e}")

    def _load_saved_signals(self):
        """Load saved calculated signals from config file."""
        try:
            if not CALC_SIGNALS_FILE.exists():
                return

            with open(CALC_SIGNALS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)

            signals = data.get('signals', {})

            for name, expr in signals.items():
                self._calculated_signals[name] = expr

                # Add to list widget
                item = QListWidgetItem(f"{name}")
                item.setData(Qt.ItemDataRole.UserRole, name)
                item.setToolTip(expr)
                self.signals_list.addItem(item)

                # Add to available columns
                self._available_columns.append(name)

            if signals:
                self.columnsChanged.emit()

        except Exception as e:
            print(f"Warning: Failed to load calculated signals: {e}")

    def refresh_all_signals(self):
        """Re-evaluate all signals with current data."""
        if not self._data_callback:
            return

        df = self._data_callback()
        if df is None:
            return

        self._results.clear()

        for name, expr in self._calculated_signals.items():
            try:
                if expr.startswith("[script]\n"):
                    script = expr[9:]
                    result = self._execute_script_safe(script, df)
                else:
                    result = self._evaluate_expression_safe(expr, df)

                if result is not None:
                    self._results[name] = result
            except Exception:
                pass
