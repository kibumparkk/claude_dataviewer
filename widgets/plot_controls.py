"""
Plot control widgets for sampling mode, options, and plot button.
"""

from typing import Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QSpinBox, QCheckBox, QPushButton,
    QGroupBox, QFormLayout
)
from PyQt6.QtCore import pyqtSignal


class PlotControls(QWidget):
    """
    Control widget for plot options including sampling mode, share X, and legend.
    """

    plotRequested = pyqtSignal()  # Emitted when Plot button is clicked
    settingsChanged = pyqtSignal()  # Emitted when any setting changes

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Options group
        group = QGroupBox("Plot Options")
        group_layout = QFormLayout(group)

        # Sampling mode
        self.sampling_combo = QComboBox()
        self.sampling_combo.addItems(["LTTB", "Simple", "Full"])
        self.sampling_combo.setCurrentText("LTTB")
        self.sampling_combo.currentTextChanged.connect(self._on_sampling_changed)
        group_layout.addRow("Sampling:", self.sampling_combo)

        # Sample count
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(100, 100000)
        self.sample_spin.setValue(10000)
        self.sample_spin.setSingleStep(1000)
        self.sample_spin.valueChanged.connect(self._on_settings_changed)
        group_layout.addRow("Sample Count:", self.sample_spin)

        # Share X checkbox
        self.share_x_check = QCheckBox("Share X Axis")
        self.share_x_check.setChecked(True)
        self.share_x_check.stateChanged.connect(self._on_settings_changed)
        group_layout.addRow(self.share_x_check)

        # Legend checkbox
        self.legend_check = QCheckBox("Show Legend")
        self.legend_check.setChecked(True)
        self.legend_check.stateChanged.connect(self._on_settings_changed)
        group_layout.addRow(self.legend_check)

        # Auto update checkbox
        self.auto_update_check = QCheckBox("Auto Update")
        self.auto_update_check.setChecked(False)
        self.auto_update_check.setToolTip("Automatically redraw when selections change")
        group_layout.addRow(self.auto_update_check)

        # Show diff checkbox
        self.show_diff_check = QCheckBox("Show Diff")
        self.show_diff_check.setChecked(False)
        self.show_diff_check.setToolTip("Show difference between first two Y signals on twin axis (gray)")
        self.show_diff_check.stateChanged.connect(self._on_settings_changed)
        group_layout.addRow(self.show_diff_check)

        layout.addWidget(group)

        # Plot button
        self.plot_button = QPushButton("Plot")
        self.plot_button.setMinimumHeight(40)
        self.plot_button.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                font-weight: bold;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.plot_button.clicked.connect(self._on_plot_clicked)
        layout.addWidget(self.plot_button)

        # Update sample spin enabled state
        self._update_sample_spin_state()

    def _on_sampling_changed(self, mode: str):
        """Handle sampling mode change."""
        self._update_sample_spin_state()
        self._on_settings_changed()

    def _update_sample_spin_state(self):
        """Update sample count spin box enabled state."""
        mode = self.sampling_combo.currentText()
        self.sample_spin.setEnabled(mode.lower() != "full")

    def _on_settings_changed(self):
        """Handle any settings change."""
        self.settingsChanged.emit()

    def _on_plot_clicked(self):
        """Handle plot button click."""
        self.plotRequested.emit()

    def get_sampling_mode(self) -> str:
        """Get current sampling mode."""
        return self.sampling_combo.currentText().lower()

    def get_sample_count(self) -> int:
        """Get current sample count."""
        return self.sample_spin.value()

    def is_share_x(self) -> bool:
        """Check if share X axis is enabled."""
        return self.share_x_check.isChecked()

    def is_show_legend(self) -> bool:
        """Check if show legend is enabled."""
        return self.legend_check.isChecked()

    def set_plot_enabled(self, enabled: bool):
        """Enable or disable the plot button."""
        self.plot_button.setEnabled(enabled)

    def is_auto_update(self) -> bool:
        """Check if auto update is enabled."""
        return self.auto_update_check.isChecked()

    def is_show_diff(self) -> bool:
        """Check if show diff is enabled."""
        return self.show_diff_check.isChecked()
