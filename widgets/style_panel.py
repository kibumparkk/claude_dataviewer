"""
Style panel for plot styling options.
"""

from typing import Optional, Dict, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox,
    QFormLayout
)
from PyQt6.QtCore import pyqtSignal
import matplotlib.pyplot as plt


class StylePanel(QWidget):
    """
    Panel for plot style settings.
    """

    styleChanged = pyqtSignal()  # Emitted when any style setting changes

    # Available matplotlib styles
    STYLES = [
        'default', 'bmh', 'classic', 'dark_background', 'fast',
        'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-v0_8',
        'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark',
        'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted',
        'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel',
        'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks',
        'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'Solarize_Light2',
        'tableau-colorblind10'
    ]

    # Marker types
    MARKERS = [
        ('o', 'Circle'),
        ('s', 'Square'),
        ('^', 'Triangle Up'),
        ('v', 'Triangle Down'),
        ('D', 'Diamond'),
        ('*', 'Star'),
        ('+', 'Plus'),
        ('x', 'Cross'),
        ('.', 'Point'),
        ('', 'None'),
    ]

    # Line styles
    LINE_STYLES = [
        (':', 'Dotted'),
        ('-', 'Solid'),
        ('--', 'Dashed'),
        ('-.', 'Dash-Dot'),
        ('', 'None'),
    ]

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Style group
        group = QGroupBox("Plot Style")
        group_layout = QFormLayout(group)

        # Theme selector
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(self.STYLES)
        self.theme_combo.setCurrentText('bmh')
        self.theme_combo.currentTextChanged.connect(self._on_style_changed)
        group_layout.addRow("Theme:", self.theme_combo)

        # Line width
        self.linewidth_spin = QDoubleSpinBox()
        self.linewidth_spin.setRange(0.1, 10.0)
        self.linewidth_spin.setValue(0.8)
        self.linewidth_spin.setSingleStep(0.1)
        self.linewidth_spin.valueChanged.connect(self._on_style_changed)
        group_layout.addRow("Line Width:", self.linewidth_spin)

        # Line style
        self.linestyle_combo = QComboBox()
        for style, name in self.LINE_STYLES:
            self.linestyle_combo.addItem(name, style)
        self.linestyle_combo.setCurrentIndex(0)  # Dotted by default
        self.linestyle_combo.currentIndexChanged.connect(self._on_style_changed)
        group_layout.addRow("Line Style:", self.linestyle_combo)

        # Marker type
        self.marker_combo = QComboBox()
        for marker, name in self.MARKERS:
            self.marker_combo.addItem(name, marker)
        self.marker_combo.setCurrentIndex(0)  # Circle by default
        self.marker_combo.currentIndexChanged.connect(self._on_style_changed)
        group_layout.addRow("Marker Type:", self.marker_combo)

        # Marker size
        self.markersize_spin = QDoubleSpinBox()
        self.markersize_spin.setRange(0, 20.0)
        self.markersize_spin.setValue(3.0)
        self.markersize_spin.setSingleStep(0.5)
        self.markersize_spin.valueChanged.connect(self._on_style_changed)
        group_layout.addRow("Marker Size:", self.markersize_spin)

        layout.addWidget(group)
        layout.addStretch()

    def _on_style_changed(self):
        """Handle style change."""
        self.styleChanged.emit()

    def get_theme(self) -> str:
        """Get selected theme."""
        return self.theme_combo.currentText()

    def get_linewidth(self) -> float:
        """Get line width."""
        return self.linewidth_spin.value()

    def get_linestyle(self) -> str:
        """Get line style."""
        return self.linestyle_combo.currentData()

    def get_marker(self) -> str:
        """Get marker type."""
        return self.marker_combo.currentData()

    def get_markersize(self) -> float:
        """Get marker size."""
        return self.markersize_spin.value()

    def get_style_dict(self) -> Dict[str, Any]:
        """Get all style settings as a dictionary."""
        return {
            'theme': self.get_theme(),
            'linewidth': self.get_linewidth(),
            'linestyle': self.get_linestyle(),
            'marker': self.get_marker(),
            'markersize': self.get_markersize(),
        }

    def apply_theme(self):
        """Apply the selected theme to matplotlib."""
        theme = self.get_theme()
        try:
            # Reset to default first to clear previous style
            plt.rcdefaults()
            plt.style.use(theme)
        except Exception:
            plt.rcdefaults()
            plt.style.use('default')
