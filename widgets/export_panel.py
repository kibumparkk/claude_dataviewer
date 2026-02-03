"""
Export panel for exporting data and plots.
"""

from typing import Optional, List
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGroupBox, QFileDialog, QMessageBox,
    QCheckBox, QComboBox
)
from PyQt6.QtCore import pyqtSignal
import pandas as pd


class ExportPanel(QWidget):
    """
    Panel for exporting data and plots.
    """

    exportDataRequested = pyqtSignal(str)  # Emits file path for data export
    exportPlotRequested = pyqtSignal(str)  # Emits file path for plot export

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._data_callback = None
        self._plot_callback = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Data export group
        data_group = QGroupBox("Export Data")
        data_layout = QVBoxLayout(data_group)

        # Format selection
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["CSV", "Excel (.xlsx)", "Parquet"])
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()
        data_layout.addLayout(format_layout)

        # Options
        self.include_index_check = QCheckBox("Include row index")
        self.include_index_check.setChecked(False)
        data_layout.addWidget(self.include_index_check)

        self.filtered_only_check = QCheckBox("Export filtered data only")
        self.filtered_only_check.setChecked(True)
        data_layout.addWidget(self.filtered_only_check)

        self.selected_columns_check = QCheckBox("Export selected columns only")
        self.selected_columns_check.setChecked(False)
        data_layout.addWidget(self.selected_columns_check)

        # Export data button
        self.export_data_btn = QPushButton("Export Data...")
        self.export_data_btn.clicked.connect(self._on_export_data)
        self.export_data_btn.setStyleSheet("""
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
        data_layout.addWidget(self.export_data_btn)

        layout.addWidget(data_group)

        # Plot export group
        plot_group = QGroupBox("Export Plot")
        plot_layout = QVBoxLayout(plot_group)

        # Image format selection
        img_format_layout = QHBoxLayout()
        img_format_layout.addWidget(QLabel("Format:"))
        self.img_format_combo = QComboBox()
        self.img_format_combo.addItems(["PNG", "SVG", "PDF", "JPG"])
        img_format_layout.addWidget(self.img_format_combo)
        img_format_layout.addStretch()
        plot_layout.addLayout(img_format_layout)

        # DPI selection
        dpi_layout = QHBoxLayout()
        dpi_layout.addWidget(QLabel("DPI:"))
        self.dpi_combo = QComboBox()
        self.dpi_combo.addItems(["72", "100", "150", "200", "300"])
        self.dpi_combo.setCurrentText("150")
        dpi_layout.addWidget(self.dpi_combo)
        dpi_layout.addStretch()
        plot_layout.addLayout(dpi_layout)

        # Export plot button
        self.export_plot_btn = QPushButton("Export Plot...")
        self.export_plot_btn.clicked.connect(self._on_export_plot)
        self.export_plot_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        plot_layout.addWidget(self.export_plot_btn)

        layout.addWidget(plot_group)
        layout.addStretch()

    def set_data_callback(self, callback):
        """Set callback function to get data for export."""
        self._data_callback = callback

    def set_plot_callback(self, callback):
        """Set callback function to get plot figure for export."""
        self._plot_callback = callback

    def _on_export_data(self):
        """Handle export data button click."""
        if not self._data_callback:
            QMessageBox.warning(self, "Warning", "No data available for export.")
            return

        # Get data from callback
        data = self._data_callback()
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            QMessageBox.warning(self, "Warning", "No data available for export.")
            return

        # Get file format and extension
        format_text = self.format_combo.currentText()
        if format_text == "CSV":
            ext = "csv"
            filter_str = "CSV Files (*.csv)"
        elif format_text == "Excel (.xlsx)":
            ext = "xlsx"
            filter_str = "Excel Files (*.xlsx)"
        else:  # Parquet
            ext = "parquet"
            filter_str = "Parquet Files (*.parquet)"

        # Get save path
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Data",
            f"data_export.{ext}",
            filter_str
        )

        if not file_path:
            return

        try:
            include_index = self.include_index_check.isChecked()

            if ext == "csv":
                data.to_csv(file_path, index=include_index)
            elif ext == "xlsx":
                data.to_excel(file_path, index=include_index, engine='openpyxl')
            else:  # parquet
                data.to_parquet(file_path, index=include_index)

            QMessageBox.information(
                self, "Success",
                f"Data exported successfully to:\n{file_path}\n\n"
                f"Rows: {len(data):,}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export data:\n{e}")

    def _on_export_plot(self):
        """Handle export plot button click."""
        if not self._plot_callback:
            QMessageBox.warning(self, "Warning", "No plot available for export.")
            return

        # Get figure from callback
        fig = self._plot_callback()
        if fig is None:
            QMessageBox.warning(self, "Warning", "No plot available for export.")
            return

        # Get file format and extension
        format_text = self.img_format_combo.currentText().lower()
        filter_map = {
            "png": "PNG Images (*.png)",
            "svg": "SVG Images (*.svg)",
            "pdf": "PDF Documents (*.pdf)",
            "jpg": "JPEG Images (*.jpg)",
        }

        # Get save path
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Plot",
            f"plot_export.{format_text}",
            filter_map.get(format_text, "All Files (*)")
        )

        if not file_path:
            return

        try:
            dpi = int(self.dpi_combo.currentText())
            fig.savefig(file_path, format=format_text, dpi=dpi, bbox_inches='tight')

            QMessageBox.information(
                self, "Success",
                f"Plot exported successfully to:\n{file_path}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export plot:\n{e}")

    def set_enabled(self, data_enabled: bool, plot_enabled: bool):
        """Enable or disable export buttons."""
        self.export_data_btn.setEnabled(data_enabled)
        self.export_plot_btn.setEnabled(plot_enabled)
