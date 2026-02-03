"""
PyQt Data Viewer - Application entry point.

A PyQt6-based data analysis viewer for large datasets with:
- Support for CSV, XLSX, Parquet, Feather, ZIP, and GZ files
- Lazy loading for memory efficiency
- LTTB and simple downsampling algorithms
- 2x2 subplot layout with shared X axis option
"""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from data_viewer import DataViewer


def main():
    """Main entry point for the application."""
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("PyQt Data Viewer")
    app.setOrganizationName("DataViewer")

    # Set application style
    app.setStyle("Fusion")

    # Create and show main window
    viewer = DataViewer()
    viewer.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
