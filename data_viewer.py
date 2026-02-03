"""
Main data viewer window integrating all components.
"""

from typing import Optional, List, Dict, Any
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QTabWidget, QLabel, QStatusBar,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QScrollArea, QGroupBox, QTextEdit,
    QProgressDialog, QApplication
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QShortcut, QKeySequence, QKeyEvent
from PyQt6.QtCore import QEvent

from data_loader import DataLoader, MultiFileLoader, INDEX_COLUMN, FILE_INDEX_COLUMN
from plot_canvas import PlotWidget
from sampling import apply_sampling
from widgets import FilePanel, XAxisSelector, YAxisSelectorGrid, PlotControls, FilterPanel, StylePanel, ExportPanel, CalculatedSignalPanel


class DataViewer(QMainWindow):
    """
    Main window for the data viewer application.
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("PyQt Data Viewer")
        self.setMinimumSize(1200, 800)

        # Data management
        self.file_loader = MultiFileLoader()
        self._current_loader: Optional[DataLoader] = None
        self._all_columns: List[str] = []  # Union of all columns from all files

        # Setup UI
        self._setup_ui()
        self._connect_signals()

        # Initial state
        self._update_ui_state()

    def _setup_ui(self):
        """Setup the main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel (controls)
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel (plot)
        self.plot_widget = PlotWidget()
        splitter.addWidget(self.plot_widget)

        # Set initial splitter sizes (30% left, 70% right)
        splitter.setSizes([350, 850])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _create_left_panel(self) -> QWidget:
        """Create the left control panel with tabs."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # File panel (always visible)
        self.file_panel = FilePanel()
        layout.addWidget(self.file_panel)

        # Tab widget
        self.tab_widget = QTabWidget()

        # Plot tab
        plot_tab = self._create_plot_tab()
        self.tab_widget.addTab(plot_tab, "Plot")

        # Statistics tab
        stats_tab = self._create_statistics_tab()
        self.tab_widget.addTab(stats_tab, "Statistics")

        # Filter tab
        filter_tab = self._create_filter_tab()
        self.tab_widget.addTab(filter_tab, "Filter")

        # Style tab
        style_tab = self._create_style_tab()
        self.tab_widget.addTab(style_tab, "Style")

        # Calculated signals tab
        calc_tab = self._create_calculated_tab()
        self.tab_widget.addTab(calc_tab, "Calc")

        # Export tab
        export_tab = self._create_export_tab()
        self.tab_widget.addTab(export_tab, "Export")

        layout.addWidget(self.tab_widget)

        return panel

    def _create_plot_tab(self) -> QWidget:
        """Create the Plot tab content."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # X axis selector
        self.x_selector = XAxisSelector()
        layout.addWidget(self.x_selector)

        # Y axis selector grid
        self.y_selector_grid = YAxisSelectorGrid()
        layout.addWidget(self.y_selector_grid)

        # Plot controls
        self.plot_controls = PlotControls()
        layout.addWidget(self.plot_controls)

        layout.addStretch()

        return tab

    def _create_statistics_tab(self) -> QWidget:
        """Create the Statistics tab content."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Statistics table
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(6)
        self.stats_table.setHorizontalHeaderLabels(
            ["Column", "Count", "Mean", "Std", "Min", "Max"]
        )
        self.stats_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self.stats_table)

        # Calculate button
        from PyQt6.QtWidgets import QPushButton
        self.calc_stats_button = QPushButton("Calculate Statistics")
        self.calc_stats_button.clicked.connect(self._on_calculate_stats)
        layout.addWidget(self.calc_stats_button)

        return tab

    def _create_filter_tab(self) -> QWidget:
        """Create the Filter tab content."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Filter panel
        self.filter_panel = FilterPanel()
        self.filter_panel.applyRequested.connect(self._on_plot_requested)
        layout.addWidget(self.filter_panel)

        return tab

    def _create_style_tab(self) -> QWidget:
        """Create the Style tab content."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Style panel
        self.style_panel = StylePanel()
        self.style_panel.styleChanged.connect(self._on_auto_update_check)
        layout.addWidget(self.style_panel)

        return tab

    def _create_calculated_tab(self) -> QWidget:
        """Create the Calculated Signals tab content."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Calculated signal panel
        self.calc_panel = CalculatedSignalPanel()
        self.calc_panel.set_data_callback(self._get_current_dataframe)
        self.calc_panel.columnsChanged.connect(self._on_calculated_columns_changed)
        layout.addWidget(self.calc_panel)

        return tab

    def _create_export_tab(self) -> QWidget:
        """Create the Export tab content."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Export panel
        self.export_panel = ExportPanel()
        self.export_panel.set_data_callback(self._get_export_data)
        self.export_panel.set_plot_callback(self._get_export_figure)
        layout.addWidget(self.export_panel)

        return tab

    def _get_export_data(self):
        """Get data for export."""
        if not self._current_loader:
            return None

        try:
            # Get all columns or selected columns
            if self.export_panel.selected_columns_check.isChecked():
                # Get selected X and Y columns
                x_sig = self.x_selector.get_selected()
                y_selections = self.y_selector_grid.get_all_selected()
                columns = set()
                if x_sig:
                    columns.add(x_sig)
                for y_list in y_selections:
                    columns.update(y_list)
                if not columns:
                    return None
                columns = list(columns)
            else:
                columns = self.file_loader.get_all_columns()

            # Load merged data from all files
            df = self.file_loader.load_columns_merged(columns)

            # Apply filter if checked
            if self.export_panel.filtered_only_check.isChecked():
                filter_conditions = self.filter_panel.get_filter_conditions()
                if filter_conditions:
                    df = self.filter_panel.apply_filter(df)

            return df

        except Exception:
            return None

    def _get_export_figure(self):
        """Get plot figure for export."""
        return self.plot_widget.canvas.fig

    def _get_current_dataframe(self):
        """Get current DataFrame for calculated signals."""
        if not self._current_loader:
            return None
        try:
            # Load all columns
            columns = self._current_loader.get_columns()
            df = self.file_loader.load_columns_merged(columns)
            return df
        except Exception:
            return None

    def _on_calculated_columns_changed(self):
        """Handle calculated columns change."""
        self._update_merged_columns()

    def _refresh_calculated_signals(self):
        """Refresh calculated signals with current data."""
        if hasattr(self, 'calc_panel'):
            self.calc_panel.refresh_all_signals()

    def _connect_signals(self):
        """Connect signals to slots."""
        # File panel signals
        self.file_panel.fileAdded.connect(self._on_file_added)
        self.file_panel.fileRemoved.connect(self._on_file_removed)
        self.file_panel.fileSelected.connect(self._on_file_selected)

        # Plot controls
        self.plot_controls.plotRequested.connect(self._on_plot_requested)
        self.plot_controls.settingsChanged.connect(self._on_settings_changed)

        # Share X checkbox specifically
        self.plot_controls.share_x_check.stateChanged.connect(self._on_share_x_changed)

        # Axis selectors - connect for auto-update
        self.x_selector.selectionChanged.connect(self._on_auto_update_check)
        self.x_selector.selectionChangedMulti.connect(lambda idx, sig: self._on_auto_update_check())
        self.y_selector_grid.selectionChanged.connect(lambda idx, sel: self._on_auto_update_check())

        # Global Enter shortcut for plotting
        self.enter_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Return), self)
        self.enter_shortcut.activated.connect(self._on_plot_requested)

        # Also connect Ctrl+Enter for plotting
        self.ctrl_enter_shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)
        self.ctrl_enter_shortcut.activated.connect(self._on_plot_requested)

    def _on_share_x_changed(self, state: int):
        """Handle Share X checkbox change."""
        share_x = self.plot_controls.is_share_x()
        self.x_selector.set_share_x(share_x)
        self._on_auto_update_check()

    def _on_settings_changed(self):
        """Handle settings change (excluding Share X which has its own handler)."""
        self._on_auto_update_check()

    def _on_auto_update_check(self):
        """Check if auto-update is enabled and trigger plot if so."""
        if self.plot_controls.is_auto_update() and self._current_loader is not None:
            # Use timer to debounce rapid changes
            if not hasattr(self, '_auto_update_timer'):
                self._auto_update_timer = QTimer(self)
                self._auto_update_timer.setSingleShot(True)
                self._auto_update_timer.timeout.connect(self._on_plot_requested)

            # Reset timer on each change (debounce)
            self._auto_update_timer.stop()
            self._auto_update_timer.start(300)  # 300ms debounce

    def _on_file_added(self, file_path: str):
        """Handle file added event."""
        # Show progress for file loading
        progress = QProgressDialog("Loading file...", None, 0, 0, self)
        progress.setWindowTitle("Loading")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(300)
        progress.show()
        QApplication.processEvents()

        try:
            key = self.file_loader.add_file(file_path)
            loader = self.file_loader.get_loader(key)
            if loader:
                progress.setLabelText("Reading column information...")
                QApplication.processEvents()

                # Set as current loader
                self._current_loader = loader
                # Update merged columns from all files
                self._update_merged_columns()
                # Refresh calculated signals with new data
                self._refresh_calculated_signals()
                self.status_bar.showMessage(f"Loaded: {loader.file_name}")

            progress.close()
        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{e}")
            self.file_panel.remove_file(file_path)

    def _on_file_removed(self, file_path: str):
        """Handle file removed event."""
        self.file_loader.remove_file(file_path)

        # Update current loader if it was removed
        loaders = self.file_loader.get_all_loaders()
        if not loaders:
            # No files left
            self._current_loader = None
        elif self._current_loader and self._current_loader.file_path not in [l.file_path for l in loaders.values()]:
            # Current loader was removed, select first available
            self._current_loader = list(loaders.values())[0] if loaders else None

        # Update merged columns after removal
        self._update_merged_columns()
        self.status_bar.showMessage("File removed")

    def _on_file_selected(self, file_path: str):
        """Handle file selection change."""
        loader = self.file_loader.get_loader(file_path)
        if loader:
            self._current_loader = loader
            self._update_ui_state()

    def _update_merged_columns(self):
        """Update column list by merging columns from all loaded files."""
        # Save current selections
        prev_x = self.x_selector.get_selected()
        prev_y_selections = self.y_selector_grid.get_all_selected()

        # Collect all unique columns from all files
        all_columns_set = set()
        loaders = self.file_loader.get_all_loaders()

        for loader in loaders.values():
            try:
                columns = loader.get_columns()
                all_columns_set.update(columns)
            except Exception:
                pass

        # If no files loaded, clear all columns
        if not loaders:
            self._all_columns = []
            self._x_columns = []
            self._y_columns = []
            self.x_selector.set_signals([])
            self.y_selector_grid.set_signals([])
            self.filter_panel.set_columns([])
            self._update_ui_state()
            return

        # Sort columns, separating virtual columns
        virtual_cols = {INDEX_COLUMN, FILE_INDEX_COLUMN}
        sorted_columns = sorted(all_columns_set - virtual_cols)

        # Add calculated columns
        if hasattr(self, 'calc_panel'):
            calc_columns = self.calc_panel.get_calculated_columns()
            sorted_columns = sorted_columns + calc_columns
            # Update calc panel with available columns (excluding calc columns to avoid recursion)
            base_columns = sorted(all_columns_set - virtual_cols)
            self.calc_panel.set_columns(list(virtual_cols) + base_columns)

        # Virtual columns list (in order)
        virtual_list = []
        if INDEX_COLUMN in all_columns_set:
            virtual_list.append(INDEX_COLUMN)
        if FILE_INDEX_COLUMN in all_columns_set:
            virtual_list.append(FILE_INDEX_COLUMN)

        # For X axis: virtual columns at TOP
        x_columns = virtual_list + sorted_columns

        # For Y axis: virtual columns at BOTTOM
        y_columns = sorted_columns + virtual_list

        self._all_columns = sorted_columns  # Store without virtual for filter
        self._x_columns = x_columns
        self._y_columns = y_columns

        # Update selectors with appropriate column order
        self.x_selector.set_signals(x_columns)
        self.y_selector_grid.set_signals(y_columns)
        self.filter_panel.set_columns(x_columns)  # Filter can use all columns

        # Restore X selection if still valid
        if prev_x and prev_x in x_columns:
            self.x_selector.select_signal(prev_x)

        # Restore Y selections if still valid
        for i, prev_y_list in enumerate(prev_y_selections):
            valid_signals = [s for s in prev_y_list if s in y_columns]
            if valid_signals and i < len(self.y_selector_grid.selectors):
                self.y_selector_grid.selectors[i].select_signals(valid_signals)

        self._update_ui_state()

    def _update_ui_state(self):
        """Update UI enabled states based on current data."""
        has_file = self._current_loader is not None
        self.plot_controls.set_plot_enabled(has_file)
        self.export_panel.set_enabled(has_file, has_file)

    def _on_plot_requested(self):
        """Handle plot button click."""
        if not self._current_loader:
            return

        # Get settings
        share_x = self.plot_controls.is_share_x()

        # Get X selections based on mode
        if share_x:
            x_signal = self.x_selector.get_selected()
            x_signals = [x_signal] * 4  # Same X for all subplots
            if not x_signal:
                QMessageBox.warning(self, "Warning", "Please select an X axis signal.")
                return
        else:
            x_signals = self.x_selector.get_selected_multi()
            # Check that at least one X is selected for subplots with Y selections
            y_selections_temp = self.y_selector_grid.get_all_selected()
            for i, (x_sig, y_list) in enumerate(zip(x_signals, y_selections_temp)):
                if y_list and not x_sig:
                    QMessageBox.warning(self, "Warning", f"Please select X axis for Plot {i+1}.")
                    return

        y_selections = self.y_selector_grid.get_all_selected()

        # Check if any Y signals are selected
        has_y_selection = any(y_list for y_list in y_selections)
        if not has_y_selection:
            QMessageBox.warning(self, "Warning", "Please select at least one Y axis signal.")
            return

        # Collect all unique X and Y signals
        all_x_signals = set(x for x in x_signals if x)
        all_y_signals = set()
        for y_list in y_selections:
            all_y_signals.update(y_list)

        # Create progress dialog
        progress = QProgressDialog("Loading data...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Processing")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(500)  # Show only if takes > 500ms
        progress.setValue(0)

        try:
            # Load data
            self.status_bar.showMessage("Loading data...")
            progress.setLabelText("Loading columns...")
            QApplication.processEvents()

            # Use set to avoid duplicate columns
            # Also include columns needed for filtering
            filter_conditions = self.filter_panel.get_filter_conditions()
            filter_columns = {c["column"] for c in filter_conditions}

            # Check if any selected signals are calculated signals
            calc_columns = set()
            if hasattr(self, 'calc_panel'):
                calc_columns = set(self.calc_panel.get_calculated_columns())

            selected_calc_signals = (all_x_signals | all_y_signals) & calc_columns

            if selected_calc_signals:
                # If calculated signals are selected, load all available columns
                # because we need base columns for the calculations
                columns_to_load = self.file_loader.get_all_columns()
            else:
                columns_to_load = list(all_x_signals | all_y_signals | filter_columns)

            # Load merged data from all files (handles different columns across files)
            df = self.file_loader.load_columns_merged(columns_to_load)

            # Apply calculated signals
            if hasattr(self, 'calc_panel') and selected_calc_signals:
                df = self.calc_panel.apply_to_dataframe(df)

            progress.setValue(30)
            QApplication.processEvents()

            if progress.wasCanceled():
                self.status_bar.showMessage("Cancelled")
                return

            # Apply filters
            original_count = len(df)
            if filter_conditions:
                progress.setLabelText("Applying filters...")
                QApplication.processEvents()
                df = self.filter_panel.apply_filter(df)
                filtered_count = len(df)
                self.filter_panel.set_status(
                    f"Filtered: {filtered_count:,} / {original_count:,} rows"
                )
            else:
                filtered_count = original_count
                self.filter_panel.set_status("No filters applied")

            progress.setValue(50)
            QApplication.processEvents()

            if progress.wasCanceled():
                self.status_bar.showMessage("Cancelled")
                return

            if len(df) == 0:
                QMessageBox.warning(self, "Warning", "No data after filtering.")
                progress.close()
                return

            # Get plot settings
            sampling_mode = self.plot_controls.get_sampling_mode()
            sample_count = self.plot_controls.get_sample_count()
            show_legend = self.plot_controls.is_show_legend()

            progress.setLabelText("Processing data...")
            progress.setValue(60)
            QApplication.processEvents()

            # Prepare data for each subplot
            y_data_per_subplot = []
            x_data_per_subplot = []
            x_labels = []
            first_sampled_count = 0  # Track first subplot's sampled count

            for i, (x_sig, y_list) in enumerate(zip(x_signals, y_selections)):
                if not y_list or not x_sig:
                    y_data_per_subplot.append([])
                    x_data_per_subplot.append(None)
                    x_labels.append("")
                    continue

                x_data = df[x_sig].values
                subplot_data = []

                for y_signal in y_list:
                    y_data = df[y_signal].values

                    # Apply sampling
                    x_sampled, y_sampled = apply_sampling(
                        x_data, y_data, sampling_mode, sample_count
                    )
                    subplot_data.append((y_signal, y_sampled))

                # Sample X data
                x_plot = apply_sampling(x_data, x_data, sampling_mode, sample_count)[0]

                # Track first valid subplot's sampled count
                if first_sampled_count == 0:
                    first_sampled_count = len(x_plot)

                y_data_per_subplot.append(subplot_data)
                x_data_per_subplot.append(x_plot)
                x_labels.append(x_sig)
                QApplication.processEvents()

            progress.setValue(80)

            if progress.wasCanceled():
                self.status_bar.showMessage("Cancelled")
                return

            progress.setLabelText("Drawing plot...")
            progress.setValue(90)
            QApplication.processEvents()

            # Apply style settings
            style = self.style_panel.get_style_dict()
            self.style_panel.apply_theme()
            self.plot_widget.set_style(
                linewidth=style['linewidth'],
                linestyle=style['linestyle'],
                marker=style['marker'],
                markersize=style['markersize']
            )

            # Apply diff setting
            show_diff = self.plot_controls.is_show_diff()
            self.plot_widget.set_show_diff(show_diff)

            # Plot
            self.plot_widget.plot_all_multi(
                x_data_per_subplot=x_data_per_subplot,
                y_data_per_subplot=y_data_per_subplot,
                x_labels=x_labels,
                title=self._current_loader.file_name,
                share_x=share_x,
                show_legend=show_legend
            )

            progress.setValue(100)

            # Status message - show filtered row count and sampled count
            filter_info = f", filtered from {original_count:,}" if filter_conditions else ""
            self.status_bar.showMessage(
                f"Plotted {first_sampled_count:,} points (from {filtered_count:,} rows{filter_info}) "
                f"using {sampling_mode.upper()} sampling"
            )

        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "Error", f"Failed to plot data:\n{e}")
            self.status_bar.showMessage("Plot failed")

    def _on_calculate_stats(self):
        """Calculate and display statistics for selected file."""
        if not self._current_loader:
            QMessageBox.warning(self, "Warning", "Please load a file first.")
            return

        try:
            self.status_bar.showMessage("Calculating statistics...")

            stats = self._current_loader.get_statistics()

            self.stats_table.setRowCount(len(stats))

            for row, (col_name, col_stats) in enumerate(stats.items()):
                self.stats_table.setItem(row, 0, QTableWidgetItem(col_name))

                if 'mean' in col_stats:
                    # Numeric column
                    self.stats_table.setItem(
                        row, 1, QTableWidgetItem(str(col_stats.get('count', '')))
                    )
                    mean_val = col_stats.get('mean')
                    self.stats_table.setItem(
                        row, 2, QTableWidgetItem(f"{mean_val:.4f}" if mean_val is not None else "")
                    )
                    std_val = col_stats.get('std')
                    self.stats_table.setItem(
                        row, 3, QTableWidgetItem(f"{std_val:.4f}" if std_val is not None else "")
                    )
                    min_val = col_stats.get('min')
                    self.stats_table.setItem(
                        row, 4, QTableWidgetItem(f"{min_val:.4f}" if min_val is not None else "")
                    )
                    max_val = col_stats.get('max')
                    self.stats_table.setItem(
                        row, 5, QTableWidgetItem(f"{max_val:.4f}" if max_val is not None else "")
                    )
                else:
                    # Non-numeric column
                    self.stats_table.setItem(
                        row, 1, QTableWidgetItem(str(col_stats.get('count', '')))
                    )
                    self.stats_table.setItem(
                        row, 2, QTableWidgetItem(f"(unique: {col_stats.get('unique', '')})")
                    )

            self.status_bar.showMessage("Statistics calculated")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to calculate statistics:\n{e}")
            self.status_bar.showMessage("Statistics calculation failed")

    def keyPressEvent(self, event: QKeyEvent):
        """
        Handle key press events globally.
        Auto-focus Y signals filter when typing printable characters.
        """
        # Check if it's a printable character and not a modifier key
        text = event.text()
        if text and text.isprintable() and not event.modifiers() & (
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier
        ):
            # Check if focus is not already on an input widget
            focused = QApplication.focusWidget()
            from PyQt6.QtWidgets import QLineEdit, QTextEdit, QSpinBox, QComboBox

            if not isinstance(focused, (QLineEdit, QTextEdit, QSpinBox)):
                # Focus the Y signals filter and append the typed character
                search_edit = self.y_selector_grid.search_edit
                search_edit.setFocus()
                search_edit.setText(search_edit.text() + text)
                return

        super().keyPressEvent(event)
