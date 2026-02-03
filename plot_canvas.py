"""
Matplotlib canvas for PyQt6 with 2x2 subplot layout.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.dates as mdates

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy, QHBoxLayout, QPushButton, QMenu


def _to_numeric_for_cursor(arr: np.ndarray) -> np.ndarray:
    """Convert array to numeric for cursor distance calculations.

    Matplotlib internally represents datetime as float (days since epoch),
    so we need to convert datetime arrays to the same representation.
    """
    if np.issubdtype(arr.dtype, np.datetime64):
        # Convert to matplotlib's date format (float days since epoch)
        return mdates.date2num(arr)
    elif np.issubdtype(arr.dtype, np.timedelta64):
        return arr.astype('timedelta64[ns]').astype(np.float64)
    return arr.astype(np.float64)
from PyQt6.QtCore import pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def _format_value(val) -> str:
    """Format a value for display, handling datetime and numeric types."""
    if isinstance(val, (np.datetime64, np.timedelta64)):
        return str(val)
    elif isinstance(val, np.ndarray):
        if np.issubdtype(val.dtype, np.datetime64):
            return str(val)
        elif np.issubdtype(val.dtype, np.timedelta64):
            return str(val)
    try:
        return f'{val:.4g}'
    except (TypeError, ValueError):
        return str(val)


class DataCursor:
    """Data cursor annotation for displaying point information. Supports dragging."""

    def __init__(self, ax, x, y, label: str, cursor_id: int, x_data: np.ndarray, y_data: np.ndarray):
        self.ax = ax
        self.x = x
        self.y = y
        self.label = label
        self.cursor_id = cursor_id
        self.x_data = x_data  # Reference to original data for snapping
        self.y_data = y_data
        self.dragging = False

        # Create annotation
        self.annotation = ax.annotate(
            f'{label}\nx={_format_value(x)}\ny={_format_value(y)}',
            xy=(x, y),
            xytext=(15, 15),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
            fontsize=8,
            picker=True
        )

        # Create marker point
        self.point, = ax.plot(x, y, 'ro', markersize=8, markeredgecolor='black', zorder=10, picker=5)

    def update_position(self, x, y):
        """Update cursor position."""
        self.x = x
        self.y = y

        # Update annotation
        self.annotation.xy = (x, y)
        self.annotation.set_text(f'{self.label}\nx={_format_value(x)}\ny={_format_value(y)}')

        # Update marker
        self.point.set_data([x], [y])

    def snap_to_nearest(self, click_x):
        """Snap cursor to nearest data point based on X coordinate."""
        if self.x_data is None or len(self.x_data) == 0:
            return

        # Convert to numeric for comparison if datetime
        x_numeric = _to_numeric_for_cursor(self.x_data)
        idx = np.argmin(np.abs(x_numeric - click_x))
        new_x = self.x_data[idx]
        new_y = self.y_data[idx]
        self.update_position(new_x, new_y)

    def remove(self):
        """Remove this cursor from the plot."""
        self.annotation.remove()
        self.point.remove()


class PlotCanvas(FigureCanvas):
    """
    Matplotlib canvas with 2x2 subplot layout.
    """

    cursorAdded = pyqtSignal(int)  # Emitted when cursor is added (cursor_id)
    cursorRemoved = pyqtSignal(int)  # Emitted when cursor is removed (cursor_id)

    def __init__(self, parent: Optional[QWidget] = None, dpi: int = 100):
        """
        Initialize the canvas.

        Args:
            parent: Parent widget
            dpi: Dots per inch for the figure
        """
        self.fig = Figure(figsize=(10, 8), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()

        # Store axes references
        self.axes: List[plt.Axes] = []
        self._share_x = True
        self._show_legend = True

        # Style settings
        self._linewidth = 0.8
        self._linestyle = ':'
        self._marker = 'o'
        self._markersize = 3.0

        # Diff settings
        self._show_diff = False
        self._twin_axes: List[Optional[plt.Axes]] = [None, None, None, None]

        # Data cursor storage
        self._cursors: List[DataCursor] = []
        self._cursor_id_counter = 0
        self._plot_data: Dict[int, Tuple[np.ndarray, List[Tuple[str, np.ndarray]]]] = {}

        # Connect mouse events
        self.mpl_connect('button_press_event', self._on_click)
        self.mpl_connect('pick_event', self._on_pick)
        self.mpl_connect('motion_notify_event', self._on_motion)
        self.mpl_connect('button_release_event', self._on_release)

        # Dragging state
        self._dragging_cursor: Optional[DataCursor] = None

        # Create initial subplot layout
        self._create_subplots()

    def _create_subplots(self, share_x: bool = True):
        """
        Create 2x2 subplot layout.

        Args:
            share_x: Whether to share X axis across all subplots
        """
        self.fig.clear()
        self.axes = []
        self._share_x = share_x
        self._cursors = []  # Clear cursors when recreating subplots
        self._plot_data = {}
        self._twin_axes = [None, None, None, None]

        if share_x:
            # Create with shared X axis
            ax1 = self.fig.add_subplot(2, 2, 1)
            ax2 = self.fig.add_subplot(2, 2, 2, sharex=ax1)
            ax3 = self.fig.add_subplot(2, 2, 3, sharex=ax1)
            ax4 = self.fig.add_subplot(2, 2, 4, sharex=ax1)
            self.axes = [ax1, ax2, ax3, ax4]
        else:
            # Independent axes
            for i in range(1, 5):
                ax = self.fig.add_subplot(2, 2, i)
                self.axes.append(ax)

        # Configure axis labels visibility
        self._configure_axis_labels()

        self.fig.tight_layout()
        self.draw()

    def _configure_axis_labels(self):
        """Configure axis labels to show only on left and bottom edges."""
        for i, ax in enumerate(self.axes):
            if self._share_x and i < 2:
                ax.tick_params(labelbottom=False)
            if i in (1, 3):
                ax.tick_params(labelleft=False)

    def _on_click(self, event):
        """Handle mouse click event for data cursor."""
        if event.inaxes is None:
            return

        # Only handle left click, and not when dragging
        if event.button != 1 or self._dragging_cursor is not None:
            return

        # Don't create cursor when toolbar is in zoom/pan mode
        toolbar = self.toolbar if hasattr(self, 'toolbar') else None
        if toolbar is None:
            # Try to find toolbar from parent
            parent = self.parent()
            if parent and hasattr(parent, 'toolbar'):
                toolbar = parent.toolbar
        if toolbar and toolbar.mode != '':
            # Toolbar is in zoom, pan, or other mode
            return

        # Find which subplot was clicked
        ax_index = None
        for i, ax in enumerate(self.axes):
            if event.inaxes == ax:
                ax_index = i
                break

        if ax_index is None or ax_index not in self._plot_data:
            return

        x_data, y_data_list = self._plot_data[ax_index]

        if len(x_data) == 0 or not y_data_list:
            return

        # Find nearest point
        click_x = event.xdata
        click_y = event.ydata

        best_dist = float('inf')
        best_point = None
        best_label = None
        best_y_data = None

        # Get axis limits to calculate relative threshold
        ax = event.inaxes
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_range = x_max - x_min
        y_range = y_max - y_min

        # Convert x_data to numeric for distance calculation
        x_numeric = _to_numeric_for_cursor(x_data)

        for label, y_data in y_data_list:
            # Find index of nearest x value using numeric comparison
            idx = np.argmin(np.abs(x_numeric - click_x))

            # Calculate normalized distance to this point
            px_numeric = x_numeric[idx]
            py = y_data[idx]
            # Normalize by axis range for fair comparison
            norm_dx = (click_x - px_numeric) / x_range if x_range > 0 else 0
            norm_dy = (click_y - py) / y_range if y_range > 0 else 0
            dist = np.sqrt(norm_dx**2 + norm_dy**2)

            if dist < best_dist:
                best_dist = dist
                best_point = (x_data[idx], y_data[idx])  # Use original x_data value
                best_label = label
                best_y_data = y_data

        # Only create cursor if click is within 5% of axis range from a data point
        CLICK_THRESHOLD = 0.05
        if best_point is not None and best_dist < CLICK_THRESHOLD:
            self._add_cursor(event.inaxes, best_point[0], best_point[1], best_label, x_data, best_y_data)

    def _on_pick(self, event):
        """Handle pick event for cursor dragging."""
        # Check if a cursor marker was picked
        for cursor in self._cursors:
            if event.artist == cursor.point:
                self._dragging_cursor = cursor
                cursor.dragging = True
                return

    def _on_motion(self, event):
        """Handle mouse motion for cursor dragging."""
        if self._dragging_cursor is None or event.inaxes is None:
            return

        # Snap to nearest data point
        self._dragging_cursor.snap_to_nearest(event.xdata)
        self.draw_idle()

    def _on_release(self, event):
        """Handle mouse release to end dragging."""
        if self._dragging_cursor is not None:
            self._dragging_cursor.dragging = False
            self._dragging_cursor = None
            self.draw()

    def _add_cursor(self, ax, x, y, label: str, x_data: np.ndarray = None, y_data: np.ndarray = None):
        """Add a data cursor at the specified point."""
        cursor_id = self._cursor_id_counter
        self._cursor_id_counter += 1

        cursor = DataCursor(ax, x, y, label, cursor_id, x_data, y_data)
        self._cursors.append(cursor)
        self.draw()
        self.cursorAdded.emit(cursor_id)

    def remove_cursor(self, cursor_id: int):
        """Remove a specific cursor by ID."""
        for cursor in self._cursors:
            if cursor.cursor_id == cursor_id:
                cursor.remove()
                self._cursors.remove(cursor)
                self.draw()
                self.cursorRemoved.emit(cursor_id)
                break

    def remove_last_cursor(self):
        """Remove the most recently added cursor."""
        if self._cursors:
            cursor = self._cursors.pop()
            cursor.remove()
            self.draw()
            self.cursorRemoved.emit(cursor.cursor_id)

    def remove_all_cursors(self):
        """Remove all cursors."""
        for cursor in self._cursors:
            cursor.remove()
            self.cursorRemoved.emit(cursor.cursor_id)
        self._cursors = []
        self.draw()

    def get_cursor_count(self) -> int:
        """Get the number of active cursors."""
        return len(self._cursors)

    def plot_data(
        self,
        subplot_index: int,
        x_data: np.ndarray,
        y_data_list: List[Tuple[str, np.ndarray]],
        x_label: str = "",
        clear: bool = True
    ):
        """
        Plot data on a specific subplot.

        Args:
            subplot_index: Index of subplot (0-3)
            x_data: X-axis data
            y_data_list: List of (label, y_data) tuples
            x_label: Label for X axis
            clear: Whether to clear the subplot before plotting
        """
        if subplot_index < 0 or subplot_index >= len(self.axes):
            return

        ax = self.axes[subplot_index]

        if clear:
            ax.clear()

        # Store plot data for cursor functionality
        self._plot_data[subplot_index] = (x_data, y_data_list)

        # Plot each Y series
        for label, y_data in y_data_list:
            ax.plot(x_data, y_data, label=label,
                    linewidth=self._linewidth,
                    linestyle=self._linestyle,
                    marker=self._marker,
                    markersize=self._markersize)

        # Plot diff on twin axis if enabled and there are 2+ signals
        if self._show_diff and len(y_data_list) >= 2:
            # Create twin axis if not exists
            if self._twin_axes[subplot_index] is None:
                self._twin_axes[subplot_index] = ax.twinx()
            twin_ax = self._twin_axes[subplot_index]
            twin_ax.clear()

            # Calculate diff between first two signals
            label1, y1 = y_data_list[0]
            label2, y2 = y_data_list[1]
            diff_data = y1 - y2
            diff_label = f"Diff ({label1} - {label2})"

            # Plot diff in gray
            twin_ax.plot(x_data, diff_data, label=diff_label,
                         linewidth=self._linewidth,
                         linestyle='--',
                         color='gray',
                         alpha=0.7)
            twin_ax.set_ylabel('Diff', color='gray', fontsize='small')
            twin_ax.tick_params(axis='y', labelcolor='gray')

            if self._show_legend:
                twin_ax.legend(loc='upper left', fontsize='small')
        else:
            # Clear twin axis if exists
            if self._twin_axes[subplot_index] is not None:
                self._twin_axes[subplot_index].clear()
                self._twin_axes[subplot_index].set_visible(False)

        # Show legend if enabled and there are labels
        if self._show_legend and y_data_list:
            ax.legend(loc='upper right', fontsize='small')

        # Re-apply label visibility settings
        self._configure_axis_labels()

    def plot_all(
        self,
        x_data: np.ndarray,
        y_data_per_subplot: List[List[Tuple[str, np.ndarray]]],
        x_label: str = "",
        title: str = "",
        share_x: bool = True,
        show_legend: bool = True
    ):
        """
        Plot data on all subplots with shared X data.

        Args:
            x_data: X-axis data (shared across all subplots)
            y_data_per_subplot: List of 4 lists, each containing (label, y_data) tuples
            x_label: Label for X axis
            title: Super title for the figure
            share_x: Whether to share X axis
            show_legend: Whether to show legends
        """
        # Convert to multi format
        x_data_per_subplot = [x_data if y_list else None for y_list in y_data_per_subplot]
        x_labels = [x_label] * 4
        self.plot_all_multi(x_data_per_subplot, y_data_per_subplot, x_labels, title, share_x, show_legend)

    def plot_all_multi(
        self,
        x_data_per_subplot: List[Optional[np.ndarray]],
        y_data_per_subplot: List[List[Tuple[str, np.ndarray]]],
        x_labels: List[str],
        title: str = "",
        share_x: bool = True,
        show_legend: bool = True
    ):
        """
        Plot data on all subplots with independent X data per subplot.

        Args:
            x_data_per_subplot: List of 4 X-axis data arrays (one per subplot, can be None)
            y_data_per_subplot: List of 4 lists, each containing (label, y_data) tuples
            x_labels: List of 4 X axis labels
            title: Super title for the figure
            share_x: Whether to share X axis
            show_legend: Whether to show legends
        """
        self._show_legend = show_legend

        # Recreate subplots if share_x setting changed
        if share_x != self._share_x:
            self._create_subplots(share_x)
        else:
            # Clear cursors when replotting
            self._cursors = []
            self._plot_data = {}

        # Clear all axes
        for ax in self.axes:
            ax.clear()

        # Plot data on each subplot
        for i, (x_data, y_data_list) in enumerate(zip(x_data_per_subplot, y_data_per_subplot)):
            if i >= 4:
                break
            if y_data_list and x_data is not None:
                self.plot_data(i, x_data, y_data_list, "", clear=False)

        # Set X labels on bottom row subplots only
        for i in (2, 3):  # Bottom row
            if i < len(x_labels) and x_labels[i]:
                self.axes[i].set_xlabel(x_labels[i])

        # Set super title
        if title:
            self.fig.suptitle(title, fontsize=10)

        # Re-configure labels and layout
        self._configure_axis_labels()
        self.fig.tight_layout()

        # Adjust for suptitle
        if title:
            self.fig.subplots_adjust(top=0.93)

        self.draw()

    def clear_all(self):
        """Clear all subplots."""
        for ax in self.axes:
            ax.clear()
        for twin_ax in self._twin_axes:
            if twin_ax is not None:
                twin_ax.clear()
                twin_ax.set_visible(False)
        self._twin_axes = [None, None, None, None]
        self.fig.suptitle("")
        self._cursors = []
        self._plot_data = {}
        self._configure_axis_labels()
        self.draw()

    def set_share_x(self, share: bool):
        """Set whether X axis is shared."""
        if share != self._share_x:
            self._create_subplots(share)

    def set_show_legend(self, show: bool):
        """Set whether to show legends."""
        self._show_legend = show

    def set_show_diff(self, show: bool):
        """Set whether to show diff on twin axis."""
        self._show_diff = show

    def set_style(self, linewidth: float = 0.8, linestyle: str = ':',
                  marker: str = 'o', markersize: float = 3.0):
        """
        Set plot style parameters.

        Args:
            linewidth: Line width
            linestyle: Line style ('-', '--', ':', '-.')
            marker: Marker type ('o', 's', '^', etc.)
            markersize: Marker size
        """
        self._linewidth = linewidth
        self._linestyle = linestyle if linestyle else 'None'
        self._marker = marker if marker else 'None'
        self._markersize = markersize


class PlotWidget(QWidget):
    """
    Widget containing the plot canvas and navigation toolbar.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create toolbar layout
        toolbar_layout = QHBoxLayout()

        # Create canvas
        self.canvas = PlotCanvas(self)

        # Create navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        toolbar_layout.addWidget(self.toolbar)

        # Add cursor control buttons
        self.remove_last_cursor_btn = QPushButton("Remove Last Cursor")
        self.remove_last_cursor_btn.setMaximumWidth(150)
        self.remove_last_cursor_btn.clicked.connect(self.canvas.remove_last_cursor)
        toolbar_layout.addWidget(self.remove_last_cursor_btn)

        self.remove_all_cursors_btn = QPushButton("Clear All Cursors")
        self.remove_all_cursors_btn.setMaximumWidth(130)
        self.remove_all_cursors_btn.clicked.connect(self.canvas.remove_all_cursors)
        toolbar_layout.addWidget(self.remove_all_cursors_btn)

        layout.addLayout(toolbar_layout)
        layout.addWidget(self.canvas)

    def plot_all(self, *args, **kwargs):
        """Proxy to canvas plot_all."""
        self.canvas.plot_all(*args, **kwargs)

    def plot_all_multi(self, *args, **kwargs):
        """Proxy to canvas plot_all_multi."""
        self.canvas.plot_all_multi(*args, **kwargs)

    def clear_all(self):
        """Proxy to canvas clear_all."""
        self.canvas.clear_all()

    def set_style(self, *args, **kwargs):
        """Proxy to canvas set_style."""
        self.canvas.set_style(*args, **kwargs)

    def set_show_diff(self, show: bool):
        """Proxy to canvas set_show_diff."""
        self.canvas.set_show_diff(show)
