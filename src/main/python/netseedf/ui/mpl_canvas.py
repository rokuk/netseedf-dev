"""Embedded matplotlib figure with its zoom/pan/save toolbar."""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget


class MplWidget(QWidget):
    hover = Signal(str)  # the axes' format_coord() text under the mouse

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        # Coordinates go to the status bar (see `hover`), leaving room here for controls.
        self.toolbar = NavigationToolbar2QT(self.canvas, self, coordinates=False)
        self._row = QHBoxLayout()
        self._row.addWidget(self.toolbar)
        self._row.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._row)
        layout.addWidget(self.canvas, 1)
        self.canvas.mpl_connect("motion_notify_event", self._motion)
        self.canvas.mpl_connect("axes_leave_event", lambda _e: self.hover.emit(""))
        self._view_key = None

    def add_to_toolbar_row(self, widget):
        self._row.addWidget(widget)

    def _motion(self, event):
        ax = event.inaxes
        if ax is None or event.xdata is None or ax not in self.figure.axes[:1]:
            return
        try:
            self.hover.emit(ax.format_coord(event.xdata, event.ydata))
        except Exception:  # never let a readout break mouse handling
            self.hover.emit("")

    def begin(self, view_key):
        """Clear the figure. Returns the previous zoom if `view_key` is unchanged."""
        previous = None
        if view_key == self._view_key and self.figure.axes:
            ax = self.figure.axes[0]
            previous = (ax.get_xlim(), ax.get_ylim())
        self._view_key = view_key
        self.figure.clear()
        return previous

    def finish(self, ax, previous_limits=None):
        """Draw; Home in the toolbar goes to the full view, the zoom is kept."""
        self.toolbar.update()
        self.toolbar.push_current()
        if previous_limits:
            ax.set_xlim(previous_limits[0])
            ax.set_ylim(previous_limits[1])
            self.toolbar.push_current()
        self.canvas.draw_idle()

    def reset(self):
        self._view_key = None
