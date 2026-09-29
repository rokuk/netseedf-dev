"""Plot tab: line plot of a 1D slice or heatmap of a 2D slice."""

import traceback
from dataclasses import dataclass

import matplotlib.dates as mdates
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout

from netseedf.core.detail import DetailTracker, padded, resolution_note
from netseedf.core.formatting import (
    dim_label,
    format_value,
    selection_text,
    value_text,
    variable_label,
)
from netseedf.core.render import cell_edges
from netseedf.core.slicing import Slice, as_float, coord_values, extract, index_window, is_finer
from netseedf.ui.base_view import DataView, wait_cursor
from netseedf.ui.mpl_canvas import MplWidget
from netseedf.ui.style_bar import StyleBar

LINE = "Line"
HEATMAP = "Heatmap"
MAX_LINE_POINTS = 200_000
MAX_IMAGE_SIZE = 2000
DETAIL_DELAY_MS = 250  # wait for zooming/panning to pause before reading more data


class Axis:
    """Positions along one plot axis, and how to label them.

    Numeric and datetime64 coordinates are used directly. Anything else
    (cftime dates, strings, missing coordinates) is plotted against its index
    and labelled with the coordinate values.
    """

    def __init__(self, da, dim):
        self.dim = dim
        self.label = dim_label(da, dim)
        self.coords = coord_values(da, dim)
        self.is_date = self.coords is not None and self.coords.dtype.kind == "M"
        if self.coords is not None and self.coords.dtype.kind in "fiu":
            self.full_pos = self.coords.astype(float)
        elif self.is_date:
            self.full_pos = mdates.date2num(self.coords)
        else:
            self.full_pos = np.arange(da.sizes[dim], dtype=float)
        self.by_index = not self.is_date and (self.coords is None or self.coords.dtype.kind not in "fiu")

    def pos(self, index):
        """Plot positions of the given full-resolution indices."""
        return self.full_pos[index]

    def values(self, index):
        """What to pass to matplotlib as the data coordinate."""
        return self.coords[index] if self.is_date else self.full_pos[index]

    def text(self, i):
        return format_value(self.coords[i]) if self.coords is not None else f"#{i}"

    def window(self, lo, hi):
        return index_window(self.full_pos, lo, hi)

    def label_axis(self, axis):
        axis.set_label_text(self.label)
        if self.by_index and self.coords is not None:
            coords = self.coords
            axis.set_major_locator(MaxNLocator(8, integer=True))

            def fmt(x, _):
                i = int(round(x))
                return format_value(coords[i]) if 0 <= i < len(coords) and abs(x - i) < 1e-6 else ""
            axis.set_major_formatter(FuncFormatter(fmt))


def _monotonic(pos):
    d = np.diff(pos)
    return bool(np.all(d > 0) or np.all(d < 0))


def _uniform(pos):
    d = np.diff(pos)
    return len(d) == 0 or bool(np.allclose(d, d[0], rtol=1e-3, atol=0))


@dataclass
class _Layer:
    """One drawn heatmap image: the overview, or a zoomed-in detail over it."""

    slice: Slice
    values: np.ndarray
    rows: np.ndarray  # full-resolution indices along y and x
    cols: np.ndarray
    artist: object

    def lookup(self, xa, ya, x, y, strict):
        """(row, col, value) nearest to x/y; None if outside the layer (when strict)."""
        ypos, xpos = ya.pos(self.rows), xa.pos(self.cols)
        if strict and not (min(xpos[0], xpos[-1]) <= x <= max(xpos[0], xpos[-1])
                           and min(ypos[0], ypos[-1]) <= y <= max(ypos[0], ypos[-1])):
            return None
        i, j = int(np.argmin(np.abs(ypos - y))), int(np.argmin(np.abs(xpos - x)))
        return self.rows[i], self.cols[j], self.values[i, j]


class PlotView(DataView):
    title = "Plot"

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self.kind = QComboBox()
        self.kind.addItems([LINE, HEATMAP])
        self.x = QComboBox(toolTip="Dimension along the x axis")
        self.y = QComboBox(toolTip="Dimension along the y axis (heatmap)")
        self.y_label = QLabel("Y:")
        self.style_bar = StyleBar()
        self.note = QLabel(minimumWidth=1)  # a long note mustn't widen the window
        bar = QHBoxLayout()
        for w in (self.kind, QLabel("X:"), self.x, self.y_label, self.y):
            bar.addWidget(w)
        bar.addWidget(self.note, 1)
        self.mpl = MplWidget()
        self.mpl.add_to_toolbar_row(self.style_bar)
        self.mpl.hover.connect(self.status)
        layout = QVBoxLayout(self.body)
        layout.setContentsMargins(4, 4, 4, 0)
        layout.addLayout(bar)
        layout.addWidget(self.mpl, 1)
        self._updating = False
        for combo in (self.kind, self.x, self.y):
            combo.currentTextChanged.connect(self._controls_changed)
        self.style_bar.changed.connect(self._restyle)
        self._detail_timer = QTimer(self, singleShot=True, interval=DETAIL_DELAY_MS)
        self._detail_timer.timeout.connect(self._update_detail)
        self._tracker = DetailTracker()
        self._ax = None
        self._detail = self._detail_layer = None

    def unavailable_reason(self):
        da = self.state.da
        if da.ndim == 0:
            return f"'{da.name}' is a single value: {format_value(da.values)}"
        if da.dtype.kind not in "fiubM":
            return f"'{da.name}' holds {da.dtype} values, which can't be plotted. See the Table tab."
        return None

    def free_dims(self):
        da = self.state.da
        if da is None or da.ndim == 0:
            return ()
        if self.kind.currentText() == HEATMAP:
            return (self.y.currentText(), self.x.currentText())
        return (self.x.currentText(),)

    def variable_changed(self):
        da = self.state.da
        self._updating = True
        dims = [] if da is None else [str(d) for d in da.dims]
        for combo in (self.x, self.y):
            combo.clear()
            combo.addItems(dims)
        heatmap_ok = len(dims) >= 2 and da.dtype.kind in "fiub"
        self.kind.model().item(1).setEnabled(heatmap_ok)
        if heatmap_ok:
            self.kind.setCurrentText(HEATMAP)
            geo = self.state.geo
            y, x = geo.dims if geo is not None and geo.kind != "points" else dims[-2:]
            self.x.setCurrentText(x)
            self.y.setCurrentText(y)
        else:
            self.kind.setCurrentText(LINE)
            if da is not None and da.ndim:
                self.x.setCurrentText(_longest_dim(da))
        self._show_controls()
        self.style_bar.reset_range()
        self.mpl.reset()
        self._updating = False

    def _show_controls(self):
        heatmap = self.kind.currentText() == HEATMAP
        for w in (self.y_label, self.y, self.style_bar):
            w.setVisible(heatmap)

    def _controls_changed(self):
        if self._updating:
            return
        if self.kind.currentText() == HEATMAP and self.x.currentText() == self.y.currentText():
            other = self.y if self.sender() is self.x else self.x
            self._updating = True
            other.setCurrentIndex((other.currentIndex() + 1) % other.count())
            self._updating = False
        self._show_controls()
        self.freeDimsChanged.emit()

    def _restyle(self):
        if self.state.da is not None and self.isVisible():
            self.update_view()

    def refresh(self):
        self._detail_timer.stop()
        self._tracker.reset()
        self._ax = self._detail = self._detail_layer = None
        if self.kind.currentText() == HEATMAP:
            self._heatmap()
        else:
            self._line()
        # Zooming or panning (including the zoom kept from before) may call for more detail.
        for event in ("xlim_changed", "ylim_changed"):
            self._ax.callbacks.connect(event, lambda _ax: self._detail_timer.start())
        self._update_detail()  # now, so a kept zoom never shows the overview alone

    def _title(self, sl):
        where = selection_text(self.state.da, sl.fixed)
        return f"{self.state.da.name}" + (f"\n{where}" if where else "")

    def _update_note(self):
        self.note.setText(resolution_note(self._base_slice, self._detail))

    def _update_detail(self):
        if self._ax is None or self.state.da is None or not self.isVisible():
            return
        try:
            with wait_cursor():
                if self.kind.currentText() == HEATMAP:
                    self._heatmap_detail()
                else:
                    self._line_detail()
        except Exception:  # the overview is still there; don't lose it over the detail
            traceback.print_exc()

    # --- line ---------------------------------------------------------------

    def _line(self):
        da = self.state.da
        dim = self.x.currentText()
        sl = extract(da, (dim,), self.state.indices, MAX_LINE_POINTS)
        self._xa = axis = Axis(da, dim)
        self._base_slice = sl
        self._line_index, self._line_y = sl.index(0), sl.values
        previous = self.mpl.begin((self.state.ref, LINE, dim))
        self._ax = ax = self.mpl.figure.add_subplot()
        (self._line_artist,) = ax.plot(axis.values(self._line_index), self._line_y, lw=1.2,
                                       marker="." if len(sl.values) <= 60 else None)
        axis.label_axis(ax.xaxis)
        ax.set_ylabel(variable_label(da))
        ax.set_title(self._title(sl), fontsize="medium")
        ax.grid(True, alpha=0.3)

        def format_coord(x, _y):
            if not len(self._line_index):
                return ""
            k = int(np.argmin(np.abs(axis.pos(self._line_index) - x)))
            return (f"{axis.dim} = {axis.text(self._line_index[k])}   "
                    f"{da.name} = {value_text(da, self._line_y[k])}")
        ax.format_coord = format_coord
        self._update_note()
        self.mpl.finish(ax, previous)

    def _line_detail(self):
        da, axis, base = self.state.da, self._xa, self._base_slice
        visible = (*self._ax.get_xlim(), 0, 1)
        if self._detail is not None and self._tracker.up_to_date(visible):
            return
        lo, hi, _, _ = padded(visible)
        window = axis.window(lo, hi)
        if window is None or not is_finer(base, {axis.dim: window}, MAX_LINE_POINTS):
            if self._detail is not None:
                self._set_line(base.index(0), base.values, None)
                self._tracker.reset()
            return
        detail = extract(da, (axis.dim,), self.state.indices, MAX_LINE_POINTS, {axis.dim: window})
        # The overview outside the window, the detail inside it: still one line.
        index = base.index(0)
        before, after = index < window[0], index >= window[1]
        self._set_line(np.concatenate([index[before], detail.index(0), index[after]]),
                       np.concatenate([base.values[before], detail.values, base.values[after]]),
                       detail)
        self._tracker.remember(visible, padded(visible))

    def _set_line(self, index, y, detail):
        self._line_index, self._line_y, self._detail = index, y, detail
        self._line_artist.set_data(self._xa.values(index), y)
        self._update_note()
        self.mpl.canvas.draw_idle()

    # --- heatmap --------------------------------------------------------------

    def _heatmap(self):
        da = self.state.da
        ydim, xdim = self.y.currentText(), self.x.currentText()
        sl = extract(da, (ydim, xdim), self.state.indices, MAX_IMAGE_SIZE)
        self._xa, self._ya = xa, ya = Axis(da, xdim), Axis(da, ydim)
        self._base_slice = sl
        self._limits = self.style_bar.limits_for(as_float(sl.values))
        previous = self.mpl.begin((self.state.ref, HEATMAP, ydim, xdim))
        self._ax = ax = self.mpl.figure.add_subplot()
        self._base = self._draw_layer(sl, zorder=1)
        # Values grow to the right and upwards, e.g. north up when latitude is stored N to S.
        ax.set_xlim(sorted(ax.get_xlim()))
        ax.set_ylim(sorted(ax.get_ylim()))
        for a, axis in ((xa, ax.xaxis), (ya, ax.yaxis)):
            a.label_axis(axis)
            if a.is_date:
                axis.axis_date()
        self.mpl.figure.colorbar(self._base.artist, ax=ax, label=variable_label(da))
        ax.set_title(self._title(sl), fontsize="medium")

        def format_coord(x, y):
            found = None
            if self._detail_layer is not None:
                found = self._detail_layer.lookup(xa, ya, x, y, strict=True)
            i, j, value = found or self._base.lookup(xa, ya, x, y, strict=False)
            return f"{xdim} = {xa.text(j)}   {ydim} = {ya.text(i)}   {da.name} = {value_text(da, value)}"
        ax.format_coord = format_coord
        self._update_note()
        self.mpl.finish(ax, previous)

    def _draw_layer(self, sl, zorder):
        values = as_float(sl.values)
        rows, cols = sl.index(0), sl.index(1)
        ypos, xpos = self._ya.pos(rows), self._xa.pos(cols)
        vmin, vmax = self._limits
        cmap = self.style_bar.style().cmap
        masked = np.ma.masked_invalid(values)
        if _monotonic(xpos) and _monotonic(ypos) and _uniform(xpos) and _uniform(ypos):
            x_edges, y_edges = cell_edges(xpos), cell_edges(ypos)
            artist = self._ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower",
                                     extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
                                     aspect="auto", interpolation="nearest", zorder=zorder)
        else:
            artist = self._ax.pcolormesh(xpos, ypos, masked, cmap=cmap, vmin=vmin, vmax=vmax,
                                         shading="nearest", zorder=zorder)
        return _Layer(sl, values, rows, cols, artist)

    def _heatmap_detail(self):
        ax = self._ax
        visible = (*ax.get_xlim(), *ax.get_ylim())
        if self._detail_layer is not None and self._tracker.up_to_date(visible):
            return
        x0, x1, y0, y1 = padded(visible)
        wx, wy = self._xa.window(x0, x1), self._ya.window(y0, y1)
        window = None if wx is None or wy is None else {self._ya.dim: wy, self._xa.dim: wx}
        if window is None or not is_finer(self._base_slice, window, MAX_IMAGE_SIZE):
            self._remove_detail()
            return
        sl = extract(self.state.da, (self._ya.dim, self._xa.dim), self.state.indices,
                     MAX_IMAGE_SIZE, window)
        ax.set_autoscale_on(False)  # adding the image mustn't move the view
        layer = self._draw_layer(sl, zorder=2)
        self._remove_detail()
        self._detail_layer, self._detail = layer, sl
        self._tracker.remember(visible, (x0, x1, y0, y1))
        self._update_note()
        self.mpl.canvas.draw_idle()

    def _remove_detail(self):
        if self._detail_layer is not None:
            self._detail_layer.artist.remove()
            self._detail_layer = self._detail = None
            self._tracker.reset()
            self._update_note()
            self.mpl.canvas.draw_idle()


def _longest_dim(da):
    return max(da.dims, key=lambda d: da.sizes[d])
