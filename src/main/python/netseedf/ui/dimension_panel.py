"""Sliders for the dimensions the current view doesn't display."""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QToolButton,
    QWidget,
)

from netseedf.core.formatting import index_label
from netseedf.ui.state import SelectionState

APPLY_INTERVAL_MS = 120  # redraw at most this often while a slider is dragged
PLAY_INTERVAL_MS = 400


class _DimRow:
    def __init__(self, panel, layout, row, dim, size):
        self.dim = dim
        self.name = QLabel(f"<b>{dim}</b> ({size})")
        self.slider = QSlider(Qt.Orientation.Horizontal, maximum=size - 1, pageStep=max(1, size // 10),
                              accessibleName=f"{dim} index")
        self.spin = QSpinBox(maximum=size - 1, accessibleName=f"{dim} index")
        self.value = QLabel(minimumWidth=150)
        self.value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.play = QToolButton(checkable=True)
        self.set_playing(False)
        for w in (self.slider, self.spin, self.play):
            w.setEnabled(size > 1)
        for col, w in enumerate((self.name, self.slider, self.spin, self.value, self.play)):
            layout.addWidget(w, row, col)
        self.slider.valueChanged.connect(self.spin.setValue)
        self.spin.valueChanged.connect(self.slider.setValue)
        self.slider.valueChanged.connect(lambda v: panel._index_changed(self, v))
        self.play.toggled.connect(lambda on: panel._play_toggled(self, on))

    def show_value(self, text):
        self.value.setText(text)
        for w in (self.slider, self.spin):
            w.setAccessibleDescription(text)  # read out with the index

    def set_playing(self, on):
        self.play.setText("⏸" if on else "▶")
        self.play.setAccessibleName(f"{'Pause' if on else 'Play'} {self.dim}")
        self.play.setToolTip(f"Pause stepping through {self.dim}" if on else f"Step through {self.dim}")

    def widgets(self):
        return (self.name, self.slider, self.spin, self.value, self.play)


class DimensionPanel(QWidget):
    def __init__(self, state: SelectionState, parent=None):
        super().__init__(parent)
        self.state = state
        self._rows: list[_DimRow] = []
        self._pending: dict[str, int] = {}
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(6, 2, 6, 2)
        self._layout.setColumnStretch(1, 1)
        self._apply_timer = QTimer(self, singleShot=True, interval=APPLY_INTERVAL_MS)
        self._apply_timer.timeout.connect(self._apply)
        self._play_timer = QTimer(self, interval=PLAY_INTERVAL_MS)
        self._play_timer.timeout.connect(self._step)
        self._playing: _DimRow | None = None
        state.indicesChanged.connect(self._sync)
        self.setVisible(False)

    def show_dims(self, free_dims):
        """Rebuild with a row for every dim of the variable not in `free_dims`."""
        self._stop_playing()
        for row in self._rows:
            for w in row.widgets():
                w.deleteLater()
        self._rows.clear()
        self._pending.clear()
        da = self.state.da
        dims = [] if da is None else [d for d in da.dims if d not in free_dims]
        for r, dim in enumerate(dims):
            row = _DimRow(self, self._layout, r, dim, da.sizes[dim])
            index = min(self.state.indices.get(dim, 0), da.sizes[dim] - 1)
            row.slider.setValue(index)
            row.show_value(index_label(da, dim, index))
            self._rows.append(row)
        self.setVisible(bool(dims))

    def focus_first(self):
        """Keyboard focus to the first dimension's slider, if there are any."""
        if self._rows:
            self._rows[0].slider.setFocus()

    def _sync(self):
        """Follow index changes made elsewhere (not the ones still pending from here)."""
        for row in self._rows:
            index = self.state.indices.get(row.dim, 0)
            if row.dim not in self._pending and row.slider.value() != index:
                row.slider.blockSignals(True)
                row.spin.blockSignals(True)
                row.slider.setValue(index)
                row.spin.setValue(index)
                row.slider.blockSignals(False)
                row.spin.blockSignals(False)
                row.show_value(index_label(self.state.da, row.dim, index))

    def _index_changed(self, row, value):
        row.show_value(index_label(self.state.da, row.dim, value))
        self._pending[row.dim] = value
        if not self._apply_timer.isActive():
            self._apply_timer.start()

    def _apply(self):
        pending, self._pending = self._pending, {}
        self.state.set_indices(pending)

    def _play_toggled(self, row, on):
        if on:
            if self._playing is not None and self._playing is not row:
                self._playing.play.setChecked(False)
            self._playing = row
            row.set_playing(True)
            self._play_timer.start()
        elif self._playing is row:
            self._stop_playing()

    def _stop_playing(self):
        self._play_timer.stop()
        if self._playing is not None:
            row, self._playing = self._playing, None
            row.set_playing(False)
            row.play.setChecked(False)

    def _step(self):
        slider = self._playing.slider
        slider.setValue((slider.value() + 1) % (slider.maximum() + 1))
