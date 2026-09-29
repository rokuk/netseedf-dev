"""Colormap and colour-range controls for the heatmap and maps."""

from PySide6.QtCore import Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QWidget

from netseedf.core.formatting import format_value
from netseedf.core.style import COLORMAPS, RANGE_MANUAL, RANGE_MODES, Style


class StyleBar(QWidget):
    changed = Signal()
    cmapPicked = Signal(str)  # the user chose another colormap here

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cmap = QComboBox()
        self.cmap.addItems(COLORMAPS)
        self.mode = QComboBox()
        self.mode.addItems(RANGE_MODES)
        self.vmin = QLineEdit(minimumWidth=70, maximumWidth=90, placeholderText="min")
        self.vmax = QLineEdit(minimumWidth=70, maximumWidth=90, placeholderText="max")
        for edit in (self.vmin, self.vmax):
            edit.setValidator(QDoubleValidator())
            edit.editingFinished.connect(self.changed)
        self.lock = QCheckBox("Keep range", toolTip="Use the same colour range for every slice "
                              "(e.g. while stepping through time)")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for w in (QLabel("Colours:"), self.cmap, QLabel("Range:"), self.mode, self.vmin,
                  QLabel("–"), self.vmax, self.lock):
            layout.addWidget(w)
        self.cmap.currentTextChanged.connect(self.changed)
        self.cmap.currentTextChanged.connect(self.cmapPicked)
        self.mode.currentTextChanged.connect(self._mode_changed)
        self.lock.toggled.connect(self._lock_toggled)
        self._locked: tuple[float, float] | None = None
        self._last: tuple[float, float] = (0.0, 1.0)
        self._mode_changed()

    def style(self) -> Style:
        return Style(self.cmap.currentText(), self.mode.currentText(),
                     _float(self.vmin.text(), self._last[0]), _float(self.vmax.text(), self._last[1]))

    def limits_for(self, values) -> tuple[float, float]:
        if self._locked is not None:
            return self._locked
        limits = self.style().limits(values)
        if self.lock.isChecked():
            self._locked = limits
        self._last = limits
        if self.mode.currentText() != RANGE_MANUAL:
            self.vmin.setText(format_value(limits[0]))
            self.vmax.setText(format_value(limits[1]))
        return limits

    def reset_lock(self):
        self._locked = None

    def set_cmap(self, name):
        """Follow a colormap chosen elsewhere, quietly: neither signal is emitted."""
        self.cmap.blockSignals(True)
        self.cmap.setCurrentText(name)
        self.cmap.blockSignals(False)

    def _mode_changed(self):
        manual = self.mode.currentText() == RANGE_MANUAL
        self.vmin.setReadOnly(not manual)
        self.vmax.setReadOnly(not manual)
        self._locked = None
        self.changed.emit()

    def _lock_toggled(self, on):
        self._locked = self._last if on else None
        self.changed.emit()


def _float(text, default):
    try:
        return float(text)
    except ValueError:
        return default
