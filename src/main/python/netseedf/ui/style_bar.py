"""Colormap and colour-range controls for the heatmap and maps."""

from PySide6.QtCore import Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QWidget

from netseedf.core.formatting import format_value
from netseedf.core.style import COLORMAPS, Style
from netseedf.ui.base_view import buddy_label


class RangeBar(QWidget):
    """A min–max range that's either automatic or typed in. Typing a number turns Auto
    off; the range then stays as it is (e.g. while stepping through time) until Auto is
    turned back on or another variable is chosen.
    """

    changed = Signal()

    def __init__(self, label, auto_text, auto_tip, parent=None):
        super().__init__(parent)
        self.auto = QCheckBox(auto_text, checked=True, toolTip=auto_tip)
        self.vmin = QLineEdit(minimumWidth=70, maximumWidth=90, placeholderText="min",
                              accessibleName="Range minimum")
        self.vmax = QLineEdit(minimumWidth=70, maximumWidth=90, placeholderText="max",
                              accessibleName="Range maximum")
        for edit in (self.vmin, self.vmax):
            edit.setValidator(QDoubleValidator())
            edit.editingFinished.connect(self._edited)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for w in (buddy_label(label, self.vmin), self.vmin, QLabel("–"), self.vmax, self.auto):
            layout.addWidget(w)
        self.auto.toggled.connect(self.changed)
        self._last: tuple[float, float] = (0.0, 1.0)  # the range last shown

    def typed(self) -> tuple[float, float]:
        """What's in the boxes (the range last shown where they don't hold a number)."""
        return _float(self.vmin.text(), self._last[0]), _float(self.vmax.text(), self._last[1])

    def show_range(self, limits):
        self._last = limits = (float(limits[0]), float(limits[1]))
        self.vmin.setText(format_value(limits[0]))
        self.vmax.setText(format_value(limits[1]))

    def reset(self):
        """Back to Auto (for another variable, whose values may be nothing alike)."""
        self.auto.blockSignals(True)
        self.auto.setChecked(True)
        self.auto.blockSignals(False)

    def _edited(self):
        edit = self.sender()
        if not edit.isModified():
            return  # just left the box without typing anything
        edit.setModified(False)
        if self.auto.isChecked():
            self.auto.setChecked(False)  # emits changed
        else:
            self.changed.emit()


class StyleBar(QWidget):
    """Colormap, and a colour range that's either automatic (each slice's min–max) or
    typed in (see RangeBar).
    """

    changed = Signal()
    cmapPicked = Signal(str)  # the user chose another colormap here

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cmap = QComboBox()
        self.cmap.addItems(COLORMAPS)
        self.range = RangeBar("&Range:", "&Auto range",
                              "Colour range from each slice's smallest to largest value")
        self.auto, self.vmin, self.vmax = self.range.auto, self.range.vmin, self.range.vmax
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for w in (buddy_label("&Colours:", self.cmap), self.cmap, self.range):
            layout.addWidget(w)
        self.cmap.currentTextChanged.connect(self.changed)
        self.cmap.currentTextChanged.connect(self.cmapPicked)
        self.range.changed.connect(self.changed)

    def style(self) -> Style:
        return Style(self.cmap.currentText(), self.auto.isChecked(), *self.range.typed())

    def limits_for(self, values) -> tuple[float, float]:
        """The colour range to draw `values` with; shows it in the boxes."""
        limits = self.style().limits(values)
        self.range.show_range(limits)
        return limits

    def reset_range(self):
        """Back to Auto (for another variable, whose values may be nothing alike)."""
        self.range.reset()

    def set_cmap(self, name):
        """Follow a colormap chosen elsewhere, quietly: neither signal is emitted."""
        self.cmap.blockSignals(True)
        self.cmap.setCurrentText(name)
        self.cmap.blockSignals(False)


def _float(text, default):
    try:
        return float(text)
    except ValueError:
        return default
