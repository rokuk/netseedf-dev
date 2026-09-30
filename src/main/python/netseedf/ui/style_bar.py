"""Colormap and colour-range controls for the heatmap and maps."""

from PySide6.QtCore import Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QWidget

from netseedf.core.formatting import format_value
from netseedf.core.style import COLORMAPS, Style
from netseedf.ui.base_view import buddy_label


class StyleBar(QWidget):
    """Colormap, and a colour range that's either automatic (each slice's min–max) or
    typed in. Typing a number turns Auto off; the range then stays as it is (e.g. while
    stepping through time) until Auto is turned back on or another variable is chosen.
    """

    changed = Signal()
    cmapPicked = Signal(str)  # the user chose another colormap here

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cmap = QComboBox()
        self.cmap.addItems(COLORMAPS)
        self.auto = QCheckBox("&Auto range", checked=True,
                              toolTip="Colour range from each slice's smallest to largest value")
        self.vmin = QLineEdit(minimumWidth=70, maximumWidth=90, placeholderText="min",
                              accessibleName="Range minimum")
        self.vmax = QLineEdit(minimumWidth=70, maximumWidth=90, placeholderText="max",
                              accessibleName="Range maximum")
        for edit in (self.vmin, self.vmax):
            edit.setValidator(QDoubleValidator())
            edit.editingFinished.connect(self._range_edited)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for w in (buddy_label("&Colours:", self.cmap), self.cmap, buddy_label("&Range:", self.vmin),
                  self.vmin, QLabel("–"), self.vmax, self.auto):
            layout.addWidget(w)
        self.cmap.currentTextChanged.connect(self.changed)
        self.cmap.currentTextChanged.connect(self.cmapPicked)
        self.auto.toggled.connect(self.changed)
        self._last: tuple[float, float] = (0.0, 1.0)  # the range last drawn with

    def style(self) -> Style:
        return Style(self.cmap.currentText(), self.auto.isChecked(),
                     _float(self.vmin.text(), self._last[0]), _float(self.vmax.text(), self._last[1]))

    def limits_for(self, values) -> tuple[float, float]:
        """The colour range to draw `values` with; shows it in the boxes."""
        self._last = limits = self.style().limits(values)
        self.vmin.setText(format_value(limits[0]))
        self.vmax.setText(format_value(limits[1]))
        return limits

    def reset_range(self):
        """Back to Auto (for another variable, whose values may be nothing alike)."""
        self.auto.blockSignals(True)
        self.auto.setChecked(True)
        self.auto.blockSignals(False)

    def set_cmap(self, name):
        """Follow a colormap chosen elsewhere, quietly: neither signal is emitted."""
        self.cmap.blockSignals(True)
        self.cmap.setCurrentText(name)
        self.cmap.blockSignals(False)

    def _range_edited(self):
        edit = self.sender()
        if not edit.isModified():
            return  # just left the box without typing anything
        edit.setModified(False)
        if self.auto.isChecked():
            self.auto.setChecked(False)  # emits changed
        else:
            self.changed.emit()


def _float(text, default):
    try:
        return float(text)
    except ValueError:
        return default
