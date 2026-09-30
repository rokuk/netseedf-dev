"""Common behaviour of the tabs that display the selected variable."""

import traceback
from contextlib import contextmanager

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QLabel, QStackedWidget, QVBoxLayout, QWidget

from netseedf.ui.state import SelectionState


@contextmanager
def wait_cursor():
    QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        yield
    finally:
        QGuiApplication.restoreOverrideCursor()


class DataView(QWidget):
    """A tab showing the current variable.

    Subclasses build their widgets into ``self.body`` and implement
    ``refresh()``. The main window calls ``update_view()`` only while the tab
    is visible, so hidden tabs don't read any data.
    """

    title = ""
    status = Signal(str)  # e.g. the value under the mouse, for the status bar
    freeDimsChanged = Signal()  # the view now shows different dimensions

    def __init__(self, state: SelectionState, parent=None):
        super().__init__(parent)
        self.state = state
        self.body = QWidget()
        self._message = QLabel(alignment=Qt.AlignmentFlag.AlignCenter, wordWrap=True)
        self._message.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._message.setMargin(24)
        self._stack = QStackedWidget()
        self._stack.addWidget(self._message)
        self._stack.addWidget(self.body)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._stack)
        self.show_message("Open a NetCDF file and select a variable.")

    def unavailable_reason(self) -> str | None:
        """Why this view can't show the current variable, or None if it can."""
        return None

    def free_dims(self) -> tuple[str, ...]:
        """Dimensions this view displays; the others get sliders."""
        return tuple(self.state.da.dims) if self.state.da is not None else ()

    def displayed_dims(self) -> tuple[str, ...]:
        """free_dims(), or all dims (so no sliders) when the variable can't be shown."""
        da = self.state.da
        if da is None:
            return ()
        return self.free_dims() if self.unavailable_reason() is None else tuple(da.dims)

    def variable_changed(self):
        """Reset per-variable controls (called for every view, visible or not)."""

    def refresh(self):
        """Draw the current state. May raise; errors are shown in the tab."""

    def update_view(self):
        if self.state.da is None:
            self.show_message("Open a NetCDF file and select a variable.")
            return
        if reason := self.unavailable_reason():
            self.show_message(reason)
            return
        try:
            with wait_cursor():
                self._stack.setCurrentWidget(self.body)
                self.refresh()
        except Exception as e:
            traceback.print_exc()
            self.show_message(f"Couldn't show '{self.state.da.name}' here:\n\n{e}")

    def show_message(self, text):
        self._message.setText(text)
        self._stack.setCurrentWidget(self._message)
