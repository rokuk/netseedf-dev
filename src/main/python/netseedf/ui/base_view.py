"""Common behaviour of the tabs that display the selected variable."""

import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QLabel, QProgressBar, QStackedWidget, QVBoxLayout, QWidget

from netseedf.core.dataset import FILE_LOCK
from netseedf.ui.state import SelectionState

# Reading a slice can take seconds, e.g. when the file's chunks span many time steps
# (see slicing.slow_layout). Views with a loader() read in a worker thread meanwhile,
# one read at a time, and stay responsive. (Tests read right away instead.)
READ_IN_BACKGROUND = True
BUSY_DELAY_MS = 300  # reads quicker than this show no sign of reading
_reader = None


def _read_in_background(read):
    global _reader
    if _reader is None:
        _reader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reader")
    _reader.submit(read)


@contextmanager
def wait_cursor():
    QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        yield
    finally:
        QGuiApplication.restoreOverrideCursor()


def buddy_label(text, widget) -> QLabel:
    """A label naming `widget` for screen readers; its &-mnemonic (Alt+key) focuses it."""
    label = QLabel(text)
    label.setBuddy(widget)
    return label


class DataView(QWidget):
    """A tab showing the current variable.

    Subclasses build their widgets into ``self.body`` and implement
    ``refresh()``. The main window calls ``update_view()`` only while the tab
    is visible, so hidden tabs don't read any data.

    A view whose data may take long to read also implements ``loader()``: the
    reading then happens in the background, and ``refresh(loaded)`` draws what
    was read. Until then the view shows what it showed before.
    """

    title = ""
    status = Signal(str)  # e.g. the value under the mouse, for the status bar
    freeDimsChanged = Signal()  # the view now shows different dimensions
    _loaded = Signal(int, object, object)  # from the reader: generation, what was read, error

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
        # Shown while reading in the background. Its room is kept, so the view doesn't jump.
        self.busy = QProgressBar(minimum=0, maximum=0, textVisible=False, maximumHeight=3,
                                 accessibleName="Reading data")
        policy = self.busy.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.busy.setSizePolicy(policy)
        self.busy.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.busy)
        layout.addWidget(self._stack)
        self._generation = 0  # counts update_view() calls; reads for earlier ones aren't drawn
        self._reading = False
        self._queued = None  # (generation, job) to read once the current read is done
        self._drawn = None  # VariableRef of what the body shows
        self._busy_timer = QTimer(self, singleShot=True, interval=BUSY_DELAY_MS)
        self._busy_timer.timeout.connect(self._show_busy)
        self._loaded.connect(self._finish_read)
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

    def loader(self):
        """A function reading what refresh() draws, or None to read in refresh() itself.

        Called on the GUI thread, so it may look at the controls. The function
        it returns runs in a worker thread: it mustn't touch any widget, nor
        anything that may change before it's done (e.g. the state.indices dict).
        """
        return None

    def refresh(self, loaded=None):
        """Draw the current state, given what loader()'s function returned.

        May raise; errors are shown in the tab.
        """

    def update_view(self):
        self._generation += 1
        if self.state.da is None:
            self.show_message("Open a NetCDF file and select a variable.")
            return
        if reason := self.unavailable_reason():
            self.show_message(reason)
            return
        try:
            job = self.loader()
            if job is None:
                self._run_refresh(self.refresh)
            elif READ_IN_BACKGROUND:
                self._queued = (self._generation, job)  # replaces any read not started yet
                if not self._reading:
                    self._start_read()
            else:
                loaded = job()
                self._run_refresh(lambda: self.refresh(loaded))
        except Exception as e:
            traceback.print_exc()
            self._failed(e)

    def _run_refresh(self, refresh):
        try:
            with wait_cursor():
                self._stack.setCurrentWidget(self.body)
                refresh()
            self._drawn = self.state.ref
        except Exception as e:
            traceback.print_exc()
            self._failed(e)

    def _failed(self, error):
        self.show_message(f"Couldn't show '{self.state.da.name}' here:\n\n{error}")

    def _start_read(self):
        (generation, job), self._queued = self._queued, None
        self._reading = True
        if self.busy.isHidden():
            self._busy_timer.start()

        def read():
            with FILE_LOCK:
                try:
                    result = job(), None
                except Exception as e:
                    traceback.print_exc()
                    result = None, e
            try:
                self._loaded.emit(generation, *result)  # queued to the GUI thread
            except RuntimeError:  # the view is gone (the app was closed meanwhile)
                pass

        _read_in_background(read)

    def _finish_read(self, generation, loaded, error):
        self._reading = False
        if self._queued is not None:  # asked for since, so this read is out of date
            self._start_read()
            return
        self._busy_timer.stop()
        self.busy.hide()
        if generation != self._generation:  # e.g. another variable that can't be shown here
            return
        if error is not None:
            self._failed(error)
        else:
            self._run_refresh(lambda: self.refresh(loaded))

    def _show_busy(self):
        self.busy.show()
        if self.state.da is not None and self._drawn != self.state.ref:  # not the old variable
            self.show_message(f"Reading '{self.state.da.name}'…")

    def show_message(self, text):
        self._drawn = None
        self._message.setText(text)
        self._stack.setCurrentWidget(self._message)
