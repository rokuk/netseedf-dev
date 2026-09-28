"""Info tab: ncdump-style header plus what the viewer made of the variable."""

from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QHBoxLayout, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from netseedf.core.dataset import header_text, variable_header_text
from netseedf.core.formatting import format_value
from netseedf.core.slicing import variable_stats
from netseedf.ui.base_view import wait_cursor

ENCODING_KEYS = ("dtype", "scale_factor", "add_offset", "_FillValue", "missing_value", "units",
                 "calendar", "zlib", "compression", "complevel", "shuffle", "chunksizes",
                 "contiguous")


class InfoView(QWidget):
    """Shows whatever is selected in the tree: a file, a group or a variable."""

    title = "Info"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.text = QPlainTextEdit(readOnly=True)
        self.text.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.stats_button = QPushButton("Compute statistics", visible=False,
                                        toolTip="Min, max and mean over the whole variable "
                                                "(reads all of its data)")
        self.stats_button.clicked.connect(self._compute_stats)
        buttons = QHBoxLayout()
        buttons.addWidget(self.stats_button)
        buttons.addStretch()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.text, 1)
        layout.addLayout(buttons)
        self._da = None
        self._base_text = ""
        self.show_text("Open a NetCDF file (File ▸ Open, or drop it on this window).")

    def show_text(self, text):
        self._da = None
        self.stats_button.setVisible(False)
        self._set(text)

    def show_file(self, opened, group="/"):
        self._da = None
        self.stats_button.setVisible(False)
        text = header_text(opened.path, group)
        if opened.warnings and group == "/":
            text = "\n".join(f"// warning: {w}" for w in opened.warnings) + "\n\n" + text
        self._set(text)

    def show_variable(self, opened, group, da, geo):
        lines = [variable_header_text(opened.path, group, str(da.name)), "", "// as read by netseedf:"]
        lines.append(f"//   shape {tuple(da.shape)}, {da.dtype} after decoding")
        encoding = {k: da.encoding[k] for k in ENCODING_KEYS if k in da.encoding}
        if encoding:
            lines.append("//   on disk: " + ", ".join(f"{k}={v}" for k, v in encoding.items()))
        if da.coords:
            lines.append("//   coordinates: " + ", ".join(
                f"{name}{tuple(c.dims)}" for name, c in da.coords.items()))
        if geo is None:
            lines.append("//   map: no latitude/longitude coordinates found")
        else:
            lines.append(f"//   map: {geo.kind} grid, lat = {geo.lat.name}, lon = {geo.lon.name}, "
                         f"spanning dims {geo.dims}")
        self._da = da
        self.stats_button.setVisible(da.dtype.kind in "fiub")
        self.stats_button.setEnabled(True)
        self._set("\n".join(lines))

    def _set(self, text):
        self._base_text = text
        self.text.setPlainText(text)

    def _compute_stats(self):
        with wait_cursor():
            try:
                s = variable_stats(self._da)
            except Exception as e:
                extra = f"//   statistics failed: {e}"
            else:
                extra = (f"//   statistics: min {format_value(s.minimum)}, max {format_value(s.maximum)}, "
                         f"mean {format_value(s.mean)}, {s.count} valid, {s.missing} missing")
        self.stats_button.setEnabled(False)
        self.text.setPlainText(self._base_text + "\n" + extra)
