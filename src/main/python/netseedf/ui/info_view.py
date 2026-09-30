"""Info tab: the file name, then tables of what is selected in the tree.

For a file or group: its attributes, dimensions and variables. For a variable:
all of its attributes, as stored in the file.
"""

from html import escape

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from netseedf.core.dataset import describe, group_header, variable_header
from netseedf.core.formatting import format_value
from netseedf.core.slicing import variable_stats
from netseedf.ui.base_view import wait_cursor


class InfoView(QWidget):
    """Shows whatever is selected in the tree: a file, a group or a variable."""

    title = "Info"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.heading = QLabel(textInteractionFlags=Qt.TextInteractionFlag.TextSelectableByMouse)
        self.subheading = QLabel(textInteractionFlags=Qt.TextInteractionFlag.TextSelectableByMouse)
        self.warnings = QLabel(wordWrap=True, visible=False)
        self.attributes = _Section(["Attribute", "Value"])
        self.dimensions = _Section(["Dimension", "Size"])
        self.variables = _Section(["Variable", "Type", "Dimensions", "Long name", "Units"])
        sections = QSplitter(Qt.Orientation.Vertical, childrenCollapsible=False)
        for section in (self.attributes, self.dimensions, self.variables):
            sections.addWidget(section)
        sections.setSizes([400, 150, 250])
        self.stats_button = QPushButton("Compute statistics", visible=False,
                                        toolTip="Min, max and mean over the whole variable "
                                                "(reads all of its data)")
        self.stats_button.clicked.connect(self._compute_stats)
        self.stats = QLabel(textInteractionFlags=Qt.TextInteractionFlag.TextSelectableByMouse)
        buttons = QHBoxLayout()
        buttons.addWidget(self.stats_button)
        buttons.addWidget(self.stats, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 0)
        layout.addWidget(self.heading)
        layout.addWidget(self.subheading)
        layout.addWidget(self.warnings)
        layout.addWidget(sections, 1)
        layout.addLayout(buttons)
        self._da = None
        self.show_text("Open a NetCDF file (File ▸ Open, or drop it on this window).")

    def show_text(self, text):
        self._show(None, text, attrs=None)

    def show_file(self, opened, group="/"):
        header = group_header(opened.path, group)
        self._show(opened, "" if group == "/" else f"Group <b>{group}</b>",
                   header.attrs, "Global attributes" if group == "/" else "Group attributes",
                   opened.warnings if group == "/" else ())
        self.dimensions.fill("Dimensions", [
            [d.name, f"{d.size} (unlimited)" if d.unlimited else str(d.size)] for d in header.dimensions])
        self.variables.fill("Variables", [
            [v.name, v.dtype, v.dims_text, *(_text(v.attrs.get(key, "")) for key in ("long_name", "units"))]
            for v in header.variables])

    def show_variable(self, opened, group, da):
        header = variable_header(opened.path, group, str(da.name))
        if header is None:  # not in the file as such; made up when opening it
            subheading = f"Variable <b>{da.name}</b> {describe(da).dims_text}, {da.dtype}"
            attrs = da.attrs
        else:
            subheading = f"Variable <b>{header.name}</b> {header.dims_text}, {header.dtype}"
            attrs = header.attrs
        if group != "/":
            subheading += f" in group {group}"
        self._show(opened, subheading, attrs, "Attributes")
        self._da = da
        self.stats_button.setVisible(da.dtype.kind in "fiub")
        self.stats_button.setEnabled(True)

    def _show(self, opened, subheading, attrs, attrs_title="", warnings=()):
        self._da = None
        self.stats_button.setVisible(False)
        self.stats.clear()
        self.heading.setText(f"<b style='font-size: large'>{opened.name}</b>" if opened else "")
        self.heading.setToolTip(str(opened.path) if opened else "")
        self.subheading.setText(subheading)
        self.subheading.setVisible(bool(subheading))
        self.warnings.setText("<br>".join(f"⚠ {escape(w)}" for w in warnings))
        self.warnings.setVisible(bool(warnings))
        self.attributes.setVisible(attrs is not None)
        if attrs is not None:
            self.attributes.fill(attrs_title, [[str(k), _text(v)] for k, v in attrs.items()])
        self.dimensions.setVisible(False)  # show_file brings these back
        self.variables.setVisible(False)

    def _compute_stats(self):
        with wait_cursor():
            try:
                s = variable_stats(self._da)
            except Exception as e:
                text = f"Statistics failed: {e}"
            else:
                text = (f"min {format_value(s.minimum)}, max {format_value(s.maximum)}, "
                        f"mean {format_value(s.mean)}, {s.count} valid, {s.missing} missing")
        self.stats_button.setEnabled(False)
        self.stats.setText(text)


class _Section(QWidget):
    """A heading over a read-only table."""

    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.heading = QLabel()
        self.table = QTableWidget(0, len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        header = self.table.horizontalHeader()
        for col in range(len(columns) - 1):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.addWidget(self.heading)
        layout.addWidget(self.table, 1)

    def fill(self, title, rows):
        self.setVisible(True)
        self.heading.setText(f"<b>{title}</b> ({len(rows)})")
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, text in enumerate(row):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self.table.setItem(r, c, item)
        self.table.resizeRowsToContents()


def _text(value):
    if isinstance(value, str):
        return value
    return ", ".join(format_value(v) for v in _as_list(value))


def _as_list(value):
    try:
        return list(value) if not isinstance(value, (bytes, str)) else [value]
    except TypeError:
        return [value]

