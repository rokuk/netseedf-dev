"""Table tab: a 2D slice of the variable, read lazily as you scroll."""

from collections import OrderedDict

import numpy as np
import pandas as pd
from PySide6.QtCore import QAbstractTableModel, Qt
from PySide6.QtGui import QColor, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableView,
    QVBoxLayout,
)

from netseedf.core.formatting import format_value, selection_text
from netseedf.core.slicing import coord_values, fixed_indices, lazy_slice
from netseedf.ui.base_view import DataView, buddy_label, wait_cursor

BLOCK = 256
MAX_BLOCKS = 64
MAX_COPY_CELLS = 1_000_000
LARGE_EXPORT_CELLS = 20_000_000


class LazyArrayModel(QAbstractTableModel):
    """Table model over a (not yet loaded) 0-2D DataArray, cached in blocks."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._da = None
        self._rows = self._cols = 0
        self._row_labels = self._col_labels = None
        self._blocks = OrderedDict()
        self._name = ""

    def set_array(self, da, row_labels=None, col_labels=None):
        self.beginResetModel()
        self._da = da
        self._rows = da.shape[0] if da.ndim >= 1 else 1
        self._cols = da.shape[1] if da.ndim == 2 else 1
        self._row_labels, self._col_labels = row_labels, col_labels
        self._name = str(da.name)
        self._blocks.clear()
        self.endResetModel()

    def rowCount(self, parent=None):  # noqa: N802 (Qt override)
        return 0 if parent is not None and parent.isValid() else self._rows

    def columnCount(self, parent=None):  # noqa: N802 (Qt override)
        return 0 if parent is not None and parent.isValid() else self._cols

    def value(self, row, col):
        key = (row // BLOCK, col // BLOCK)
        block = self._blocks.get(key)
        if block is None:
            block = self._read_block(*key)
            self._blocks[key] = block
            if len(self._blocks) > MAX_BLOCKS:
                self._blocks.popitem(last=False)
        else:
            self._blocks.move_to_end(key)
        return block[row % BLOCK, col % BLOCK]

    def _read_block(self, bi, bj):
        da = self._da
        if da.ndim == 0:
            return np.asarray(da.values).reshape(1, 1)
        rows = slice(bi * BLOCK, (bi + 1) * BLOCK)
        if da.ndim == 1:
            return np.asarray(da[rows].values).reshape(-1, 1)
        return np.asarray(da[rows, bj * BLOCK:(bj + 1) * BLOCK].values)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or self._da is None:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return format_value(self.value(index.row(), index.column()))
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if self._da.dtype.kind in "fiucb":
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            return None
        if role == Qt.ItemDataRole.ForegroundRole:
            v = self.value(index.row(), index.column())
            if isinstance(v, float | np.floating) and np.isnan(v):
                return QColor(Qt.GlobalColor.gray)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if self._da is None:
            return None
        horizontal = orientation == Qt.Orientation.Horizontal
        labels = self._col_labels if horizontal else self._row_labels
        if role == Qt.ItemDataRole.DisplayRole:
            if horizontal and self._da.ndim < 2:
                return self._name
            if labels is not None:
                return format_value(labels[section])
            return str(section)
        if role == Qt.ItemDataRole.ToolTipRole:
            axis = 1 if horizontal else 0
            if axis < self._da.ndim:
                return f"{self._da.dims[axis]}[{section}]"
        return None


class _Table(QTableView):
    """Table view with Ctrl+C copying the selection as tab-separated text."""

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Copy):
            self.copy_selection()
        else:
            super().keyPressEvent(event)

    def copy_selection(self):
        ranges = self.selectionModel().selection()
        if ranges.isEmpty():
            return
        top = min(r.top() for r in ranges)
        bottom = max(r.bottom() for r in ranges)
        left = min(r.left() for r in ranges)
        right = max(r.right() for r in ranges)
        if (bottom - top + 1) * (right - left + 1) > MAX_COPY_CELLS:
            QMessageBox.information(self, "Copy", "The selection is too large to copy; "
                                    "use Export CSV instead.")
            return
        model = self.model()
        selected = {(i.row(), i.column()) for i in self.selectionModel().selectedIndexes()}
        lines = []
        for r in range(top, bottom + 1):
            cells = (format_value(model.value(r, c)) if (r, c) in selected else ""
                     for c in range(left, right + 1))
            lines.append("\t".join(cells))
        QGuiApplication.clipboard().setText("\n".join(lines))


class TableView(DataView):
    title = "Table"

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self.rows = QComboBox(toolTip="Dimension shown down the rows")
        self.cols = QComboBox(toolTip="Dimension shown across the columns")
        self.swap = QPushButton("&Swap", toolTip="Swap rows and columns")
        self.export = QPushButton("&Export CSV…")
        self.info = QLabel()
        bar = QHBoxLayout()
        for w in (buddy_label("&Rows:", self.rows), self.rows, buddy_label("&Columns:", self.cols), self.cols,
                  self.swap):
            bar.addWidget(w)
        bar.addWidget(self.info, 1)
        bar.addWidget(self.export)
        self.model = LazyArrayModel(self)
        self.table = _Table()
        self.table.setModel(self.model)
        self.table.verticalHeader().setDefaultSectionSize(22)
        layout = QVBoxLayout(self.body)
        layout.setContentsMargins(4, 4, 4, 0)
        layout.addLayout(bar)
        layout.addWidget(self.table, 1)
        self.rows.currentTextChanged.connect(self._dims_picked)
        self.cols.currentTextChanged.connect(self._dims_picked)
        self.swap.clicked.connect(self._swap)
        self.export.clicked.connect(self._export)
        self._updating = False

    def free_dims(self):
        da = self.state.da
        if da is None or da.ndim == 0:
            return ()
        if da.ndim == 1:
            return (da.dims[0],)
        return (self.rows.currentText(), self.cols.currentText())

    def variable_changed(self):
        da = self.state.da
        self._updating = True
        dims = [] if da is None else [str(d) for d in da.dims]
        for combo in (self.rows, self.cols):
            combo.clear()
            combo.addItems(dims)
        if len(dims) >= 2:
            self.rows.setCurrentText(dims[-2])
            self.cols.setCurrentText(dims[-1])
        for w in (self.rows, self.cols, self.swap):
            w.setEnabled(len(dims) >= 2)
        self._updating = False

    def refresh(self):
        da = self.state.da
        free = self.free_dims()
        lazy = lazy_slice(da, free, self.state.indices)
        labels = [coord_values(da, d) for d in free]
        self.model.set_array(lazy, *labels)
        fixed = selection_text(da, fixed_indices(da, free, self.state.indices))
        shape = " × ".join(str(n) for n in lazy.shape) or "1"
        self.info.setText(f"  {shape} values" + (f"  at {fixed}" if fixed else ""))

    def _dims_picked(self):
        if self._updating:
            return
        if self.rows.currentText() == self.cols.currentText():
            # Keep the two different: move the other combo to a free dim.
            other = self.cols if self.sender() is self.rows else self.rows
            for i in range(other.count()):
                if other.itemText(i) != self.sender().currentText():
                    self._updating = True
                    other.setCurrentIndex(i)
                    self._updating = False
                    break
        self.freeDimsChanged.emit()

    def _swap(self):
        rows, cols = self.rows.currentText(), self.cols.currentText()
        self._updating = True
        self.rows.setCurrentText(cols)
        self.cols.setCurrentText(rows)
        self._updating = False
        self.freeDimsChanged.emit()

    def _export(self):
        da = self.state.da
        lazy = lazy_slice(da, self.free_dims(), self.state.indices)
        if lazy.size > LARGE_EXPORT_CELLS:
            answer = QMessageBox.question(self, "Export CSV", f"This slice has {lazy.size:,} values. "
                                          "Writing it may take a while. Continue?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", f"{da.name}.csv",
                                              "CSV files (*.csv)")
        if not path:
            return
        with wait_cursor():
            to_frame(lazy).to_csv(path)


def to_frame(lazy) -> pd.DataFrame:
    """The slice as a table: rows/columns labelled with coordinate values."""
    values = np.asarray(lazy.values)
    name = str(lazy.name)
    if lazy.ndim == 0:
        return pd.DataFrame({name: [values[()]]})
    index = _labels(lazy, 0)
    if lazy.ndim == 1:
        return pd.DataFrame({name: values}, index=index)
    return pd.DataFrame(values, index=index, columns=_labels(lazy, 1))


def _labels(lazy, axis):
    dim = lazy.dims[axis]
    values = coord_values(lazy, dim)
    return pd.Index(values if values is not None else np.arange(lazy.shape[axis]), name=dim)
