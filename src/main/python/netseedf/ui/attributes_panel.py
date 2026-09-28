"""Attributes of whatever is selected in the tree."""

from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from netseedf.core.formatting import format_value


class AttributesPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.heading = QLabel()
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Attribute", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.addWidget(self.heading)
        layout.addWidget(self.table, 1)
        self.show_attrs("", {})

    def show_attrs(self, heading, attrs):
        self.heading.setText(f"<b>Attributes</b> {heading}")
        self.table.setRowCount(len(attrs))
        for row, (key, value) in enumerate(attrs.items()):
            text = value if isinstance(value, str) else ", ".join(
                format_value(v) for v in _as_list(value))
            key_item, value_item = QTableWidgetItem(str(key)), QTableWidgetItem(text)
            value_item.setToolTip(text)
            self.table.setItem(row, 0, key_item)
            self.table.setItem(row, 1, value_item)
        self.table.resizeRowsToContents()


def _as_list(value):
    try:
        return list(value) if not isinstance(value, (bytes, str)) else [value]
    except TypeError:
        return [value]
