"""Tree of open files, their groups and variables."""

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem

from netseedf.core.dataset import OpenedFile, VariableRef, describe

ROLE = Qt.ItemDataRole.UserRole


@dataclass(frozen=True)
class Node:
    kind: str  # "file", "group", "folder" or "variable"
    file_id: int
    group: str = "/"
    name: str = ""


class VariableTree(QTreeWidget):
    fileSelected = Signal(int, str)  # file id, group path
    variableSelected = Signal(VariableRef)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["Name", "Dimensions"])
        self.setColumnWidth(0, 170)
        self.setUniformRowHeights(True)
        self.currentItemChanged.connect(self._current_changed)

    def add_file(self, file_id, opened: OpenedFile) -> QTreeWidgetItem:
        item = QTreeWidgetItem([opened.name, ""])
        item.setData(0, ROLE, Node("file", file_id))
        item.setToolTip(0, str(opened.path))
        bold = QFont(item.font(0))
        bold.setBold(True)
        item.setFont(0, bold)
        self.addTopLevelItem(item)
        group_items = {"/": item}
        for path, ds in opened.groups.items():
            parent_path = path.rsplit("/", 1)[0] or "/"
            if path != "/":
                group_item = QTreeWidgetItem([path.rsplit("/", 1)[1], "group"])
                group_item.setData(0, ROLE, Node("group", file_id, path))
                group_items[parent_path].addChild(group_item)
                group_items[path] = group_item
            parent_ds = opened.groups.get(parent_path) if path != "/" else None
            self._add_variables(group_items[path], file_id, path, ds, parent_ds)
        item.setExpanded(True)
        return item

    def _add_variables(self, parent, file_id, group, ds, parent_ds):
        for name in ds.data_vars:
            parent.addChild(self._variable_item(file_id, group, ds[name]))
        # Coordinates inherited from the parent group are listed there instead.
        own = [c for c in ds.coords if parent_ds is None or c not in parent_ds.coords]
        if own:
            folder = QTreeWidgetItem(["Coordinates", ""])
            folder.setData(0, ROLE, Node("folder", file_id, group))
            for name in own:
                folder.addChild(self._variable_item(file_id, group, ds[name]))
            parent.addChild(folder)

    @staticmethod
    def _variable_item(file_id, group, da):
        info = describe(da)
        item = QTreeWidgetItem([info.name, "(" + ", ".join(info.dims) + ")" if info.dims else "scalar"])
        item.setData(0, ROLE, Node("variable", file_id, group, info.name))
        tip = [f"<b>{info.name}</b>"]
        if info.long_name:
            tip.append(info.long_name)
        tip.append(f"{info.dims_text}, {info.dtype}")
        if info.units:
            tip.append(f"units: {info.units}")
        item.setToolTip(0, "<br>".join(tip))
        item.setToolTip(1, info.dims_text)
        return item

    def remove_file(self, file_id):
        for i in range(self.topLevelItemCount()):
            if self.topLevelItem(i).data(0, ROLE).file_id == file_id:
                self.takeTopLevelItem(i)
                return

    def file_item(self, file_id):
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            if item.data(0, ROLE).file_id == file_id:
                return item
        return None

    def current_file_id(self):
        item = self.currentItem()
        return item.data(0, ROLE).file_id if item is not None else None

    def variable_items(self, file_id):
        """All variable items of a file, data variables first."""
        root = self.file_item(file_id)
        found = []
        stack = [root] if root is not None else []
        while stack:
            item = stack.pop(0)
            for i in range(item.childCount()):
                child = item.child(i)
                if child.data(0, ROLE).kind == "variable":
                    found.append(child)
                else:
                    stack.append(child)
        return found

    def _current_changed(self, item, _previous):
        if item is None:
            return
        node: Node = item.data(0, ROLE)
        if node.kind == "variable":
            self.variableSelected.emit(VariableRef(node.file_id, node.group, node.name))
        else:
            self.fileSelected.emit(node.file_id, node.group)
