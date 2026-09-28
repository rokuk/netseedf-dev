"""The variable being looked at, shared by all views."""

import traceback

import xarray as xr
from PySide6.QtCore import QObject, Signal

from netseedf.core.coords import GeoInfo, find_geo
from netseedf.core.dataset import OpenedFile, VariableRef


class SelectionState(QObject):
    variableChanged = Signal()
    indicesChanged = Signal()  # position along one of the non-displayed dims

    def __init__(self, parent=None):
        super().__init__(parent)
        self.file: OpenedFile | None = None
        self.ref: VariableRef | None = None
        self.dataset: xr.Dataset | None = None
        self.da: xr.DataArray | None = None
        self.geo: GeoInfo | None = None
        self.indices: dict[str, int] = {}

    def set_variable(self, file: OpenedFile, ref: VariableRef):
        self.file, self.ref = file, ref
        self.dataset = file.dataset(ref.group)
        self.da = self.dataset[ref.name]
        try:
            self.geo = find_geo(self.da, self.dataset)
        except Exception:  # an odd coordinate shouldn't stop the other views
            traceback.print_exc()
            self.geo = None
        # self.indices is kept, so e.g. the time step survives switching variables
        # (it's clamped to each variable's size where it's used).
        self.variableChanged.emit()

    def clear(self):
        self.file = self.ref = self.dataset = self.da = self.geo = None
        self.indices = {}
        self.variableChanged.emit()

    def set_indices(self, changes: dict[str, int]):
        changes = {d: i for d, i in changes.items() if self.indices.get(d, 0) != i}
        if changes:
            self.indices.update(changes)
            self.indicesChanged.emit()
