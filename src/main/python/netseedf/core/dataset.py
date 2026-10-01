"""Opening NetCDF files and describing what is in them."""

import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr

from netseedf.core import cf

ENGINE = "netcdf4"

# netCDF-C and HDF5 aren't thread-safe. xarray serializes reading values, but not opening
# files and reading their headers; whatever does those holds this, and so does every read
# in the background (see DataView), so neither happens while the other does.
FILE_LOCK = threading.RLock()

FILE_FILTER = "NetCDF files (*.nc *.nc4 *.cdf *.netcdf *.h5 *.hdf5 *.he5);;All files (*)"

# Tried in order until one opens the file. Times are decoded by cf.decode_times, one
# variable at a time (see there why not by xarray). Should CF decoding fail
# anyway, show values as stored in the file instead of refusing it.
_DECODE_ATTEMPTS = (
    {"decode_times": False, "decode_timedelta": False},
    {"decode_cf": False},
)
_NO_CF = "CF conventions could not be applied; all values are shown as stored in the file."
# xarray warnings that aren't news to the user
_QUIET = ("has multiple fill values", "Ambiguous reference date string")


@dataclass(frozen=True)
class VariableRef:
    """Identifies a variable: which open file, which group, which name."""

    file_id: int
    group: str
    name: str


@dataclass
class OpenedFile:
    path: Path
    groups: dict[str, xr.Dataset]  # group path ("/", "/a", "/a/b") -> dataset
    warnings: list[str] = field(default_factory=list)
    _closers: list = field(default_factory=list, repr=False)
    # Read when opening: the GUI mustn't wait for the file while it's read in the background.
    headers: dict[str, "GroupHeader"] = field(default_factory=dict, repr=False)

    @property
    def name(self):
        return self.path.name

    def dataset(self, group="/"):
        return self.groups[group]

    def variable(self, group, name) -> xr.DataArray:
        return self.groups[group][name]

    def group_header(self, group="/") -> "GroupHeader":
        """See group_header()."""
        if group not in self.headers:
            self.headers[group] = group_header(self.path, group)
        return self.headers[group]

    def variable_header(self, group, name) -> "VariableHeader | None":
        """See variable_header()."""
        while True:
            for var in self.group_header(group).variables:
                if var.name == name:
                    return var
            if group == "/":
                return None
            group = group.rsplit("/", 1)[0] or "/"  # e.g. an inherited coordinate

    def close(self):
        with FILE_LOCK:  # not while a background read uses the file (it would open it again)
            for closer in self._closers:
                closer.close()
            self._closers.clear()


def open_file(path) -> OpenedFile:
    """Open a NetCDF file lazily, including all of its groups."""
    with FILE_LOCK:
        return _open_file(Path(path))


def _open_file(path):
    first_error = None
    for kwargs in _DECODE_ATTEMPTS:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                groups, closers = _open_groups(path, **kwargs)
                groups = {group: _distinct_dims(ds) for group, ds in groups.items()}
                groups, messages = _apply_cf(groups, **kwargs)
        except (ValueError, TypeError, OverflowError) as e:
            first_error = first_error or e
            continue
        lines = (str(w.message).splitlines()[0] for w in caught
                 if issubclass(w.category, (xr.SerializationWarning, RuntimeWarning)))
        messages += _unique(line for line in lines if not any(q in line for q in _QUIET))
        return OpenedFile(path, groups, messages, closers, file_headers(path))
    raise first_error


def _apply_cf(groups, decode_cf=True, **_):
    """What xarray leaves out of CF decoding; returns the groups and notes for the user."""
    if not decode_cf:
        return groups, [_NO_CF]
    messages, undecoded = [], []
    for group, ds in groups.items():
        groups[group], failed = cf.decode_times(ds)
        undecoded += failed
    if undecoded:
        names = ", ".join(_unique(undecoded))
        messages.append(f"Time values could not be decoded and are shown as raw numbers ({names}).")
    groups = {group: cf.mask_invalid(ds) for group, ds in groups.items()}
    return cf.attach_group_coordinates(groups), messages


def _distinct_dims(ds: xr.Dataset) -> xr.Dataset:
    """Repeated dimensions, e.g. of a matrix AVK(time, PRESSURE, PRESSURE), under names of their own.

    xarray selects by dimension name, so it can't slice such variables. The
    second PRESSURE becomes PRESSURE_2, with the same coordinate.
    """
    updates, coords = {}, {}
    for name, var in ds.variables.items():
        if len(set(var.dims)) == len(var.dims):
            continue
        dims, seen = [], {}
        for d in var.dims:
            seen[d] = seen.get(d, 0) + 1
            new = d if seen[d] == 1 else f"{d}_{seen[d]}"
            dims.append(new)
            if new != d and new not in coords and d in ds.indexes:
                coords[new] = xr.Variable((new,), ds[d].values, ds[d].attrs)
        renamed = var.copy(deep=False)  # still lazy
        renamed.dims = tuple(dims)
        updates[name] = renamed
    if not updates:
        return ds
    ds = ds.drop_vars(list(updates))
    return ds.assign_coords(coords).assign(updates)


def _open_groups(path, **kwargs):
    try:
        tree = xr.open_datatree(path, engine=ENGINE, **kwargs)
    except ValueError as e:
        if "not aligned" not in str(e):
            raise
        # Child groups whose dimensions clash with their parents can't form a
        # DataTree; open them independently (no coordinate inheritance).
        groups = xr.open_groups(path, engine=ENGINE, **kwargs)
        return groups, list(groups.values())
    groups = {node.path: node.to_dataset(inherit=True) for node in tree.subtree}
    return groups, [tree]


def _unique(items):
    return list(dict.fromkeys(items))


@dataclass(frozen=True)
class VariableInfo:
    name: str
    dims: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    long_name: str
    units: str

    @property
    def dims_text(self):
        return _dims_text(self.dims, self.shape)


def _dims_text(dims, shape):
    if not dims:
        return "scalar"
    return "(" + ", ".join(f"{d}={n}" for d, n in zip(dims, shape, strict=True)) + ")"


def describe(da: xr.DataArray) -> VariableInfo:
    attrs = da.attrs
    return VariableInfo(
        name=str(da.name),
        dims=tuple(str(d) for d in da.dims),
        shape=tuple(da.shape),
        dtype=str(da.dtype),
        long_name=str(attrs.get("long_name") or attrs.get("description") or attrs.get("standard_name") or ""),
        units=str(attrs.get("units", "")),
    )


def data_variables(ds: xr.Dataset):
    return [describe(ds[name]) for name in ds.data_vars]


def coordinate_variables(ds: xr.Dataset):
    return [describe(ds[name]) for name in ds.coords]


# --- the header as stored in the file ----------------------------------------

_CDL_TYPES = {
    "int8": "byte", "uint8": "ubyte", "int16": "short", "uint16": "ushort",
    "int32": "int", "uint32": "uint", "int64": "int64", "uint64": "uint64",
    "float32": "float", "float64": "double",
}


@dataclass(frozen=True)
class DimensionHeader:
    name: str
    size: int
    unlimited: bool


@dataclass(frozen=True)
class VariableHeader:
    name: str
    dtype: str  # CDL type on disk, e.g. "short" for packed data
    dims: tuple[str, ...]
    shape: tuple[int, ...]
    attrs: dict

    @property
    def dims_text(self):
        return _dims_text(self.dims, self.shape)


@dataclass(frozen=True)
class GroupHeader:
    dimensions: list[DimensionHeader]
    variables: list[VariableHeader]
    attrs: dict  # global (or group) attributes


def group_header(path, group="/") -> GroupHeader:
    """Dimensions, variables and attributes of the file (or one group), as ``ncdump -h`` lists them."""
    with FILE_LOCK, netCDF4.Dataset(path) as nc:
        return _group_header(_node(nc, group))


def file_headers(path) -> dict[str, GroupHeader]:
    """group_header() of every group in the file, by group path."""
    with FILE_LOCK, netCDF4.Dataset(path) as nc:
        headers, todo = {}, [nc]
        while todo:
            node = todo.pop()
            headers[node.path] = _group_header(node)
            todo += node.groups.values()
        return headers


def _group_header(node):
    return GroupHeader(
        dimensions=[DimensionHeader(d.name, len(d), d.isunlimited()) for d in node.dimensions.values()],
        variables=[_variable_header(v) for v in node.variables.values()],
        attrs=_attrs(node),
    )


def variable_header(path, group, name) -> VariableHeader | None:
    """The variable as stored, with all of its attributes (xarray moves some to ``encoding``).

    None if the file has no such variable (one netseedf made up).
    """
    with FILE_LOCK, netCDF4.Dataset(path) as nc:
        node = _node(nc, group)
        for n in (node, *_parents(node)):  # e.g. an inherited coordinate
            if name in n.variables:
                return _variable_header(n.variables[name])
    return None


def _node(nc, group):
    return nc if group == "/" else nc[group.lstrip("/")]


def _parents(node):
    while node.parent is not None:
        node = node.parent
        yield node


def _variable_header(var):
    return VariableHeader(var.name, _cdl_type(var), tuple(var.dimensions), tuple(var.shape), _attrs(var))


def _attrs(obj):
    return {key: obj.getncattr(key) for key in obj.ncattrs()}


def _cdl_type(var):
    if isinstance(var.datatype, (netCDF4.VLType, netCDF4.CompoundType, netCDF4.EnumType)):
        return var.datatype.name
    if var.dtype is str:
        return "string"
    dt = np.dtype(var.dtype)
    if dt.kind == "S" and dt.itemsize == 1:
        return "char"
    return _CDL_TYPES.get(dt.name, dt.name)
