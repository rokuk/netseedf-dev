"""Opening NetCDF files and describing what is in them."""

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr

ENGINE = "netcdf4"

FILE_FILTER = "NetCDF files (*.nc *.nc4 *.cdf *.netcdf *.h5 *.hdf5 *.he5);;All files (*)"

# Tried in order until one opens the file. Undecodable metadata (e.g. time
# units like "days since yesterday") makes xarray raise, so fall back to
# showing raw values instead of refusing the file.
_DECODE_ATTEMPTS = (
    ({}, None),
    ({"decode_times": False, "decode_timedelta": False},
     "Time values could not be decoded and are shown as raw numbers."),
    ({"decode_cf": False},
     "CF conventions could not be applied; all values are shown as stored in the file."),
)


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

    @property
    def name(self):
        return self.path.name

    def dataset(self, group="/"):
        return self.groups[group]

    def variable(self, group, name) -> xr.DataArray:
        return self.groups[group][name]

    def close(self):
        for closer in self._closers:
            closer.close()
        self._closers.clear()


def open_file(path) -> OpenedFile:
    """Open a NetCDF file lazily, including all of its groups."""
    path = Path(path)
    first_error = None
    for kwargs, note in _DECODE_ATTEMPTS:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                groups, closers = _open_groups(path, **kwargs)
        except (ValueError, TypeError, OverflowError) as e:
            first_error = first_error or e
            continue
        messages = [note] if note else []
        messages += _unique(str(w.message).splitlines()[0] for w in caught
                            if issubclass(w.category, (xr.SerializationWarning, RuntimeWarning)))
        return OpenedFile(path, groups, messages, closers)
    raise first_error


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
        if not self.dims:
            return "scalar"
        return "(" + ", ".join(f"{d}={n}" for d, n in zip(self.dims, self.shape, strict=True)) + ")"


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


# --- ncdump -h style header -------------------------------------------------

_CDL_TYPES = {
    "int8": "byte", "uint8": "ubyte", "int16": "short", "uint16": "ushort",
    "int32": "int", "uint32": "uint", "int64": "int64", "uint64": "uint64",
    "float32": "float", "float64": "double",
}


def header_text(path, group="/") -> str:
    """CDL description of the file (or one group), like ``ncdump -h``."""
    with netCDF4.Dataset(path) as nc:
        node = nc if group == "/" else nc[group.lstrip("/")]
        title = Path(path).stem if group == "/" else f"group: {node.name}"
        lines = [f"netcdf {title} {{" if group == "/" else f"{title} {{"]
        _cdl_group(node, lines, indent="")
        lines.append("}")
    return "\n".join(lines)


def variable_header_text(path, group, name) -> str:
    with netCDF4.Dataset(path) as nc:
        node = nc if group == "/" else nc[group.lstrip("/")]
        if name not in node.variables:  # e.g. an inherited coordinate
            for parent in _parents(node):
                if name in parent.variables:
                    node = parent
                    break
        lines = []
        _cdl_variable(node.variables[name], lines, "")
    return "\n".join(lines)


def _parents(node):
    while node.parent is not None:
        node = node.parent
        yield node


def _cdl_group(node, lines, indent):
    if node.dimensions:
        lines.append(f"{indent}dimensions:")
        for dim in node.dimensions.values():
            if dim.isunlimited():
                lines.append(f"{indent}\t{dim.name} = UNLIMITED ; // ({len(dim)} currently)")
            else:
                lines.append(f"{indent}\t{dim.name} = {len(dim)} ;")
    if node.variables:
        lines.append(f"{indent}variables:")
        for var in node.variables.values():
            _cdl_variable(var, lines, indent + "\t")
    attrs = node.ncattrs()
    if attrs:
        lines.append("")
        lines.append(f"{indent}// {'global' if node.parent is None else 'group'} attributes:")
        for key in attrs:
            lines.append(f"{indent}\t\t:{key} = {_cdl_value(node.getncattr(key))} ;")
    for child in node.groups.values():
        lines.append("")
        lines.append(f"{indent}group: {child.name} {{")
        _cdl_group(child, lines, indent + "  ")
        lines.append(f"{indent}  }} // group {child.name}")


def _cdl_variable(var, lines, indent):
    dtype = _cdl_type(var)
    dims = ", ".join(var.dimensions)
    lines.append(f"{indent}{dtype} {var.name}({dims}) ;" if dims else f"{indent}{dtype} {var.name} ;")
    for key in var.ncattrs():
        lines.append(f"{indent}\t{var.name}:{key} = {_cdl_value(var.getncattr(key))} ;")


def _cdl_type(var):
    if isinstance(var.datatype, (netCDF4.VLType, netCDF4.CompoundType, netCDF4.EnumType)):
        return var.datatype.name
    if var.dtype is str:
        return "string"
    dt = np.dtype(var.dtype)
    if dt.kind == "S" and dt.itemsize == 1:
        return "char"
    return _CDL_TYPES.get(dt.name, dt.name)


def _cdl_value(value):
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    arr = np.atleast_1d(value)
    return ", ".join(_cdl_scalar(v) for v in arr)


def _cdl_scalar(v):
    if isinstance(v, (np.floating, float)):
        text = "NaN" if np.isnan(v) else repr(float(v))
        return text + ("f" if isinstance(v, np.float32) else "")
    if isinstance(v, (bytes, np.bytes_)):
        return '"' + v.decode(errors="replace") + '"'
    return str(v)
