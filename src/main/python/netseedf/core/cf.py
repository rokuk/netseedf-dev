"""Parts of the CF conventions that xarray doesn't apply when it decodes a file.

Section numbers refer to https://cfconventions.org/cf-conventions/cf-conventions.html
"""

import posixpath
import re
from datetime import timedelta

import netCDF4
import numpy as np
import xarray as xr
from xarray.backends import BackendArray
from xarray.core import indexing

# --- 2.5.1 Missing data: valid range and default fill values -------------------------------


def mask_invalid(ds: xr.Dataset) -> xr.Dataset:
    """Values outside valid_min / valid_max / valid_range, or never written, as NaN.

    Values are masked lazily, as they're read.
    """
    updates = {}
    for name, var in ds.variables.items():
        if isinstance(var, xr.IndexVariable):  # coordinate variables can't have missing values
            continue
        mask = _invalid_mask(var)
        if mask is not None:
            func, dtype = mask
            data = indexing.LazilyIndexedArray(_MappedArray(var, func, dtype))
            updates[name] = xr.Variable(var.dims, data, var.attrs, var.encoding)
    if not updates:
        return ds
    coords = {k: v for k, v in updates.items() if k in ds.coords}
    return ds.assign_coords(coords).assign({k: v for k, v in updates.items() if k not in coords})


def _invalid_mask(var: xr.Variable):
    """(function turning read values into masked ones, dtype of the result), or None."""
    if var.dtype.kind not in "fiu":
        return None
    enc, attrs = var.encoding, var.attrs
    disk = np.dtype(enc.get("dtype", var.dtype))
    lo, hi = _valid_limits(attrs)
    # Unwritten values hold the default fill value, unless the variable says otherwise.
    # Bytes don't have one (NUG). Integers become floats to hold NaN, except where that
    # would lose more than it gains: flags are categories, int64 doesn't fit in a float,
    # and a grid mapping's value means nothing.
    fill = None
    if ("_FillValue" not in enc and "_FillValue" not in attrs
            and disk.kind in "fiu" and disk.itemsize > 1 and "_Unsigned" not in enc
            and (var.dtype.kind == "f" or _integer_data(var))):
        fill = netCDF4.default_fillvals.get(disk.str[1:])
    if lo is None and hi is None and fill is None:
        return None

    scale, offset = enc.get("scale_factor"), enc.get("add_offset")
    packed = scale is not None or offset is not None
    # 8.1: valid range attributes of the packed type are in packed units.
    limits_packed = packed and any(np.asarray(attrs[k]).dtype == disk
                                   for k in ("valid_range", "valid_min", "valid_max") if k in attrs)
    dtype = var.dtype if var.dtype.kind == "f" else np.dtype("f4" if var.dtype.itemsize <= 2 else "f8")

    def func(values):
        values = np.asarray(values)
        stored = values
        if packed and (limits_packed or fill is not None):
            stored = (values.astype("f8") - (offset or 0)) / (scale if scale is not None else 1)
            if disk.kind in "iu":
                stored = np.round(stored)
        compared = stored if limits_packed else values
        bad = np.zeros(values.shape, dtype=bool)
        with np.errstate(invalid="ignore"):
            if lo is not None:
                bad |= compared < lo
            if hi is not None:
                bad |= compared > hi
            if fill is not None:
                bad |= stored == fill
        out = values.astype(dtype, copy=True)
        out[bad] = np.nan
        return out

    return func, dtype


def _integer_data(var: xr.Variable) -> bool:
    """An integer variable holding measurements, whose unwritten values should be NaN."""
    return (var.dtype.kind in "iu" and var.dtype.itemsize < 8
            and not {"flag_values", "flag_masks", "grid_mapping_name"} & var.attrs.keys())


def _valid_limits(attrs):
    if "valid_range" in attrs:
        rng = np.ravel(attrs["valid_range"])
        if rng.size == 2:
            return rng[0], rng[1]
    return attrs.get("valid_min"), attrs.get("valid_max")


class _MappedArray(BackendArray):
    """A variable with `func` applied to whatever part of it is read."""

    def __init__(self, variable, func, dtype):
        self.variable, self.func = variable, func
        self.shape, self.dtype = variable.shape, np.dtype(dtype)

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(key, self.shape, indexing.IndexingSupport.OUTER,
                                                  self._getitem)

    def _getitem(self, key):
        return self.func(self.variable[key].values)


# --- 4.4 Time coordinates -------------------------------------------------------------------------

# Days on which TAI - UTC grew by one second (it was 10 s from 1972-01-01).
LEAP_SECONDS = np.array([
    "1972-07-01", "1973-01-01", "1974-01-01", "1975-01-01", "1976-01-01", "1977-01-01", "1978-01-01",
    "1979-01-01", "1980-01-01", "1981-07-01", "1982-07-01", "1983-07-01", "1985-07-01", "1988-01-01",
    "1990-01-01", "1991-01-01", "1992-07-01", "1993-07-01", "1994-07-01", "1996-01-01", "1997-07-01",
    "1999-01-01", "2006-01-01", "2009-01-01", "2012-07-01", "2015-07-01", "2017-01-01",
], dtype="datetime64[s]")


# The UTC offset at the end of a reference time (4.4): "-6:00", "+0530", "0:00" (as ARM
# writes UTC), "UTC", or "Z" straight after an ISO time.
_UTC_OFFSET = re.compile(
    r"^(?P<units>.*\bsince\s+\S+[ T]\d{1,2}:\d{1,2}(?::\d{1,2}(?:\.\d*)?)?)"
    r"(?:\s+(?P<offset>[+-]?\d{1,2}(?::?\d{2})?|UTC|GMT)|\s*Z)\s*$", re.IGNORECASE)


def split_utc_offset(units: str) -> tuple[str, int]:
    """Time units without their UTC offset, and the offset in minutes (east of UTC)."""
    m = _UTC_OFFSET.match(units.strip())
    if m is None:
        return units, 0
    offset = m["offset"] or "0"
    if offset.upper() in ("UTC", "GMT"):
        return m["units"], 0
    sign = -1 if offset.startswith("-") else 1
    digits = offset.lstrip("+-").replace(":", "")
    hours, minutes = (digits[:-2], digits[-2:]) if len(digits) > 2 else (digits, "0")
    return m["units"], sign * (int(hours) * 60 + int(minutes))


def decode_times(ds: xr.Dataset) -> tuple[xr.Dataset, list[str]]:
    """Decode the times of a file opened with decode_times=False, one variable at a time.

    Variables that can be decoded are, even if others can't. Unlike xarray,
    this handles UTC offsets in every calendar (xarray ignores them in cftime
    calendars, and misreads unsigned ones like "0:00") and the utc and tai
    calendars (CF 1.12). Returns the dataset and the names of the variables
    whose times couldn't be decoded.
    """
    attrs = {name: dict(var.attrs) for name, var in ds.variables.items()}
    for a in list(attrs.values()):  # 7.1: bounds take units and calendar from their variable
        bounds = a.get("bounds")
        if bounds in attrs and "units" not in attrs[bounds]:
            attrs[bounds].update({k: a[k] for k in ("units", "calendar") if k in a})
    updates, failed = {}, []
    for name, var in ds.variables.items():
        if " since " not in str(attrs[name].get("units", "")):
            continue
        var = var.copy(deep=False)  # still lazy
        var.attrs = attrs[name]
        try:
            updates[name] = _decode_time(var)
        except Exception:
            failed.append(str(name))
    coords = {k: v for k, v in updates.items() if k in ds.coords}
    ds = ds.assign_coords(coords).assign({k: v for k, v in updates.items() if k not in coords})
    return ds, failed


def _decode_time(var: xr.Variable) -> xr.Variable:
    calendar = str(var.attrs.get("calendar", "standard"))
    special = calendar.lower() in ("utc", "tai")
    original_units = var.attrs["units"]
    units, offset = split_utc_offset(str(original_units))
    attrs = {**var.attrs, "units": units}
    if special:
        # TAI has no leap seconds: its dates are those of the standard calendar. UTC's
        # elapsed times include leap seconds, which are taken off afterwards.
        attrs["calendar"] = "standard"
    var = var.copy(deep=False)
    var.attrs = attrs
    decoded = _decode_cf(var)
    utc = calendar.lower() == "utc" and decoded.dtype.kind == "M"
    if offset or utc:
        # Local time minus its offset is UTC.
        shift = np.timedelta64(offset, "m") if decoded.dtype.kind == "M" else timedelta(minutes=offset)
        values = decoded.values - shift
        if utc:
            reference = _decode_cf(xr.Variable(("n",), np.zeros(1), attrs)).values[0] - shift
            table = LEAP_SECONDS.astype(values.dtype)
            leaps = np.searchsorted(table, values, side="right") - np.searchsorted(table, reference,
                                                                                     side="right")
            values = values - leaps.astype("timedelta64[s]")
        decoded = xr.Variable(decoded.dims, values, decoded.attrs, decoded.encoding)
    decoded.encoding["units"] = original_units
    if special:
        decoded.encoding["calendar"] = calendar
    return decoded


def _decode_cf(var: xr.Variable) -> xr.Variable:
    ds = xr.Dataset({"t": var})
    return xr.decode_cf(ds, mask_and_scale=False, concat_characters=False, decode_coords=False,
                        decode_timedelta=False)["t"].variable


# --- 2.7 Groups ------------------------------------------------------------------------------------


def attach_group_coordinates(groups: dict[str, xr.Dataset]) -> dict[str, xr.Dataset]:
    """Add coordinates that variables name in other groups (2.7).

    Names in the coordinates attribute may be paths ("/lat", "../lat") or plain
    names, which are looked for in the variable's group, then its parent and so
    on up to the root (2.7.1 search by proximity). xarray only finds the latter
    in the variable's own group.
    """
    groups = dict(groups)
    for path in list(groups):  # parents come before their children
        ds = groups[path]
        extra = {}
        for var in ds.variables.values():
            refs = str(var.encoding.get("coordinates") or var.attrs.get("coordinates") or "").split()
            for ref in refs:
                name = posixpath.basename(ref)
                if name in ds.variables or name in extra:
                    continue
                found = _find_variable(groups, path, ref)
                if found is not None and all(ds.sizes.get(d) == n for d, n in found.sizes.items()):
                    extra[name] = found
        if extra:
            groups[path] = ds.assign_coords(extra)
    return groups


def _find_variable(groups, path, ref):
    if "/" in ref:
        full = posixpath.normpath(ref if ref.startswith("/") else posixpath.join(path, ref))
        group, name = posixpath.split(full)
        ds = groups.get(group)
        return None if ds is None else ds.variables.get(name)
    while True:
        ds = groups.get(path)
        if ds is not None and ref in ds.variables:
            return ds.variables[ref]
        if path == "/":
            return None
        path = posixpath.dirname(path)


# --- 9 Discrete sampling geometries ---------------------------------------------------------------

FEATURE_TYPES = {"point", "timeseries", "trajectory", "profile", "timeseriesprofile", "trajectoryprofile"}


def is_discrete_sampling(ds: xr.Dataset | None) -> bool:
    """Does the file hold features (stations, trajectories, ...) rather than grids?"""
    return ds is not None and str(ds.attrs.get("featureType", "")).lower() in FEATURE_TYPES
