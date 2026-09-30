"""Turning values, times and variables into short human-readable text."""

import numpy as np
import pandas as pd


def format_value(v, precision=6) -> str:
    if v is None:
        return ""
    if isinstance(v, np.ndarray) and v.ndim == 0:
        v = v[()]
    if isinstance(v, np.datetime64):
        return format_time(pd.Timestamp(v)) if not np.isnat(v) else "NaT"
    if isinstance(v, np.timedelta64):
        return str(pd.Timedelta(v)) if not np.isnat(v) else "NaT"
    if isinstance(v, pd.Timestamp):
        return format_time(v)
    if _is_cftime(v):
        return format_time(v)
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode(errors="replace")
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "NaN"
        return f"{v:.{precision}g}"
    return str(v)


def format_time(t) -> str:
    """Date only for midnight; seconds only when they're non-zero."""
    if (t.hour, t.minute, t.second) == (0, 0, 0) and not getattr(t, "microsecond", 0):
        return t.strftime("%Y-%m-%d")
    if t.second == 0:
        return t.strftime("%Y-%m-%d %H:%M")
    return t.strftime("%Y-%m-%d %H:%M:%S")


def _is_cftime(v):
    return type(v).__module__.startswith("cftime")


def is_time_like(values: np.ndarray) -> bool:
    if values.dtype.kind == "M":
        return True
    return values.dtype == object and values.size > 0 and _is_cftime(values.flat[0])


# Spellings of Celsius (lowercased, spaces and underscores removed) shown as °C.
CELSIUS_UNITS = {"degreecelsius", "degreescelsius", "degc", "degreec", "degreesc", "celsius"}


def display_units(units) -> str:
    """Units as shown on plots and maps: degree_Celsius and friends become °C."""
    if not units:
        return ""
    units = str(units)
    return "°C" if units.lower().replace(" ", "").replace("_", "") in CELSIUS_UNITS else units


def variable_label(da) -> str:
    """'Long name [units]' for axis and colorbar labels."""
    name = da.attrs.get("long_name") or da.attrs.get("standard_name") or str(da.name)
    units = display_units(da.attrs.get("units"))
    return f"{name} [{units}]" if units else str(name)


def dim_label(da, dim) -> str:
    if dim in da.coords:
        return variable_label(da.coords[dim])
    return str(dim)


def index_label(da, dim, index) -> str:
    """'2024-01-05' / '5 m' for position `index` along `dim` (or just the index)."""
    if dim in da.coords and da.coords[dim].dims == (dim,):
        coord = da.coords[dim]
        text = format_value(coord.values[index])
        units = display_units(coord.attrs.get("units"))
        if units and not is_time_like(coord.values[:1]):
            text += f" {units}"
        return text
    return f"#{index}"


def selection_text(da, fixed) -> str:
    """'time = 2024-01-05, depth = 5 m' for the dims held fixed in a slice."""
    return ", ".join(f"{d} = {index_label(da, d, i)}" for d, i in fixed.items() if da.sizes[d] > 1)


def position_text(lat, lon) -> str:
    ns = "N" if lat >= 0 else "S"
    lon = (lon + 180) % 360 - 180
    ew = "E" if lon >= 0 else "W"
    return f"{abs(lat):.3f}°{ns}, {abs(lon):.3f}°{ew}"


def value_text(da, value) -> str:
    units = display_units(da.attrs.get("units"))
    text = format_value(value)
    return f"{text} {units}" if units and text not in ("NaN", "") else text
