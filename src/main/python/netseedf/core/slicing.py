"""Cutting 1D/2D slices out of N-dimensional variables."""

import math
from dataclasses import dataclass

import numpy as np
import xarray as xr

# Strided reads can be slow in netCDF-C, so slices up to this many elements
# are read contiguously and thinned in memory.
CONTIGUOUS_READ_LIMIT = 50_000_000


@dataclass
class Slice:
    values: np.ndarray  # dims in the order of `dims`
    dims: tuple[str, ...]
    steps: tuple[int, ...]  # downsampling stride per dim, 1 = full resolution
    fixed: dict[str, int]  # index used for every other dimension
    # Which full-resolution indices `values` holds along each dim: a slice
    # (start, stop, step) or, for a hand-picked set of points, an index array.
    selection: tuple = ()

    @property
    def downsampled(self):
        return any(s > 1 for s in self.steps)

    @property
    def max_step(self):
        return max(self.steps, default=1)

    def index(self, axis) -> np.ndarray:
        """Full-resolution indices of the values along `axis`."""
        sel = self.selection[axis]
        return np.arange(sel.start, sel.stop, sel.step) if isinstance(sel, slice) else sel


def fixed_indices(da: xr.DataArray, free_dims, indices) -> dict[str, int]:
    """Index for every dim that isn't shown, clamped into range (default 0)."""
    return {
        d: min(max(int(indices.get(d, 0)), 0), da.sizes[d] - 1)
        for d in da.dims
        if d not in free_dims
    }


def lazy_slice(da: xr.DataArray, free_dims, indices) -> xr.DataArray:
    """The not-yet-loaded sub-array with only `free_dims` left, in that order."""
    return da.isel(fixed_indices(da, free_dims, indices)).transpose(*free_dims)


def extract(da: xr.DataArray, free_dims, indices, max_size=None, window=None) -> Slice:
    """Load a slice with at most `max_size` values along each free dim.

    `window` limits dims to part of their range, e.g. what's visible after
    zooming in: {dim: (start, stop)} or {dim: array of indices}. The window
    is thinned only as far as needed to fit `max_size`.
    """
    free_dims = tuple(free_dims)
    fixed = fixed_indices(da, free_dims, indices)
    sub = da.isel(fixed).transpose(*free_dims)
    selection, steps = [], []
    for d, n in zip(free_dims, sub.shape, strict=True):
        sel, step = _selection(n, (window or {}).get(d), max_size)
        selection.append(sel)
        steps.append(step)
    return Slice(load_selection(sub, selection), free_dims, tuple(steps), fixed, tuple(selection))


def _selection(n, window, max_size):
    if isinstance(window, np.ndarray):
        step = max(1, math.ceil(len(window) / max_size)) if max_size else 1
        return window[::step], step
    start, stop = window if window is not None else (0, n)
    start = min(max(int(start), 0), max(n - 1, 0))
    stop = min(max(int(stop), start + 1), n)
    step = max(1, math.ceil((stop - start) / max_size)) if max_size else 1
    return slice(start, stop, step), step


def load_selection(sub: xr.DataArray, selection) -> np.ndarray:
    """Read the selected indices of a lazy array (one entry of `selection` per dim)."""
    box, local = {}, []
    for dim, sel in zip(sub.dims, selection, strict=True):
        if isinstance(sel, slice):
            box[dim] = slice(sel.start, sel.stop)
            local.append(slice(None, None, sel.step))
        else:
            lo = int(sel.min()) if len(sel) else 0
            box[dim] = slice(lo, int(sel.max()) + 1 if len(sel) else 0)
            local.append(sel - lo)
    sub = sub.isel(box)
    if sub.size <= CONTIGUOUS_READ_LIMIT or all(isinstance(s, slice) and s.step == 1 for s in local):
        return np.asarray(sub.values)[tuple(local)]
    return np.asarray(sub.isel(dict(zip(sub.dims, local, strict=True))).values)


def coord_values(da: xr.DataArray, dim, sel=slice(None)) -> np.ndarray | None:
    """Values of the 1D coordinate along `dim` (at `sel`), if the variable has one."""
    if dim not in da.coords or da.coords[dim].dims != (dim,):
        return None
    return np.asarray(da.coords[dim].values)[sel]


def window_steps(window, max_size) -> dict[str, int]:
    """Stride each dim of `window` would need to fit `max_size`."""
    return {d: max(1, math.ceil((len(w) if isinstance(w, np.ndarray) else w[1] - w[0]) / max_size))
            for d, w in window.items()}


def is_finer(base: Slice, window, max_size) -> bool:
    """Would loading `window` show more detail than `base` along some dim?"""
    steps = window_steps(window, max_size)
    return any(steps[d] < s for d, s in zip(base.dims, base.steps, strict=True) if d in steps)


def index_window(positions, lo, hi):
    """(start, stop) of the positions within [lo, hi], plus one on each side."""
    lo, hi = min(lo, hi), max(lo, hi)
    inside = np.flatnonzero((positions >= lo) & (positions <= hi))
    if inside.size == 0:
        return None
    return max(int(inside[0]) - 1, 0), min(int(inside[-1]) + 2, len(positions))


def pad_range(lo, hi, fraction):
    pad = (hi - lo) * fraction
    return lo - pad, hi + pad


@dataclass
class Stats:
    count: int
    missing: int
    minimum: float
    maximum: float
    mean: float


def variable_stats(da: xr.DataArray, block_elements=10_000_000) -> Stats:
    """Min/max/mean over the whole variable, read in blocks along the first dim."""
    if da.ndim == 0:
        blocks = [np.atleast_1d(da.values)]
    else:
        per_row = max(1, da.size // max(da.shape[0], 1))
        rows = max(1, block_elements // per_row)
        dim = da.dims[0]
        blocks = (da.isel({dim: slice(i, i + rows)}).values for i in range(0, da.shape[0], rows))
    count = missing = 0
    lo, hi, total = np.inf, -np.inf, 0.0
    for block in blocks:
        values = as_float(np.asarray(block))
        if values is None:
            raise ValueError(f"Statistics need numeric data, not {da.dtype}")
        finite = values[np.isfinite(values)]
        count += finite.size
        missing += values.size - finite.size
        if finite.size:
            lo, hi = min(lo, finite.min()), max(hi, finite.max())
            total += finite.sum(dtype=float)
    nan = float("nan")
    return Stats(count, missing, lo if count else nan, hi if count else nan,
                 total / count if count else nan)


def as_float(values: np.ndarray) -> np.ndarray:
    """Numeric data as floats with NaN for missing; None for non-numeric data."""
    if values.dtype.kind in "fc":
        return values.real.astype(float, copy=False)
    if values.dtype.kind in "iub":
        return values.astype(float)
    return None
