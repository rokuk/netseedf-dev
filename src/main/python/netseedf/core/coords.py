"""Finding latitude/longitude for a variable and looking up values by position."""

from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from netseedf.core.slicing import Slice, as_float, extract, load_selection

LAT_NAMES = {"lat", "latitude", "nav_lat", "xlat", "xlat_m", "lat_rho", "gphit", "y_lat", "lats"}
LON_NAMES = {"lon", "long", "longitude", "nav_lon", "xlong", "xlong_m", "lon_rho", "glamt", "x_lon",
             "lons"}
LAT_UNITS = {"degreesnorth", "degreenorth", "degreesn", "degreen", "degn"}
LON_UNITS = {"degreeseast", "degreeeast", "degreese", "degreee", "dege"}

# Beyond this Web Mercator (and so Leaflet) can't show anything.
MERCATOR_MAX_LAT = 85.0511


@dataclass(frozen=True, eq=False)
class GeoInfo:
    """How a variable's dimensions map onto the globe."""

    kind: str  # "regular" (1D lat x 1D lon), "curvilinear" (2D lat/lon) or "points"
    lat: xr.DataArray
    lon: xr.DataArray
    dims: tuple[str, ...]  # data dims spanned by the grid: (y, x), or (point,)
    cache: dict = field(default_factory=dict, repr=False)  # full point coordinates, see point_window


def find_geo(da: xr.DataArray, ds: xr.Dataset | None = None) -> GeoInfo | None:
    lats, lons = [], []
    for name, var in _candidates(da, ds).items():
        role = _role(name, var)
        if role == "lat":
            lats.append(var)
        elif role == "lon":
            lons.append(var)
    dims = set(da.dims)
    for lat in lats:  # 1D lat and lon on different dims
        for lon in lons:
            if lat.ndim == lon.ndim == 1 and lat.dims != lon.dims and {*lat.dims, *lon.dims} <= dims:
                return GeoInfo("regular", lat, lon, (lat.dims[0], lon.dims[0]))
    for lat in lats:  # 2D (or more, e.g. WRF's XLAT(Time, y, x)) sharing the last two dims
        for lon in lons:
            if lat.ndim >= 2 and lon.ndim >= 2 and lat.dims[-2:] == lon.dims[-2:]:
                if set(lat.dims) <= dims and set(lon.dims) <= dims:
                    return GeoInfo("curvilinear", lat, lon, tuple(lat.dims[-2:]))
    for lat in lats:  # stations / swath points: lat and lon along the same dim
        for lon in lons:
            if lat.ndim == lon.ndim == 1 and lat.dims == lon.dims and lat.dims[0] in dims:
                return GeoInfo("points", lat, lon, lat.dims)
    return None


def _candidates(da, ds):
    found = {str(name): da.coords[name] for name in da.coords}
    if ds is not None:
        for name, var in ds.variables.items():
            if name != da.name and str(name) not in found and var.ndim >= 1 \
                    and set(var.dims) <= set(da.dims):
                found[str(name)] = var
    return {name: var for name, var in found.items() if var.dtype.kind in "fiu"}


def _role(name, var):
    standard_name = str(var.attrs.get("standard_name", "")).lower()
    if standard_name == "latitude":
        return "lat"
    if standard_name == "longitude":
        return "lon"
    if standard_name.startswith("grid_"):  # rotated-pole coordinates aren't geographic
        return None
    units = str(var.attrs.get("units", "")).lower().replace(" ", "").replace("_", "")
    if units in LAT_UNITS:
        return "lat"
    if units in LON_UNITS:
        return "lon"
    if name.lower() in LAT_NAMES:
        return "lat"
    if name.lower() in LON_NAMES:
        return "lon"
    return None


@dataclass
class GeoGrid:
    """A slice of data ready to be drawn on a map.

    regular:     lat (ny,) ascending, lon (nx,) ascending, values (ny, nx)
    curvilinear: lat, lon, values all (ny, nx)
    points:      lat, lon, values all (n,)
    """

    kind: str
    lat: np.ndarray
    lon: np.ndarray
    values: np.ndarray
    slice: Slice
    # Full-resolution index along each grid axis of every row/column (or point).
    index: tuple = ()
    lon_0_360: bool = False  # longitudes are in 0..360 rather than -180..180

    @property
    def bounds(self):
        """(south, west, north, east) of the grid cell centres."""
        return (float(np.nanmin(self.lat)), float(np.nanmin(self.lon)),
                float(np.nanmax(self.lat)), float(np.nanmax(self.lon)))

    @property
    def is_global(self):
        south, west, north, east = self.bounds
        return east - west > 300 and north - south > 120


def geo_grid(da: xr.DataArray, geo: GeoInfo, indices, max_size=None, window=None,
             like: GeoGrid | None = None) -> GeoGrid:
    """Load a map-ready slice; `window` restricts it to part of the grid (see geo_window).

    `like` makes the longitudes use the same convention as another grid, so a
    zoomed-in detail lines up with the overview it's drawn over.
    """
    sl = extract(da, geo.dims, indices, max_size, window)
    values = as_float(sl.values)
    if values is None:
        raise ValueError(f"'{da.name}' isn't numeric ({da.dtype}), so it can't be drawn on a map.")
    lat = _coord_slice(geo.lat, sl)
    lon, lon_0_360 = normalize_lon(_coord_slice(geo.lon, sl), like.lon_0_360 if like else None)
    index = tuple(sl.index(axis) for axis in range(len(sl.dims)))
    if geo.kind == "regular":
        rows, cols = np.argsort(lat, kind="stable"), np.argsort(lon, kind="stable")
        lat, lon, values = lat[rows], lon[cols], values[np.ix_(rows, cols)]
        index = (index[0][rows], index[1][cols])
    return GeoGrid(geo.kind, lat, lon, values, sl, index, lon_0_360)


def with_cyclic_column(grid: GeoGrid) -> GeoGrid:
    """Repeat the first column 360° east when a regular grid wraps the globe.

    Without it, drawing leaves a gap of half a cell at the antimeridian.
    """
    if grid.kind != "regular" or grid.lon.size < 2:
        return grid
    step = np.median(np.diff(grid.lon))
    if grid.lon[-1] - grid.lon[0] + step < 359.99 or grid.lon[-1] + step - 360 > grid.lon[0] + 1e-6:
        return grid
    lon = np.append(grid.lon, grid.lon[0] + 360)
    values = np.concatenate([grid.values, grid.values[:, :1]], axis=1)
    index = (grid.index[0], np.append(grid.index[1], grid.index[1][0])) if grid.index else ()
    return GeoGrid(grid.kind, grid.lat, lon, values, grid.slice, index, grid.lon_0_360)


def _coord_slice(var: xr.DataArray, sl: Slice) -> np.ndarray:
    # Extra dims of the coordinate (e.g. Time on WRF's XLAT) use the data's indices.
    extra = {d: min(sl.fixed.get(d, 0), var.sizes[d] - 1) for d in var.dims if d not in sl.dims}
    sub = var.isel(extra).transpose(*[d for d in sl.dims if d in var.dims])
    selection = [sl.selection[sl.dims.index(d)] for d in sub.dims]
    return np.asarray(load_selection(sub, selection), dtype=float)


def in_box(lat, lon, box):
    """Mask of positions inside (south, west, north, east); longitudes wrap."""
    south, west, north, east = box
    lat, lon = np.broadcast_arrays(lat, lon)
    inside = (lat >= south) & (lat <= north)
    if east - west < 360:
        inside &= (lon - west) % 360 <= east - west
    return inside


def geo_window(geo: GeoInfo, grid: GeoGrid, box) -> dict | None:
    """Full-resolution index window covering the part of `grid` inside `box`.

    `box` is (south, west, north, east), e.g. the visible part of a map. The
    result can be passed to geo_grid() to load that part in more detail.
    """
    if grid.kind == "points":
        return point_window(geo, grid, box)
    inside = in_box(grid.lat[:, None] if grid.kind == "regular" else grid.lat,
                    grid.lon[None, :] if grid.kind == "regular" else grid.lon, box)
    rows, cols = np.flatnonzero(inside.any(axis=1)), np.flatnonzero(inside.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    window = {}
    for dim, idx, step in zip(grid.slice.dims, (grid.index[0][rows], grid.index[1][cols]),
                              grid.slice.steps, strict=True):
        # One coarse step extra: cells cut by the edge, and ones the stride skipped.
        window[dim] = (int(idx.min()) - step, int(idx.max()) + step + 1)
    return window


def point_window(geo: GeoInfo, grid: GeoGrid, box) -> dict | None:
    """Indices of all points inside `box` (not just the ones in the thinned grid)."""
    sl = grid.slice
    key = tuple(sorted((d, i) for d, i in sl.fixed.items() if d in geo.lat.dims + geo.lon.dims))
    if key not in geo.cache:
        dim = sl.dims[0]
        full = Slice(np.empty(0), sl.dims, (1,), sl.fixed, (slice(0, geo.lat.sizes[dim], 1),))
        geo.cache.clear()  # keep only the current time step's coordinates
        geo.cache[key] = (_coord_slice(geo.lat, full),
                          normalize_lon(_coord_slice(geo.lon, full), grid.lon_0_360)[0])
    lat, lon = geo.cache[key]
    idx = np.flatnonzero(in_box(lat, lon, box))
    return {sl.dims[0]: idx} if idx.size else None


def normalize_lon(lon: np.ndarray, lon_0_360=None) -> tuple[np.ndarray, bool]:
    """Longitudes in -180..180 or 0..360, and whether it's the latter.

    Without `lon_0_360`, picks whichever keeps the data contiguous: a Pacific
    box stored as 150..210 stays that way instead of being split across the
    map; global grids become -180..180.
    """
    west = (lon + 180) % 360 - 180
    east = lon % 360
    if lon_0_360 is None:
        lon_0_360 = bool(_span(east) < _span(west))
    return (east if lon_0_360 else west), lon_0_360


def _span(a):
    return np.nanmax(a) - np.nanmin(a) if a.size else 0.0


def lon_diff(a, b):
    """Signed difference a - b in degrees, wrapped into [-180, 180)."""
    return (np.asarray(a) - b + 180) % 360 - 180


class GridLocator:
    """Finds the grid cell under a lat/lon position (e.g. the mouse)."""

    def __init__(self, grid: GeoGrid):
        self.grid = grid
        if grid.kind != "regular":
            self._tolerance = self._typical_spacing() if grid.kind == "curvilinear" \
                else max(0.02 * float(np.hypot(*_extent(grid))), 0.05)

    def nearest(self, lat, lon):
        """Index into grid.values, or None when outside the data."""
        g = self.grid
        if not (np.isfinite(lat) and np.isfinite(lon)):
            return None
        if g.kind == "regular":
            i = _nearest_1d(g.lat, lat)
            j = _nearest_1d(g.lon, lon, wrap=True)
            return None if i is None or j is None else (i, j)
        d2 = (g.lat - lat) ** 2 + (lon_diff(g.lon, lon) * np.cos(np.deg2rad(lat))) ** 2
        if not np.isfinite(d2).any():
            return None
        k = int(np.nanargmin(d2))
        if d2.flat[k] > self._tolerance ** 2:
            return None
        return np.unravel_index(k, g.values.shape)

    def value_at(self, lat, lon):
        idx = self.nearest(lat, lon)
        return None if idx is None else (idx, self.grid.values[idx])

    def _typical_spacing(self):
        lat, lon = self.grid.lat, self.grid.lon
        steps = [np.hypot(np.diff(lat, axis=a), lon_diff(np.diff(lon, axis=a), 0)) for a in (0, 1)
                 if lat.shape[a] > 1]
        spacing = np.nanmedian(np.concatenate([s.ravel() for s in steps])) if steps else 1.0
        return float(spacing) if np.isfinite(spacing) and spacing > 0 else 1.0


def _extent(grid):
    south, west, north, east = grid.bounds
    return north - south, east - west


def _nearest_1d(coord, q, wrap=False):
    diff = lon_diff(coord, q) if wrap else coord - q
    i = int(np.argmin(np.abs(diff)))
    if len(coord) > 1:
        neighbours = [abs(coord[k] - coord[i]) for k in (i - 1, i + 1) if 0 <= k < len(coord)]
        half_cell = 0.5 * min(neighbours) * 1.0001
    else:
        half_cell = 0.5
    return i if abs(diff[i]) <= half_cell else None
