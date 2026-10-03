"""Finding latitude/longitude for a variable and looking up values by position."""

from dataclasses import dataclass, field

import numpy as np
import pyproj
import xarray as xr

from netseedf.core.cf import is_discrete_sampling
from netseedf.core.slicing import Slice, as_float, extract, load_selection

LAT_NAMES = {"lat", "latitude", "nav_lat", "xlat", "xlat_m", "lat_rho", "gphit", "y_lat", "lats"}
LON_NAMES = {"lon", "long", "longitude", "nav_lon", "xlong", "xlong_m", "lon_rho", "glamt", "x_lon",
             "lons"}
LAT_UNITS = {"degreesnorth", "degreenorth", "degreesn", "degreen", "degn"}
LON_UNITS = {"degreeseast", "degreeeast", "degreese", "degreee", "dege"}

# Beyond this Web Mercator (and so Leaflet) can't show anything.
MERCATOR_MAX_LAT = 85.0511
WRAPS_AROUND = 350.0  # degrees of longitude: data spanning more goes all the way round (a pole)

# Projection coordinates (CF 4.4) in these units, as metres.
LENGTH_UNITS = {"m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
                "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0, "kilometre": 1000.0,
                "kilometres": 1000.0}


@dataclass(frozen=True, eq=False)
class Projection:
    """The map projection a grid was made in (CF 5.6), with its 1D x and y in metres.

    Its cells are rectangles in x and y, so they can be drawn exactly, wherever
    they are: averaging 2D latitudes and longitudes can't, e.g. around a pole.
    """

    crs: pyproj.CRS
    x: np.ndarray  # along the grid's second dim
    y: np.ndarray  # along its first

    def edges(self, index) -> tuple[np.ndarray, np.ndarray]:
        """(x, y) edges of the cells drawn at the full-resolution `index` (rows, columns)."""
        rows, cols = index
        return _index_edges(self.x, cols), _index_edges(self.y, rows)

    def centre(self) -> tuple[float, float]:
        """(lat, lon) to centre a map on: the middle of the grid, with its y axis upright.

        At a pole any longitude is the middle, so it's the one the grid has going
        down from a north pole (up from a south pole), as on the grid itself.
        """
        to_lonlat = pyproj.Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        x, y = (self.x[0] + self.x[-1]) / 2, (self.y[0] + self.y[-1]) / 2
        lon, lat = to_lonlat.transform(x, y)
        step = 0.01 * abs(self.y[-1] - self.y[0]) / max(self.y.size - 1, 1)
        towards_pole = to_lonlat.transform(x, y + (-step if lat >= 0 else step))[0]
        return float(lat), float(towards_pole)


def _index_edges(full, idx):
    return block_edges(np.arange(full.size), idx, cell_edges(full))


@dataclass(frozen=True, eq=False)
class GeoInfo:
    """How a variable's dimensions map onto the globe."""

    kind: str  # "regular" (1D lat x 1D lon), "curvilinear" (2D lat/lon) or "points"
    lat: xr.DataArray
    lon: xr.DataArray
    dims: tuple[str, ...]  # data dims spanned by the grid: (y, x), or (point,)
    # Cell boundaries (CF 7.1): (n, 2) for regular grids, (ny, nx, 4) for curvilinear ones
    lat_bounds: xr.DataArray | None = None
    lon_bounds: xr.DataArray | None = None
    projection: Projection | None = None  # curvilinear grids made in a map projection
    cache: dict = field(default_factory=dict, repr=False)  # full point coordinates, see point_window


def find_geo(da: xr.DataArray, ds: xr.Dataset | None = None) -> GeoInfo | None:
    geo = _find_geo(da, ds)
    if geo is None or geo.kind == "points":
        return geo
    nv = 2 if geo.kind == "regular" else 4
    projection = _projection(da, ds, geo.dims) if geo.kind == "curvilinear" else None
    return GeoInfo(geo.kind, geo.lat, geo.lon, geo.dims,
                   _bounds(geo.lat, da, ds, nv), _bounds(geo.lon, da, ds, nv), projection)


def _find_geo(da, ds):
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
    # Features (CF 9) aren't grids, even with 2D positions, e.g. lat(trajectory, obs).
    features = is_discrete_sampling(ds)
    for lat in [] if features else lats:  # 2D (or more, e.g. WRF's XLAT(Time, y, x)), same last 2 dims
        for lon in lons:
            if lat.ndim >= 2 and lon.ndim >= 2 and lat.dims[-2:] == lon.dims[-2:]:
                if set(lat.dims) <= dims and set(lon.dims) <= dims:
                    return GeoInfo("curvilinear", lat, lon, tuple(lat.dims[-2:]))
    for lat in lats:  # stations / swath points: lat and lon along the same dim
        for lon in lons:
            if lat.ndim == lon.ndim == 1 and lat.dims == lon.dims and lat.dims[0] in dims:
                return GeoInfo("points", lat, lon, lat.dims)
            # Features: points along the sample dim (last, 9.3), one feature at a time.
            if features and lat.ndim >= 2 and lat.dims == lon.dims and set(lat.dims) <= dims:
                return GeoInfo("points", lat, lon, lat.dims[-1:])
    return None


def _bounds(coord, da, ds, nv):
    """The coordinate's boundary variable (CF 7.1), if it has a usable one."""
    name = coord.attrs.get("bounds") or coord.encoding.get("bounds")
    if not name:
        return None
    if name in da.coords:
        bounds = da.coords[name]
    elif ds is not None and name in ds.variables:
        bounds = ds[name]
    else:
        return None
    if bounds.ndim != coord.ndim + 1 or bounds.dims[:-1] != coord.dims or bounds.shape[-1] != nv \
            or bounds.dtype.kind not in "fiu":
        return None
    return bounds


def _projection(da, ds, dims) -> Projection | None:
    """The variable's grid mapping (CF 5.6), if it's a map projection with x and y along `dims`."""
    name = da.attrs.get("grid_mapping") or da.encoding.get("grid_mapping")
    if not name or ds is None:
        return None
    name = str(name).split(":")[0].strip()  # the first mapping of the extended form "crs: x y ..."
    if name not in ds.variables:
        return None
    try:
        crs = pyproj.CRS.from_cf(dict(ds[name].attrs))
    except (pyproj.exceptions.CRSError, ValueError, KeyError, TypeError):
        return None
    if not crs.is_projected:  # e.g. rotated poles, whose coordinates are in degrees
        return None
    candidates = _candidates(da, ds).values()
    y, x = (_projection_coord(candidates, dim, f"projection_{axis}_coordinate")
            for dim, axis in zip(dims, "yx", strict=True))
    return None if x is None or y is None else Projection(crs, x, y)


def _projection_coord(candidates, dim, standard_name):
    """A 1D projection coordinate along `dim`, in metres."""
    for var in candidates:
        if var.dims != (dim,) or var.attrs.get("standard_name") != standard_name:
            continue
        scale = LENGTH_UNITS.get(str(var.attrs.get("units", "m")).strip().lower())
        values = np.asarray(var.values, dtype=float)
        steps = np.diff(values)
        if scale and np.isfinite(values).all() and (np.all(steps > 0) or np.all(steps < 0)):
            return values * scale
    return None


def _candidates(da, ds):
    found = {str(name): da.coords[name] for name in da.coords}
    if ds is not None:
        for name, var in ds.variables.items():
            if name != da.name and str(name) not in found and var.ndim >= 1 \
                    and set(var.dims) <= set(da.dims):
                found[str(name)] = ds[name]  # a DataArray, like the coordinates
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
    # Regular grids: edges of the drawn rows/columns (ny + 1, nx + 1). Always edges of
    # full-resolution cells, so a thinned-out grid lines up with the real one.
    lat_edges: np.ndarray | None = None
    lon_edges: np.ndarray | None = None
    # Curvilinear grids at full resolution with cell bounds: (lat, lon) of the cell
    # corners, each (ny + 1, nx + 1), longitudes without jumps.
    corners: tuple[np.ndarray, np.ndarray] | None = None
    from_bounds: bool = False  # the cell edges/corners come from bounds (CF 7.1)
    # Curvilinear grids in a map projection (GeoInfo.projection): (x, y) edges of the
    # drawn columns and rows in its coordinates, real cell edges like lat_edges/lon_edges.
    xy_edges: tuple[np.ndarray, np.ndarray] | None = None
    crs: pyproj.CRS | None = None  # of xy_edges

    def corner_lonlat(self) -> tuple[np.ndarray, np.ndarray] | None:
        """(lat, lon) of the drawn cells' corners (ny + 1, nx + 1), exactly, from xy_edges.

        Longitudes are in the grid's convention (see lon_0_360), so they jump by
        360 degrees where the grid crosses the antimeridian (or goes round a pole).
        """
        if self.xy_edges is None or self.crs is None:
            return None
        x, y = np.meshgrid(*self.xy_edges)
        lon, lat = pyproj.Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True).transform(x, y)
        return np.asarray(lat), normalize_lon(np.asarray(lon), self.lon_0_360)[0]

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
    lat, lon = _positions(geo, sl)
    lon, lon_0_360 = normalize_lon(lon, like.lon_0_360 if like else None)
    index = tuple(sl.index(axis) for axis in range(len(sl.dims)))
    if geo.kind == "regular":
        rows, cols = np.argsort(lat, kind="stable"), np.argsort(lon, kind="stable")
        lat, lon, values = lat[rows], lon[cols], values[np.ix_(rows, cols)]
        index = (index[0][rows], index[1][cols])
        (full_lat, lat_edges), (full_lon, lon_edges) = regular_edges(geo, lon_0_360)
        return GeoGrid(geo.kind, lat, lon, values, sl, index, lon_0_360,
                       block_edges(full_lat, lat, lat_edges), block_edges(full_lon, lon, lon_edges),
                       from_bounds=_moved(full_lat, lat_edges) or _moved(full_lon, lon_edges))
    corners = None
    if geo.kind == "curvilinear" and geo.lat_bounds is not None and geo.lon_bounds is not None \
            and not sl.downsampled:
        from netseedf.core.gridlines import unwrap_lon  # (gridlines needs this module)

        unwrapped = unwrap_lon(lon)
        raw_lon = _coord_slice(geo.lon, sl)
        # Each vertex within 180 degrees of its cell's centre, like the unwrapped centres
        lon_b = unwrapped[..., None] + lon_diff(_bounds_slice(geo.lon_bounds, sl), raw_lon[..., None])
        corners = vertex_corners(lat, unwrapped, _bounds_slice(geo.lat_bounds, sl), lon_b)
    xy_edges = geo.projection.edges(index) if geo.projection is not None else None
    return GeoGrid(geo.kind, lat, lon, values, sl, index, lon_0_360,
                   corners=corners, from_bounds=corners is not None, xy_edges=xy_edges,
                   crs=geo.projection.crs if geo.projection is not None else None)


def regular_edges(geo: GeoInfo, lon_0_360):
    """((lat, lat edges), (lon, lon edges)) of a regular grid at full resolution, sorted.

    Edges come from the cell bounds when there are any (and they're contiguous),
    else they're halfway between the centres.
    """
    lat = np.asarray(geo.lat.values, dtype=float)
    raw_lon = np.asarray(geo.lon.values, dtype=float)
    lon = normalize_lon(raw_lon, lon_0_360)[0]
    lat_b = None if geo.lat_bounds is None else np.asarray(geo.lat_bounds.values, dtype=float)
    lon_b = None if geo.lon_bounds is None else np.asarray(geo.lon_bounds.values, dtype=float)
    if lon_b is not None:  # shifted by 360 degrees along with their centres
        lon_b = lon_b + (lon - raw_lon)[:, None]
    return axis_edges(lat, lat_b), axis_edges(lon, lon_b)


def cells_from_bounds(geo: GeoInfo) -> bool:
    """Do the grid's bounds put its cells anywhere but around (and halfway between) the centres?"""
    if geo.kind == "curvilinear":
        return geo.lat_bounds is not None and geo.lon_bounds is not None
    if geo.kind != "regular":
        return False
    (lat, lat_edges), (lon, lon_edges) = regular_edges(geo, None)
    return _moved(lat, lat_edges) or _moved(lon, lon_edges)


def _moved(centres, edges):
    return not np.allclose(edges, cell_edges(centres), rtol=0, atol=1e-6)


def axis_edges(centres, bounds=None, tolerance=1e-4):
    """Sorted centres and the (n + 1) edges of their cells, from bounds (n, 2) if possible."""
    order = np.argsort(centres, kind="stable")
    centres = centres[order]
    if bounds is not None:
        b = np.sort(bounds[order], axis=1)
        if np.isfinite(b).all() and np.allclose(b[1:, 0], b[:-1, 1], rtol=0, atol=tolerance):
            return centres, np.append(b[:, 0], b[-1, 1])
    return centres, cell_edges(centres)


def _bounds_slice(bounds: xr.DataArray, sl: Slice) -> np.ndarray:
    """Bounds of the cells in a slice: (..., number of vertices)."""
    vertex = bounds.dims[-1]
    return np.stack([_coord_slice(bounds.isel({vertex: k}), sl) for k in range(bounds.sizes[vertex])],
                    axis=-1)


def vertex_corners(lat, lon, lat_b, lon_b, tolerance=1e-4):
    """Corners (ny + 1, nx + 1) shared by neighbouring cells, from each cell's 4 vertices.

    The vertices can come in any order (CF only asks for anticlockwise), so
    each corner is matched to the nearest vertex. None if neighbouring cells
    don't agree on the corners they share, or the grid is too small.
    """
    from netseedf.core.gridlines import cell_corners  # (gridlines needs this module)

    ny, nx = lat.shape
    if ny < 2 or nx < 2:
        return None
    guess_lat, guess_lon = cell_corners(lat), cell_corners(lon)
    corner_lat, corner_lon = np.full((ny + 1, nx + 1), np.nan), np.full((ny + 1, nx + 1), np.nan)
    j, i = np.ogrid[0:ny, 0:nx]
    for dj, di in ((0, 0), (0, 1), (1, 1), (1, 0)):
        g_lat, g_lon = guess_lat[dj:dj + ny, di:di + nx], guess_lon[dj:dj + ny, di:di + nx]
        d2 = (lat_b - g_lat[..., None]) ** 2 + (lon_b - g_lon[..., None]) ** 2
        k = np.argmin(np.where(np.isfinite(d2), d2, np.inf), axis=-1)
        for corner, b in ((corner_lat, lat_b), (corner_lon, lon_b)):
            v = b[j, i, k]
            sub = corner[dj:dj + ny, di:di + nx]
            both = np.isfinite(sub) & np.isfinite(v)
            if not np.allclose(sub[both], v[both], rtol=0, atol=tolerance):
                return None
            unset = ~np.isfinite(sub)
            sub[unset] = v[unset]
    return corner_lat, corner_lon


def fill_missing_positions(lat, lon, values):
    """A curvilinear grid's (lat, lon, values), ready for a mesh, which can't have missing positions.

    Missing positions (CF 2.5.1) continue the grid around them, linearly, so the
    cells next to them keep their size (and a mesh its shape). Their values
    become missing, so they aren't drawn.
    """
    missing = ~(np.isfinite(lat) & np.isfinite(lon))
    if not missing.any() or missing.all():
        return lat, lon, np.where(missing, np.nan, values)
    # Longitudes relative to a typical one, so they're continued without jumps of 360°.
    ref = float(np.median(lon[~missing]))
    lat_c = np.clip(continue_grid(np.where(missing, np.nan, lat)), -90, 90)
    lon_c = ref + continue_grid(np.where(missing, np.nan, lon_diff(lon, ref)))
    return (np.where(missing, lat_c, lat), np.where(missing, lon_c, lon),
            np.where(missing, np.nan, values))


def continue_grid(a):
    """A 2D array with its NaNs continued linearly from the valid values.

    Along the rows first (inside gaps: interpolated, at the ends: extrapolated,
    from one value: repeated), then the same along the columns for rows with none.
    """
    return _continue_rows(_continue_rows(a).T).T


def _continue_rows(a):
    a = np.array(a, dtype=float)
    x = np.arange(a.shape[1], dtype=float)
    for row in a:
        ok = np.isfinite(row)
        if ok.all() or not ok.any():
            continue
        xs, ys = x[ok], row[ok]
        if xs.size == 1:
            row[~ok] = ys[0]
            continue
        filled = np.interp(x, xs, ys)
        before, after = x < xs[0], x > xs[-1]
        filled[before] = ys[0] + (x[before] - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
        filled[after] = ys[-1] + (x[after] - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        row[~ok] = filled[~ok]
    return a


def cell_edges(centres):
    """Edges between 1D cell centres, extending half a cell at both ends."""
    c = np.asarray(centres, dtype=float)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = (c[:-1] + c[1:]) / 2
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


def block_edges(full, centres, edges=None):
    """Edges of the cells drawn at `centres`, some (sorted) values of the coordinate `full`.

    When the grid is thinned out, each drawn cell stands for the real cells
    halfway to its neighbours (at the ends: all the way to where the next one
    would be, so the skipped cells there are covered too). Its edges are real
    cell edges, not midpoints between the drawn centres, so it lines up with
    the full-resolution grid. `edges` are those of the full-resolution cells
    (by default halfway between the sorted `full` values).
    """
    full = np.sort(full)
    edges = cell_edges(full) if edges is None else edges
    pos = np.clip(np.searchsorted(full, centres), 0, full.size - 1)
    if pos.size == 1:
        return edges[[pos[0], pos[0] + 1]]
    gaps = np.diff(pos)
    inner = pos[:-1] + (gaps + 1) // 2
    first = max(pos[0] - gaps[0] + 1, 0)
    last = min(pos[-1] + gaps[-1], full.size)
    return edges[np.concatenate([[first], inner, [last]])]


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
    lon_edges = grid.lon_edges
    if lon_edges is not None:  # the last column now reaches the repeated first one
        lon_edges = np.concatenate([lon_edges[:-1], lon_edges[:2] + 360])
    return GeoGrid(grid.kind, grid.lat, lon, values, grid.slice, index, grid.lon_0_360,
                   grid.lat_edges, lon_edges, from_bounds=grid.from_bounds)


def _positions(geo: GeoInfo, sl: Slice) -> tuple[np.ndarray, np.ndarray]:
    """Latitudes and longitudes of a slice. Positions off the globe count as missing.

    Auxiliary coordinates may have missing values (CF 2.5.1), and not every file
    marks them: e.g. undefined satellite pixels holding garbage like 2e9.
    """
    lat, lon = _coord_slice(geo.lat, sl), _coord_slice(geo.lon, sl)
    if geo.kind != "regular":  # coordinate variables can't have missing values
        with np.errstate(invalid="ignore"):
            off = ~((np.abs(lat) <= 90) & (np.abs(lon) <= 720))
        if off.any():
            lat, lon = np.where(off, np.nan, lat), np.where(off, np.nan, lon)
    return lat, lon


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
        if grid.kind == "regular":
            # In view may be both ends of the stored longitudes (e.g. of 0..360 around 0°),
            # which a (start, stop) range could only span with everything in between.
            window[dim] = _index_window(idx, step, geo.lat.size if dim == geo.dims[0] else geo.lon.size)
    return window


def _index_window(idx, step, n):
    """(start, stop) of the full-resolution indices within `step` of `idx`, or all of
    them (an array) when they're in more than one run."""
    near = np.unique(np.clip((idx[:, None] + np.arange(-step, step + 1)).ravel(), 0, n - 1))
    if near[-1] - near[0] + 1 == near.size:
        return int(near[0]), int(near[-1]) + 1
    return near


def point_window(geo: GeoInfo, grid: GeoGrid, box) -> dict | None:
    """Indices of all points inside `box` (not just the ones in the thinned grid)."""
    sl = grid.slice
    key = tuple(sorted((d, i) for d, i in sl.fixed.items() if d in geo.lat.dims + geo.lon.dims))
    if key not in geo.cache:
        dim = sl.dims[0]
        full = Slice(np.empty(0), sl.dims, (1,), sl.fixed, (slice(0, geo.lat.sizes[dim], 1),))
        geo.cache.clear()  # keep only the current time step's coordinates
        lat, lon = _positions(geo, full)
        geo.cache[key] = (lat, normalize_lon(lon, grid.lon_0_360)[0])
    lat, lon = geo.cache[key]
    idx = np.flatnonzero(in_box(lat, lon, box))
    return {sl.dims[0]: idx} if idx.size else None


def normalize_lon(lon: np.ndarray, lon_0_360=None) -> tuple[np.ndarray, bool]:
    """Longitudes in -180..180 or 0..360, and whether it's the latter.

    Without `lon_0_360`, picks whichever keeps the data contiguous: a Pacific
    box stored as 150..210 stays that way instead of being split across the
    map; global grids become -180..180. So do grids around a pole: either way
    they span almost 360°, and which one is a hair shorter is chance, but the
    seam (where cells straddling it are left out) belongs at the antimeridian,
    not at Greenwich.
    """
    west = (lon + 180) % 360 - 180
    east = lon % 360
    if lon_0_360 is None:
        lon_0_360 = bool(_span(east) < min(_span(west), WRAPS_AROUND))
    return (east if lon_0_360 else west), lon_0_360


def _span(a):
    return np.nanmax(a) - np.nanmin(a) if a.size else 0.0


def lon_diff(a, b):
    """Signed difference a - b in degrees, wrapped into [-180, 180)."""
    return (np.asarray(a) - b + 180) % 360 - 180


class GridLocator:
    """Finds the grid cell under a lat/lon position (e.g. the mouse)."""

    def __init__(self, grid: GeoGrid, tolerance=None):
        """`tolerance`: how far (in degrees) a point may be from a position to count."""
        self.grid = grid
        if grid.kind != "regular":
            self._tolerance = tolerance or (self._typical_spacing() if grid.kind == "curvilinear"
                                            else max(0.02 * float(np.hypot(*_extent(grid))), 0.05))

    def nearest(self, lat, lon):
        """Index into grid.values, or None when outside the data."""
        g = self.grid
        if not (np.isfinite(lat) and np.isfinite(lon)):
            return None
        if g.kind == "regular":
            if g.lat_edges is not None and g.lon_edges is not None:  # the cell holding the position
                i = _in_cell(g.lat_edges, lat)
                j = next((j for k in (0, -360, 360) if (j := _in_cell(g.lon_edges, lon + k)) is not None),
                         None)
            else:
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



@dataclass
class PickedPoint:
    """The full-resolution grid cell (or point) at a clicked position."""

    lat: float
    lon: float
    value: float
    index: dict[str, int]  # index along each of the grid's dims


def pick_point(da: xr.DataArray, geo: GeoInfo, grid: GeoGrid, lat, lon, indices) -> PickedPoint | None:
    """The cell under lat/lon at full resolution, even if `grid` is downsampled.

    `grid` finds the neighbourhood; the cells the stride skipped around it are
    then read to find the nearest one. Points (stations) come in no particular
    order, so for them every point near lat/lon is read instead.
    """
    locator = GridLocator(grid)
    found = locator.nearest(lat, lon)
    sl = grid.slice
    if sl.downsampled and grid.kind == "points":
        tol = locator._tolerance
        dlon = tol / max(np.cos(np.deg2rad(lat)), 0.1)
        window = point_window(geo, grid, (lat - tol, lon - dlon, lat + tol, lon + dlon))
        if window is not None:
            fine = geo_grid(da, geo, indices, None, window, like=grid)
            grid, found = fine, GridLocator(fine, tolerance=tol).nearest(lat, lon)
    elif sl.downsampled and found is not None:
        centre = [int(grid.index[axis][found[axis]]) for axis in range(len(sl.dims))]
        window = {d: (c - step, c + step + 1) for d, c, step in zip(sl.dims, centre, sl.steps, strict=True)}
        fine = geo_grid(da, geo, indices, None, window, like=grid)
        fine_found = GridLocator(fine).nearest(lat, lon)
        if fine_found is not None:
            grid, found = fine, fine_found
    if found is None:
        return None
    index = {d: int(grid.index[axis][found[axis]]) for axis, d in enumerate(grid.slice.dims)}
    if grid.kind == "regular":
        cell_lat, cell_lon = grid.lat[found[0]], grid.lon[found[1]]
    else:
        cell_lat, cell_lon = grid.lat[found], grid.lon[found]
    return PickedPoint(float(cell_lat), float(cell_lon), float(grid.values[found]), index)


def _extent(grid):
    south, west, north, east = grid.bounds
    return north - south, east - west


def _in_cell(edges, q):
    """Index of the cell between ascending `edges` that holds `q`."""
    if not edges[0] <= q <= edges[-1]:
        return None
    return min(int(np.searchsorted(edges, q, side="right")) - 1, len(edges) - 2)


def _nearest_1d(coord, q, wrap=False):
    diff = lon_diff(coord, q) if wrap else coord - q
    i = int(np.argmin(np.abs(diff)))
    if len(coord) > 1:
        neighbours = [abs(coord[k] - coord[i]) for k in (i - 1, i + 1) if 0 <= k < len(coord)]
        half_cell = 0.5 * min(neighbours) * 1.0001
    else:
        half_cell = 0.5
    return i if abs(diff[i]) <= half_cell else None
