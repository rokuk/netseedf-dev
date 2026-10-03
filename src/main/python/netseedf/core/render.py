"""Rendering map data as images and colours for the Leaflet web map.

Regular grids become one pixel per cell, with the edges of every row and
column: the page stretches each between its edges, so cells line up with
the basemap and the grid lines at any zoom.

Curvilinear grids with few enough cells (at full resolution) become one
quadrilateral per cell, with the same corners as the grid lines. Others
become an image stretched linearly between its bounds in Web Mercator, so
their image rows are laid out in Mercator y, not in latitude. Otherwise the
overlay would drift away from the basemap towards the poles.

Curvilinear grids made in a map projection (e.g. polar ones) are always such
an image, but made the other way round: each pixel is projected into the
grid's own x and y to find its cell. That is exact where interpolating
latitudes and longitudes can't be: around a pole their longitudes go all the
way round, so cells (and their corners) there have no sensible longitude.
"""

import base64
import io
from dataclasses import dataclass

import matplotlib
import numpy as np
import pyproj
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize, to_hex
from matplotlib.figure import Figure
from PIL import Image

from netseedf.core.coords import (
    MERCATOR_MAX_LAT,
    GeoGrid,
    cell_edges,
    fill_missing_positions,
    normalize_lon,
)
from netseedf.core.gridlines import cell_corners, grid_corners

PIXELS_PER_CELL = 3
MAX_POLYGON_CELLS = 20_000  # more and drawing every cell as a polygon gets slow
# Resampled cell edges land on the nearest image pixel, so they're off by up to half a pixel.
# Zoomed in, a few pixels per cell makes that visible (and it shifts whenever the
# detail is reloaded), so small images are scaled up to about screen resolution.
MIN_PX = 2048
PROJECT_EVERY = 4  # pixels: projected exactly on a lattice this coarse, interpolated in between


def _image_size(width, height, max_px):
    """(width, height) scaled up by a whole number (keeps whole pixels per cell) to MIN_PX."""
    width, height = max(int(width), 1), max(int(height), 1)
    longest = max(width, height)
    scale = max(1, min(-(-MIN_PX // longest), max_px // longest))
    return min(width * scale, max_px), min(height * scale, max_px)


def mercator_y(lat):
    lat = np.clip(lat, -MERCATOR_MAX_LAT, MERCATOR_MAX_LAT)
    return np.log(np.tan(np.pi / 4 + np.deg2rad(lat) / 2))


def inverse_mercator_y(y):
    return np.rad2deg(2 * np.arctan(np.exp(y)) - np.pi / 2)


@dataclass
class Overlay:
    png: bytes
    south: float
    west: float
    north: float
    east: float
    # Regular grids: edges of the image rows (north to south) and columns (west to east).
    rows: list[float] | None = None
    cols: list[float] | None = None

    @property
    def data_url(self):
        return "data:image/png;base64," + base64.b64encode(self.png).decode("ascii")

    @property
    def size(self):
        return Image.open(io.BytesIO(self.png)).size


def _cmap(name):
    return matplotlib.colormaps[name].with_extremes(bad=(0, 0, 0, 0))


def render_overlay(grid: GeoGrid, cmap, vmin, vmax, max_px=4096) -> Overlay | None:
    """Colour-mapped transparent PNG of a regular or curvilinear grid."""
    if grid.kind == "regular":
        return _render_regular(grid, cmap, vmin, vmax, max_px)
    if grid.kind == "curvilinear" and grid.xy_edges is not None and grid.crs is not None:
        return _render_projected(grid, cmap, vmin, vmax, max_px)
    if grid.kind == "curvilinear":
        return _render_curvilinear(grid, cmap, vmin, vmax, max_px)
    raise ValueError(f"Can't render a {grid.kind} grid as an image")


def _render_regular(grid, cmap, vmin, vmax, max_px):
    lat_edges = grid.lat_edges if grid.lat_edges is not None else cell_edges(grid.lat)
    lon_edges = grid.lon_edges if grid.lon_edges is not None else cell_edges(grid.lon)
    lat_edges = np.clip(lat_edges, -MERCATOR_MAX_LAT, MERCATOR_MAX_LAT)
    shown = np.flatnonzero(lat_edges[1:] > lat_edges[:-1])  # not wholly beyond Mercator's limit
    if not shown.size:
        return None  # all of it is too close to a pole for Web Mercator
    first, last = shown[0], shown[-1] + 1
    lat_edges, values = lat_edges[first:last + 1], grid.values[first:last]

    rgba = _cmap(cmap)(Normalize(vmin, vmax)(np.ma.masked_invalid(values[::-1])), bytes=True)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="png")
    return Overlay(buf.getvalue(), float(lat_edges[0]), float(lon_edges[0]), float(lat_edges[-1]),
                   float(lon_edges[-1]), rows=[float(y) for y in lat_edges[::-1]],
                   cols=[float(x) for x in lon_edges])


@dataclass
class Cells:
    """A curvilinear grid's cells as quadrilaterals, see cell_polygons."""

    shape: tuple[int, int]  # (ny, nx) cells
    lat: list[float]  # corners, (ny + 1) x (nx + 1) row by row
    lon: list[float]
    colors: list[str | None]  # per cell, row by row; None where there's no value
    south: float
    west: float
    north: float
    east: float


def cell_polygons(grid: GeoGrid, cmap, vmin, vmax, max_cells=MAX_POLYGON_CELLS) -> Cells | None:
    """Every cell of a full-resolution curvilinear grid, if there aren't too many.

    The corners are the ones the grid lines are drawn with, so the two match.
    None when an image has to do instead, always for grids in a map projection:
    their image is exact (see _render_projected), and their quadrilaterals in
    latitude and longitude wouldn't be, e.g. around a pole.
    """
    if grid.kind != "curvilinear" or grid.slice.downsampled or grid.xy_edges is not None:
        return None
    if grid.values.size > max_cells or min(grid.values.shape) < 2:
        return None
    lat, lon = grid_corners(grid)
    ok = np.isfinite(lat) & np.isfinite(lon)
    if not ok.any():
        return None
    corners_ok = ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]
    corners_ok &= np.isfinite(grid.lat) & np.isfinite(grid.lon)  # no cell where its position is missing
    values = np.where(corners_ok, np.asarray(grid.values, dtype=float), np.nan)
    colors = point_colors(values.ravel(), cmap, vmin, vmax)
    return Cells(values.shape, np.round(lat, 6).ravel().tolist(), np.round(lon, 6).ravel().tolist(),
                 colors, float(np.min(lat[ok])), float(np.min(lon[ok])),
                 float(np.max(lat[ok])), float(np.max(lon[ok])))


def _render_curvilinear(grid, cmap, vmin, vmax, max_px):
    ok = np.isfinite(grid.lat) & np.isfinite(grid.lon)
    if not ok.any():
        return None
    lat, lon, values = fill_missing_positions(grid.lat, grid.lon, np.array(grid.values, dtype=float))
    values[_seam_mask(lon)] = np.nan  # cells straddling the antimeridian would smear across
    ny, nx = values.shape
    width = int(np.clip(nx * PIXELS_PER_CELL, 256, max_px))
    height = int(np.clip(ny * PIXELS_PER_CELL, 256, max_px))
    width, height = _image_size(width, height, max_px)

    fig = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor="none")
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    style = dict(cmap=_cmap(cmap), norm=Normalize(vmin, vmax), antialiased=False)
    if min(ny, nx) < 2:
        x, y = lon, mercator_y(lat)
        ax.pcolormesh(x, y, np.ma.masked_invalid(values), shading="nearest", **style)
        inside = np.ones(x.shape, dtype=bool)
    else:
        # The corners halfway between the centres (what shading="nearest" would use), given
        # explicitly: matplotlib's guess at them doesn't allow for missing positions.
        x, y = cell_corners(lon), mercator_y(cell_corners(lat))
        ax.pcolormesh(x, y, np.ma.masked_invalid(values), shading="flat", **style)
        inside = corners_of(ok)  # the image covers the cells with positions, not the made-up ones
    x0, x1 = float(np.min(x[inside])), float(np.max(x[inside]))
    y0, y1 = max(float(np.min(y[inside])), mercator_y(-90)), min(float(np.max(y[inside])), mercator_y(90))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, transparent=True)
    return Overlay(buf.getvalue(), float(inverse_mercator_y(y0)), float(x0),
                   float(inverse_mercator_y(y1)), float(x1))


def _render_projected(grid, cmap, vmin, vmax, max_px):
    """A grid in a map projection, resampled onto Web Mercator pixels."""
    extent = projected_extent(grid)
    if extent is None:
        return None
    south, west, north, east = extent
    y0, y1 = mercator_y(south), mercator_y(north)
    ny, nx = grid.values.shape
    longest = int(np.clip(max(ny, nx) * PIXELS_PER_CELL, 256, max_px))
    aspect = (y1 - y0) / np.deg2rad(east - west)  # height / width
    width, height = (longest, longest * aspect) if aspect <= 1 else (longest / aspect, longest)
    width, height = _image_size(width, height, max_px)

    lon = west + (np.arange(width) + 0.5) * (east - west) / width
    lat = inverse_mercator_y(y1 - (np.arange(height) + 0.5) * (y1 - y0) / height)
    to_xy = pyproj.Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    x, y = _project_lattice(to_xy, lat, lon)
    x_edges, y_edges = grid.xy_edges
    col, row = _cell_index(x_edges, x), _cell_index(y_edges, y)
    inside = (col >= 0) & (row >= 0)
    colors = _cmap(cmap)(Normalize(vmin, vmax)(np.ma.masked_invalid(grid.values)), bytes=True)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[inside] = colors[row[inside], col[inside]]
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="png")
    return Overlay(buf.getvalue(), float(south), float(west), float(north), float(east))


def projected_extent(grid: GeoGrid, samples=256):
    """(south, west, north, east) a projected grid covers, within Web Mercator's latitudes.

    From its outline: inside it, latitudes and longitudes only go further at a
    pole. A grid around one covers all longitudes, in its convention (see
    lon_0_360). None if it's all closer to a pole than Web Mercator goes.
    """
    (x0, x1), (y0, y1) = ((float(e[0]), float(e[-1])) for e in grid.xy_edges)
    t = np.linspace(0, 1, samples, endpoint=False)
    x = np.concatenate([x0 + (x1 - x0) * t, np.full(samples, x1), x1 - (x1 - x0) * t, np.full(samples, x0)])
    y = np.concatenate([np.full(samples, y0), y0 + (y1 - y0) * t, np.full(samples, y1), y1 - (y1 - y0) * t])
    lon, lat = (np.asarray(a) for a in
                pyproj.Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True).transform(x, y))
    ok = np.isfinite(lon) & np.isfinite(lat)
    if not ok.any():
        return None
    lon, lat = lon[ok], lat[ok]
    south, north = float(np.min(lat)), float(np.max(lat))
    to_xy = pyproj.Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    for pole in (90.0, -90.0):
        px, py = to_xy.transform(0.0, pole)
        if np.isfinite(px) and np.isfinite(py) \
                and min(x0, x1) <= px <= max(x0, x1) and min(y0, y1) <= py <= max(y0, y1):
            south, north = (south, pole) if pole > 0 else (pole, north)
            west = 0.0 if grid.lon_0_360 else -180.0
            return _mercator_extent(south, west, north, west + 360)
    # Going round the outline, longitudes without jumps of 360°.
    lon = np.rad2deg(np.unwrap(np.deg2rad(lon)))
    west, east = float(np.min(lon)), float(np.max(lon))
    middle = (west + east) / 2
    shift = float(normalize_lon(np.array([middle]), grid.lon_0_360)[0][0]) - middle
    return _mercator_extent(south, west + shift, north, east + shift)


def _mercator_extent(south, west, north, east):
    south, north = max(south, -MERCATOR_MAX_LAT), min(north, MERCATOR_MAX_LAT)
    return (south, west, north, east) if south < north and west < east else None


def _project_lattice(transformer, lat, lon):
    """Projected (x, y) of every pixel at (lat, lon), each (lat.size, lon.size).

    Projected exactly every PROJECT_EVERY pixels, bilinearly in between: map
    projections are smooth, so that's as good, and many times faster.
    """
    rows, cols = _lattice(lat.size), _lattice(lon.size)
    lo, la = np.meshgrid(lon[cols], lat[rows])
    x, y = transformer.transform(lo, la)
    return tuple(_interpolate(np.asarray(a, dtype=float), rows, cols, lat.size, lon.size)
                 for a in (x, y))


def _lattice(n):
    return np.unique(np.append(np.arange(0, n, PROJECT_EVERY), n - 1))


def _interpolate(a, rows, cols, height, width):
    """Values on the lattice `a` (rows x cols) at every pixel (height x width)."""
    (i, s), (j, t) = _lerp(rows, height), _lerp(cols, width)
    a = a[:, j] * (1 - t) + a[:, j + 1] * t if cols.size > 1 else a[:, j]
    return a[i] * (1 - s)[:, None] + a[i + 1] * s[:, None] if rows.size > 1 else a[i]


def _lerp(points, n):
    """For each of 0..n-1: the lattice point before it, and how far it is towards the next."""
    p = np.arange(n)
    if points.size == 1:
        return np.zeros(n, dtype=int), np.zeros(n)
    i = np.clip(np.searchsorted(points, p, side="right") - 1, 0, points.size - 2)
    return i, (p - points[i]) / (points[i + 1] - points[i])


def _cell_index(edges, q):
    """Index of the cell between monotonic `edges` holding each of `q`; -1 if none does."""
    n = edges.size - 1
    ascending = edges[-1] > edges[0]
    with np.errstate(invalid="ignore"):
        i = np.searchsorted(edges if ascending else edges[::-1], q, side="right") - 1
        i = np.where(np.isfinite(q) & (i >= 0) & (i < n), i, -1)
    return i if ascending else np.where(i >= 0, n - 1 - i, -1)


def corners_of(cells):
    """Which corners (ny + 1, nx + 1) belong to any of the `cells` (ny, nx)."""
    corners = np.zeros((cells.shape[0] + 1, cells.shape[1] + 1), dtype=bool)
    for dj in (0, 1):
        for di in (0, 1):
            corners[dj:dj + cells.shape[0], di:di + cells.shape[1]] |= cells
    return corners


def _seam_mask(lon):
    mask = np.zeros(lon.shape, dtype=bool)
    for axis in (0, 1):
        if lon.shape[axis] < 2:
            continue
        jump = np.abs(np.diff(lon, axis=axis)) > 180
        lead = [slice(None)] * 2
        lead[axis] = slice(None, -1)
        trail = [slice(None)] * 2
        trail[axis] = slice(1, None)
        mask[tuple(lead)] |= jump
        mask[tuple(trail)] |= jump
    return mask


def point_colors(values, cmap, vmin, vmax) -> list[str | None]:
    colors = _cmap(cmap)(Normalize(vmin, vmax)(np.ma.masked_invalid(values)))
    return [to_hex(c) if np.isfinite(v) else None for c, v in zip(colors, values, strict=True)]


def legend_colors(cmap, n=16) -> list[str]:
    cm = matplotlib.colormaps[cmap]
    return [to_hex(cm(i / (n - 1))) for i in range(n)]

