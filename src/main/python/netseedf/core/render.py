"""Rendering map data as images and colours for the Leaflet web map.

Regular grids become one pixel per cell, with the edges of every row and
column: the page stretches each between its edges, so cells line up with
the basemap and the grid lines at any zoom.

Curvilinear grids with few enough cells (at full resolution) become one
quadrilateral per cell, with the same corners as the grid lines. Others
become an image stretched linearly between its bounds in Web Mercator, so
their image rows are laid out in Mercator y, not in latitude. Otherwise the
overlay would drift away from the basemap towards the poles.
"""

import base64
import io
from dataclasses import dataclass

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize, to_hex
from matplotlib.figure import Figure
from PIL import Image

from netseedf.core.coords import MERCATOR_MAX_LAT, GeoGrid, cell_edges
from netseedf.core.gridlines import cell_corners, unwrap_lon

PIXELS_PER_CELL = 3
MAX_POLYGON_CELLS = 20_000  # more and drawing every cell as a polygon gets slow
# Resampled cell edges land on the nearest image pixel, so they're off by up to half a pixel.
# Zoomed in, a few pixels per cell makes that visible (and it shifts whenever the
# detail is reloaded), so small images are scaled up to about screen resolution.
MIN_PX = 2048


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
    None when an image has to do instead.
    """
    if grid.kind != "curvilinear" or grid.slice.downsampled:
        return None
    if grid.values.size > max_cells or min(grid.values.shape) < 2:
        return None
    lat, lon = cell_corners(grid.lat), cell_corners(unwrap_lon(grid.lon))
    ok = np.isfinite(lat) & np.isfinite(lon)
    if not ok.any():
        return None
    corners_ok = ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]
    values = np.where(corners_ok, np.asarray(grid.values, dtype=float), np.nan)
    colors = point_colors(values.ravel(), cmap, vmin, vmax)
    return Cells(values.shape, np.round(lat, 6).ravel().tolist(), np.round(lon, 6).ravel().tolist(),
                 colors, float(np.min(lat[ok])), float(np.min(lon[ok])),
                 float(np.max(lat[ok])), float(np.max(lon[ok])))


def _render_curvilinear(grid, cmap, vmin, vmax, max_px):
    lat, lon = grid.lat, grid.lon
    values = np.array(grid.values, dtype=float)
    values[_seam_mask(lon)] = np.nan  # cells straddling the antimeridian would smear across
    ok = np.isfinite(lat) & np.isfinite(lon)
    if not ok.any():
        return None
    x, y = lon, mercator_y(lat)
    ny, nx = values.shape
    width = int(np.clip(nx * PIXELS_PER_CELL, 256, max_px))
    height = int(np.clip(ny * PIXELS_PER_CELL, 256, max_px))
    width, height = _image_size(width, height, max_px)

    fig = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor="none")
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.pcolormesh(x, y, np.ma.masked_invalid(values), cmap=_cmap(cmap), norm=Normalize(vmin, vmax),
                  shading="nearest", antialiased=False)
    ax.autoscale(tight=True)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    y0, y1 = max(y0, mercator_y(-90)), min(y1, mercator_y(90))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, transparent=True)
    return Overlay(buf.getvalue(), float(inverse_mercator_y(y0)), float(x0),
                   float(inverse_mercator_y(y1)), float(x1))


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

