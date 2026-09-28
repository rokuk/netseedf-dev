"""Rendering map data as images and colours for the Leaflet web map.

Leaflet stretches an image overlay linearly between its bounds in Web
Mercator, so the image rows are laid out in Mercator y, not in latitude.
Otherwise the overlay would drift away from the basemap towards the poles.
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

from netseedf.core.coords import MERCATOR_MAX_LAT, GeoGrid

PIXELS_PER_CELL = 3


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


def cell_edges(centres):
    """Edges between 1D cell centres, extending half a cell at both ends."""
    c = np.asarray(centres, dtype=float)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = (c[:-1] + c[1:]) / 2
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


def _render_regular(grid, cmap, vmin, vmax, max_px):
    lat_edges = np.clip(cell_edges(grid.lat), -90, 90)
    lon_edges = cell_edges(grid.lon)
    south, north = max(lat_edges[0], -MERCATOR_MAX_LAT), min(lat_edges[-1], MERCATOR_MAX_LAT)
    if south >= north:
        return None  # all of it is too close to a pole for Web Mercator
    west, east = lon_edges[0], lon_edges[-1]
    y_s, y_n = mercator_y(south), mercator_y(north)

    # Enough pixels for the smallest cell (in Mercator) to get a few of them.
    y_edges = mercator_y(lat_edges)
    min_dy = np.min(np.diff(y_edges)[np.diff(y_edges) > 0], initial=y_n - y_s)
    min_dx = np.min(np.diff(lon_edges))
    height = int(np.clip(np.ceil((y_n - y_s) / min_dy * PIXELS_PER_CELL), 1, max_px))
    width = int(np.clip(np.ceil((east - west) / min_dx * PIXELS_PER_CELL), 1, max_px))

    # Nearest cell for each pixel centre; top row is north.
    x = west + (np.arange(width) + 0.5) * (east - west) / width
    lat = inverse_mercator_y(y_n - (np.arange(height) + 0.5) * (y_n - y_s) / height)
    cols = np.searchsorted(lon_edges, x, side="right") - 1
    rows = np.searchsorted(lat_edges, lat, side="right") - 1
    cols_ok = (cols >= 0) & (cols < grid.lon.size)
    rows_ok = (rows >= 0) & (rows < grid.lat.size)
    img = grid.values[np.ix_(np.clip(rows, 0, grid.lat.size - 1), np.clip(cols, 0, grid.lon.size - 1))]
    img = np.where(rows_ok[:, None] & cols_ok[None, :], img, np.nan)

    rgba = _cmap(cmap)(Normalize(vmin, vmax)(np.ma.masked_invalid(img)), bytes=True)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="png")
    return Overlay(buf.getvalue(), float(south), float(west), float(north), float(east))


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

