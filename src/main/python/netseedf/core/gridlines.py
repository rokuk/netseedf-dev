"""Outlines of a variable's grid cells, as lines to draw over a map."""

import numpy as np
import xarray as xr

from netseedf.core.coords import GeoGrid, GeoInfo, fill_missing_positions, geo_grid, geo_window, regular_edges

Line = list[list[float]]  # [[lat, lon], ...]


def grid_lines(da: xr.DataArray, geo: GeoInfo, grid: GeoGrid, indices, box, max_lines) -> list[Line] | None:
    """Full-resolution cell edges inside `box` (south, west, north, east).

    `grid` is the (maybe downsampled) grid on the map; its longitude
    convention is used for the lines. Returns None when more than `max_lines`
    lines would be needed: drawing only some of them would misrepresent the grid.
    Points (stations) have no cells, so no lines.
    """
    if grid.kind == "regular":
        return _regular_lines(geo, grid, box, max_lines)
    if grid.kind == "curvilinear":
        return _curvilinear_lines(da, geo, grid, indices, box, max_lines)
    return []


def _regular_lines(geo, grid, box, max_lines):
    south, west, north, east = box
    (_, lat_edges), (_, lon_edges) = regular_edges(geo, grid.lon_0_360)
    lat_edges = np.clip(lat_edges, -90, 90)
    lats = lat_edges[(lat_edges >= south) & (lat_edges <= north)]
    # The map may show more than one copy of the world, so repeat the grid every 360°.
    shifts = [k * 360.0 for k in range(int(np.floor((west - lon_edges[-1]) / 360)),
                                       int(np.ceil((east - lon_edges[0]) / 360)) + 1)]
    # (unique: a global grid's first and last edge meet at the seam)
    lons = np.unique([x for k in shifts for x in lon_edges + k if west <= x <= east])
    if len(lats) + len(lons) > max_lines:
        return None
    lo_lat, hi_lat = max(south, lat_edges[0]), min(north, lat_edges[-1])
    lines = [[[lo_lat, x], [hi_lat, x]] for x in lons] if lo_lat < hi_lat else []
    for k in shifts:
        lo_lon, hi_lon = max(west, lon_edges[0] + k), min(east, lon_edges[-1] + k)
        if lo_lon < hi_lon:
            lines += [[[y, lo_lon], [y, hi_lon]] for y in lats]
    return [[[float(a), float(b)] for a, b in line] for line in lines]


def _curvilinear_lines(da, geo, grid, indices, box, max_lines):
    window = geo_window(geo, grid, box)
    if window is None:
        return []
    ny, nx = (stop - start for start, stop in window.values())
    if ny + nx + 2 > max_lines:
        return None
    fine = geo_grid(da, geo, indices, None, window, like=grid)
    if min(fine.lat.shape) < 2:
        return []
    lat, lon = grid_corners(fine)
    rows = [np.column_stack([lat[i], lon[i]]) for i in range(lat.shape[0])]
    cols = [np.column_stack([lat[:, j], lon[:, j]]) for j in range(lat.shape[1])]
    return [part for line in rows + cols for part in _finite_runs(line)]


def grid_corners(grid: GeoGrid) -> tuple[np.ndarray, np.ndarray]:
    """(lat, lon) of a curvilinear grid's cell corners: from its bounds, or estimated.

    Longitudes have no jumps of 360 degrees. Where positions are missing, corners
    come from the nearest valid ones; those only amid missing positions are NaN.
    """
    if grid.corners is not None:
        return grid.corners
    lat, lon, _ = fill_missing_positions(grid.lat, grid.lon, grid.values)
    lat, lon = cell_corners(lat), cell_corners(unwrap_lon(lon))
    missing = ~(np.isfinite(grid.lat) & np.isfinite(grid.lon))
    if missing.any():
        m = np.pad(missing, 1, constant_values=True)
        lost = m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:]
        lat, lon = np.where(lost, np.nan, lat), np.where(lost, np.nan, lon)
    return lat, lon


def unwrap_lon(lon: np.ndarray) -> np.ndarray:
    """2D longitudes without jumps of 360° (across the antimeridian), for drawing cells."""
    return np.rad2deg(np.unwrap(np.unwrap(np.deg2rad(lon), axis=1), axis=0))


def cell_corners(centres: np.ndarray) -> np.ndarray:
    """Corners (ny + 1, nx + 1) of cells with 2D centres (ny, nx), both at least 2."""
    ny, nx = centres.shape
    p = np.empty((ny + 2, nx + 2))
    p[1:-1, 1:-1] = centres
    p[0, 1:-1] = 2 * centres[0] - centres[1]  # one extrapolated row/column on each side
    p[-1, 1:-1] = 2 * centres[-1] - centres[-2]
    p[:, 0] = 2 * p[:, 1] - p[:, 2]
    p[:, -1] = 2 * p[:, -2] - p[:, -3]
    return (p[:-1, :-1] + p[1:, :-1] + p[:-1, 1:] + p[1:, 1:]) / 4


def _finite_runs(points: np.ndarray) -> list[Line]:
    """The line split where coordinates are missing, as lists (for JSON)."""
    ok = np.isfinite(points).all(axis=1)
    runs, start = [], None
    for i, good in enumerate([*ok, False]):
        if good and start is None:
            start = i
        elif not good and start is not None:
            if i - start >= 2:
                runs.append(points[start:i].tolist())
            start = None
    return runs
