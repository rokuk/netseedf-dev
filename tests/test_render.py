import io

import numpy as np
import pytest
import xarray as xr
from PIL import Image

from netseedf.core.coords import GeoGrid, find_geo, geo_grid, normalize_lon, with_cyclic_column
from netseedf.core.gridlines import cell_corners
from netseedf.core.render import (
    cell_edges,
    cell_polygons,
    inverse_mercator_y,
    legend_colors,
    mercator_y,
    point_colors,
    render_overlay,
)
from netseedf.core.slicing import Slice


def _pixels(overlay):
    return np.asarray(Image.open(io.BytesIO(overlay.png)).convert("RGBA"))


def _grid(lat, lon, values):
    sl = Slice(values, ("lat", "lon"), (1, 1), {})
    return GeoGrid("regular", np.asarray(lat, float), np.asarray(lon, float), values, sl)


def test_mercator_round_trip():
    lat = np.array([-80.0, -45, 0, 30, 60, 85])
    np.testing.assert_allclose(inverse_mercator_y(mercator_y(lat)), lat)


def test_cell_edges():
    np.testing.assert_allclose(cell_edges([0, 10, 20]), [-5, 5, 15, 25])


def test_regular_overlay_bounds_are_cell_edges():
    grid = _grid([-45, 0, 45], [0, 90, 180, 270], np.arange(12.0).reshape(3, 4))
    overlay = render_overlay(grid, "viridis", 0, 11)
    assert (overlay.south, overlay.north) == (-67.5, 67.5)
    assert (overlay.west, overlay.east) == (-45, 315)


def test_regular_overlay_has_a_row_per_latitude_band():
    # Two latitude bands, 10..50 and 50..90 (clipped to Mercator's limit).
    lat = np.array([30.0, 70.0])
    values = np.array([[0.0], [1.0]])
    grid = GeoGrid("regular", lat, np.array([0.0]), values,
                   Slice(values, ("lat", "lon"), (1, 1), {}))
    overlay = render_overlay(grid, "Greys", 0, 1)
    assert (overlay.south, overlay.north) == (10, 85.0511)
    assert overlay.rows == [85.0511, 50, 10]  # the page stretches each row between these
    px = _pixels(overlay)
    assert px.shape[:2] == (2, 1)
    assert tuple(px[0, 0, :3]) == (0, 0, 0) and tuple(px[1, 0, :3]) == (255, 255, 255)


def test_regular_overlay_is_a_pixel_per_cell():
    lat, lon = np.arange(30) * 0.1 + 60.05, np.arange(40) * 0.1 + 5.05
    overlay = render_overlay(_grid(lat, lon, np.ones((30, 40))), "Greys", 0, 1)
    assert overlay.size == (40, 30)
    np.testing.assert_allclose(overlay.rows, cell_edges(lat)[::-1])
    np.testing.assert_allclose((overlay.west, overlay.east), (5, 9))


def test_uneven_columns_keep_a_pixel_each():
    lon = np.array([0.0, 1, 3, 7])
    overlay = render_overlay(_grid([0, 1], lon, np.arange(8.0).reshape(2, 4)), "Greys", 0, 7)
    assert overlay.size == (4, 2)
    assert overlay.cols == [-0.5, 0.5, 2, 5, 9]  # the page stretches each column between these


def test_thinned_grid_is_drawn_with_real_cell_edges(samples):
    with xr.open_dataset(samples["regular_global.nc"]) as ds:
        var = next(iter(ds.data_vars.values()))
        geo = find_geo(var, ds)
        grid = geo_grid(var, geo, {}, max_size=20)
        full_lat = cell_edges(np.sort(ds.lat.values))
        full_lon = cell_edges(np.sort(normalize_lon(ds.lon.values, grid.lon_0_360)[0]))
    assert grid.slice.downsampled
    overlay = render_overlay(grid, "viridis", *np.nanpercentile(grid.values, [0, 100]))
    assert overlay.size == (grid.lon.size, grid.lat.size)
    assert set(np.round(overlay.rows[1:-1], 6)) <= set(np.round(full_lat, 6))  # ends: Mercator's limit
    assert set(np.round(overlay.cols, 6)) <= set(np.round(full_lon, 6))
    # Every drawn cell holds the real cell it was sampled from, and the whole grid is covered.
    assert np.all((grid.lat_edges[:-1] < grid.lat) & (grid.lat < grid.lat_edges[1:]))
    assert np.all((grid.lon_edges[:-1] < grid.lon) & (grid.lon < grid.lon_edges[1:]))
    assert (grid.lon_edges[0], grid.lon_edges[-1]) == (full_lon[0], full_lon[-1])


def test_cyclic_column_meets_the_first_one(samples):
    with xr.open_dataset(samples["regular_global.nc"]) as ds:
        var = next(iter(ds.data_vars.values()))
        grid = with_cyclic_column(geo_grid(var, find_geo(var, ds), {}, max_size=20))
    assert grid.lon_edges.size == grid.lon.size + 1
    np.testing.assert_allclose(grid.lon_edges[-2:], grid.lon_edges[:2] + 360)


def test_nan_is_transparent():
    grid = _grid([0, 10], [0, 10], np.array([[np.nan, 1.0], [2.0, 3.0]]))
    px = _pixels(render_overlay(grid, "viridis", 0, 3))
    assert px[-1, 0, 3] == 0  # south-west cell
    assert px[0, -1, 3] == 255


def test_curvilinear_overlay(samples):
    with xr.open_dataset(samples["curvilinear.nc"]) as ds:
        grid = geo_grid(ds["temp"], find_geo(ds["temp"], ds), {})
    overlay = render_overlay(grid, "viridis", *np.nanpercentile(grid.values, [0, 100]))
    south, west, north, east = grid.bounds
    assert overlay.south < south < north < overlay.north
    assert overlay.west < west < east < overlay.east
    assert (_pixels(overlay)[..., 3] > 0).mean() > 0.3


def test_small_curvilinear_grid_becomes_polygons(samples):
    with xr.open_dataset(samples["curvilinear.nc"]) as ds:
        var = ds["temp"]
        grid = geo_grid(var, find_geo(var, ds), {})
    grid.values[0, 0] = np.nan
    cells = cell_polygons(grid, "viridis", 0, 30)
    ny, nx = grid.values.shape
    assert cells.shape == (ny, nx)
    assert len(cells.lat) == len(cells.lon) == (ny + 1) * (nx + 1)
    # The same corners the grid lines are drawn with.
    np.testing.assert_allclose(np.reshape(cells.lat, (ny + 1, nx + 1)), cell_corners(grid.lat), atol=1e-6)
    assert cells.colors[0] is None and all(c.startswith("#") for c in cells.colors[1:])
    assert cells.south < np.nanmin(grid.lat) < np.nanmax(grid.lat) < cells.north


def test_polygons_only_for_few_full_resolution_cells(samples):
    with xr.open_dataset(samples["curvilinear.nc"]) as ds:
        var, geo = ds["temp"], find_geo(ds["temp"], ds)
        full, thinned = geo_grid(var, geo, {}), geo_grid(var, geo, {}, max_size=10)
    assert cell_polygons(full, "viridis", 0, 30, max_cells=full.values.size - 1) is None
    assert thinned.slice.downsampled and cell_polygons(thinned, "viridis", 0, 30) is None
    regular = _grid([0, 1], [0, 1], np.ones((2, 2)))
    assert cell_polygons(regular, "viridis", 0, 1) is None


def test_polar_only_grid_cannot_be_shown():
    assert render_overlay(_grid([87, 89], [0, 10], np.ones((2, 2))), "viridis", 0, 1) is None


def test_point_colors():
    colors = point_colors(np.array([0.0, np.nan, 1.0]), "Greys", 0, 1)
    assert colors[1] is None
    assert colors[0] == "#ffffff" and colors[2] == "#000000"


def test_legend_colors():
    assert len(legend_colors("viridis", 8)) == 8


@pytest.mark.parametrize("name", ["regular_global.nc", "pacific.nc", "netcdf3.nc", "packed.nc"])
def test_sample_grids_render(samples, name):
    with xr.open_dataset(samples[name]) as ds:
        var = next(iter(ds.data_vars.values()))
        grid = geo_grid(var, find_geo(var, ds), {})
    overlay = render_overlay(grid, "viridis", *np.nanpercentile(grid.values, [0, 100]))
    assert overlay.west < overlay.east and overlay.south < overlay.north
    assert overlay.data_url.startswith("data:image/png;base64,")
