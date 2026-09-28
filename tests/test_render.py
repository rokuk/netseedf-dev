import io

import numpy as np
import pytest
import xarray as xr
from PIL import Image

from netseedf.core.coords import GeoGrid, find_geo, geo_grid
from netseedf.core.render import (
    cell_edges,
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


def test_regular_overlay_rows_follow_mercator():
    # Two latitude bands, 10..50 and 50..90 (clipped to Mercator's limit).
    lat = np.array([30.0, 70.0])
    values = np.array([[0.0], [1.0]])
    grid = GeoGrid("regular", lat, np.array([0.0]), values,
                   Slice(values, ("lat", "lon"), (1, 1), {}))
    overlay = render_overlay(grid, "Greys", 0, 1)
    assert (overlay.south, overlay.north) == (10, 85.0511)
    px = _pixels(overlay)
    height = px.shape[0]
    y_n, y_s = mercator_y(overlay.north), mercator_y(overlay.south)

    def row_of(lat_deg):
        return int((y_n - mercator_y(lat_deg)) / (y_n - y_s) * height)

    # Band boundary (lat 50, midway between the centres) must be where Mercator puts it.
    assert tuple(px[row_of(52), 0, :3]) != tuple(px[row_of(48), 0, :3])
    assert tuple(px[row_of(55), 0]) == tuple(px[row_of(80), 0])
    assert tuple(px[row_of(45), 0]) == tuple(px[row_of(15), 0])


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
