"""How long the slow parts take: opening files, reading slices, rendering maps.

Not run by default (they write a large sample file first):

    uv run pytest -m perf

Each test fails when its step takes longer than its budget. The budgets are about
five times what the steps took on a laptop (at least 0.25 s), so they catch
regressions that make something several times slower, not noise. Every read starts from a freshly
opened file, so nothing comes from HDF5's chunk cache.

On a slower machine, scale every budget, e.g. NETSEEDF_PERF_BUDGET_SCALE=3.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from netseedf.core.coords import GeoInfo, find_geo, geo_grid, geo_window
from netseedf.core.dataset import describe, open_file
from netseedf.core.render import render_overlay
from netseedf.core.slicing import extract, point_series, slow_layout, variable_stats
from sample_data import _field

pytestmark = pytest.mark.perf

# What the views ask for (see cartopy_map_view, web_map_view, table_view).
MAP_SIZE = 2000
CURVILINEAR_SIZE = 800
TABLE_BLOCK = 256

NT, NLAT, NLON = 24, 721, 1440  # a day of hourly 0.25° data, like ERA5
NY, NX = 600, 800  # curvilinear grid

TESTFILES = Path(__file__).parent / "testfiles"
BUDGET_SCALE = float(os.environ.get("NETSEEDF_PERF_BUDGET_SCALE", 1))
REAL_FILES = sorted(TESTFILES.glob("*.nc")) if TESTFILES.is_dir() else []


def _timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def _within(seconds, budget, what):
    budget *= BUDGET_SCALE
    assert seconds <= budget, f"{what} took {seconds:.3f} s, budget {budget:g} s"


@pytest.fixture(scope="module")
def big_file(tmp_path_factory):
    """A large regular grid stored two ways, and a large curvilinear grid."""
    path = tmp_path_factory.mktemp("perf") / "big.nc"
    time_ = pd.date_range("2024-01-01", periods=NT, freq="h")
    lat = np.linspace(90, -90, NLAT)
    lon = np.linspace(0, 359.75, NLON)
    # With noise, the values compress about as well as measured data (a smooth field compresses far better).
    rng = np.random.default_rng(0)
    regular = np.stack([_field(lat[:, None], lon[None, :], t / 4) for t in range(NT)]).astype("float32")
    regular += rng.normal(0, 0.5, regular.shape).astype("float32")
    j, i = np.meshgrid(np.arange(NY), np.arange(NX), indexing="ij")
    clat, clon = 30 + 0.05 * j + 0.01 * i, -20 + 0.06 * i - 0.01 * j
    curvilinear = np.stack([_field(clat, clon, t / 4) for t in range(NT)]).astype("float32")
    curvilinear += rng.normal(0, 0.5, curvilinear.shape).astype("float32")
    geo = {"units": "K"}
    ds = xr.Dataset(
        {
            # one time step per chunk: maps are quick, time series slow
            "t2m": (("time", "lat", "lon"), regular, geo),
            # chunks span all time steps: time series are quick, maps slow (see slow_layout)
            "t2m_by_time": (("time", "lat", "lon"), regular, geo),
            "temp": (("time", "y", "x"), curvilinear, {**geo, "coordinates": "clat clon"}),
        },
        coords={
            "time": time_,
            "lat": ("lat", lat, {"units": "degrees_north", "standard_name": "latitude"}),
            "lon": ("lon", lon, {"units": "degrees_east", "standard_name": "longitude"}),
            "clat": (("y", "x"), clat, {"units": "degrees_north", "standard_name": "latitude"}),
            "clon": (("y", "x"), clon, {"units": "degrees_east", "standard_name": "longitude"}),
        },
    )
    compressed = {"zlib": True, "complevel": 1}
    ds.to_netcdf(path, engine="netcdf4", encoding={
        "t2m": {**compressed, "chunksizes": (1, NLAT, NLON)},
        "t2m_by_time": {**compressed, "chunksizes": (NT, 91, 180)},
        "temp": {**compressed, "chunksizes": (1, NY, NX)},
    })
    return path


@pytest.fixture
def opened(big_file):
    f = open_file(big_file)
    yield f
    f.close()


def _variable(opened, name) -> tuple[xr.DataArray, GeoInfo]:
    ds = opened.dataset("/")
    geo = find_geo(ds[name], ds)
    assert geo is not None
    return ds[name], geo


def test_open(big_file):
    def open_and_describe():
        f = open_file(big_file)
        ds = f.dataset("/")
        for name in ds.data_vars:
            describe(ds[name])
            find_geo(ds[name], ds)
        return f

    f, seconds = _timed(open_and_describe)
    f.close()
    _within(seconds, 1.5, "Opening the file and finding its grids")


def test_regular_map(opened):
    da, geo = _variable(opened, "t2m")
    grid, read = _timed(geo_grid, da, geo, {"time": 5}, MAP_SIZE)
    _, render = _timed(render_overlay, grid, "viridis", 250, 300)
    _within(read, 0.25, "Reading a time step of a regular grid")
    _within(render, 0.75, "Rendering a regular grid for the web map")


def test_regular_map_with_slow_layout(opened):
    da, geo = _variable(opened, "t2m_by_time")
    assert slow_layout(da, geo.dims) is not None  # the case the views warn about
    _, seconds = _timed(geo_grid, da, geo, {"time": 5}, MAP_SIZE)
    _within(seconds, 1, "Reading a time step stored in chunks spanning all time steps")


def test_zoomed_detail(opened):
    da, geo = _variable(opened, "t2m")
    overview = geo_grid(da, geo, {"time": 5}, 400)
    window = geo_window(geo, overview, (35, 5, 50, 25))  # (south, west, north, east)
    assert window is not None
    _, seconds = _timed(geo_grid, da, geo, {"time": 5}, MAP_SIZE, window, like=overview)
    _within(seconds, 0.25, "Reading the zoomed-in detail")


def test_curvilinear_map(opened):
    da, geo = _variable(opened, "temp")
    assert geo.kind == "curvilinear"
    grid, read = _timed(geo_grid, da, geo, {"time": 5}, CURVILINEAR_SIZE)
    _, render = _timed(render_overlay, grid, "viridis", 250, 300)
    _within(read, 0.25, "Reading a time step of a curvilinear grid")
    _within(render, 2, "Rendering a curvilinear grid for the web map")


@pytest.mark.parametrize("name, budget", [("t2m_by_time", 0.25), ("t2m", 1)])
def test_point_time_series(opened, name, budget):
    da, _ = _variable(opened, name)
    series, seconds = _timed(lambda: point_series(da, {"lat": NLAT // 2, "lon": NLON // 3}, {}).values)
    assert series.shape == (NT,)
    _within(seconds, budget, f"Reading the time series of one grid point of {name}")


def test_table_block(opened):
    da, _ = _variable(opened, "t2m")
    window = {"lat": (300, 300 + TABLE_BLOCK), "lon": (600, 600 + TABLE_BLOCK)}
    _, seconds = _timed(extract, da, ("lat", "lon"), {"time": 5}, None, window)
    _within(seconds, 0.25, "Reading a block of the table")


def test_variable_stats(opened):
    da, _ = _variable(opened, "t2m")
    stats, seconds = _timed(variable_stats, da)
    assert stats.count == NT * NLAT * NLON
    _within(seconds, 2, "Statistics of the whole variable")


@pytest.mark.skipif(not REAL_FILES, reason="tests/testfiles isn't there")
@pytest.mark.parametrize("path", REAL_FILES, ids=lambda p: p.name)
def test_real_file_first_map(path):
    """Open a real-world file and draw the first step of its largest mappable variable."""
    def first_map():
        f = open_file(path)
        try:
            ds = f.dataset("/")
            mappable = [(ds[n], g) for n in ds.data_vars if (g := find_geo(ds[n], ds)) is not None]
            if not mappable:
                pytest.skip(f"{path.name} has nothing to put on a map")
            da, geo = max(mappable, key=lambda v: v[0].size)
            grid = geo_grid(da, geo, {}, MAP_SIZE)
            return render_overlay(grid, "viridis", *np.nanpercentile(grid.values, [2, 98]))
        finally:
            f.close()

    _, seconds = _timed(first_map)
    _within(seconds, 8, f"Opening {path.name} and drawing its first map")
