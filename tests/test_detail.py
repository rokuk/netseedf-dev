"""Zoomed-in views load the visible part at a finer resolution."""

import numpy as np
import pytest
import xarray as xr

from netseedf.core.coords import find_geo, geo_grid, geo_window
from netseedf.core.detail import DetailTracker, padded, resolution_note
from netseedf.core.slicing import extract, index_window, is_finer


@pytest.fixture
def ds(samples):
    handles = []

    def _open(name):
        d = xr.open_dataset(samples[name])
        handles.append(d)
        return d

    yield _open
    for d in handles:
        d.close()


def test_extract_window(ds):
    salinity = ds("four_d.nc")["salinity"]
    sl = extract(salinity, ("lat", "lon"), {"time": 1}, max_size=100, window={"lat": (5, 15)})
    assert sl.values.shape == (10, 40)
    assert sl.selection[0] == slice(5, 15, 1)
    np.testing.assert_array_equal(sl.index(0), np.arange(5, 15))
    np.testing.assert_array_equal(sl.values, salinity.isel(time=1, depth=0).values[5:15])


def test_extract_window_is_thinned_only_as_needed(ds):
    salinity = ds("four_d.nc")["salinity"]
    sl = extract(salinity, ("lat", "lon"), {}, max_size=10, window={"lon": (0, 20)})
    assert sl.steps == (3, 2)
    np.testing.assert_array_equal(sl.index(1), np.arange(0, 20, 2))


def test_extract_index_array_window(ds):
    precip = ds("points.nc")["precip"]
    idx = np.array([3, 7, 8, 30])
    sl = extract(precip, ("station",), {"time": 2}, window={"station": idx})
    np.testing.assert_array_equal(sl.index(0), idx)
    np.testing.assert_array_equal(sl.values, precip.isel(time=2).values[idx])


def test_is_finer():
    base = extract(xr.DataArray(np.zeros((100, 100)), dims=("y", "x")), ("y", "x"), {}, max_size=10)
    assert base.steps == (10, 10)
    assert is_finer(base, {"y": (0, 50), "x": (0, 100)}, 10)
    assert not is_finer(base, {"y": (0, 100), "x": (0, 100)}, 10)


def test_index_window():
    pos = np.arange(10.0)
    assert index_window(pos, 3.2, 6.7) == (3, 8)
    assert index_window(pos, 6.7, 3.2) == (3, 8)
    assert index_window(pos, 20, 30) is None


def test_regular_geo_window(ds):
    d = ds("regular_global.nc")
    geo = find_geo(d["sst"], d)
    base = geo_grid(d["sst"], geo, {}, max_size=30)  # every 3rd lat, 6th lon
    window = geo_window(geo, base, (40, 0, 60, 30))  # south, west, north, east
    lat_idx = np.flatnonzero((d.lat.values >= 40) & (d.lat.values <= 60))
    lon_idx = np.flatnonzero(d.lon.values <= 30)
    assert window["lat"][0] <= lat_idx.min() and window["lat"][1] > lat_idx.max()
    assert window["lon"][0] <= lon_idx.min() and window["lon"][1] > lon_idx.max()
    assert is_finer(base.slice, window, 30)
    detail = geo_grid(d["sst"], geo, {}, 30, window, like=base)
    assert not detail.slice.downsampled
    assert detail.bounds[0] <= 40 and detail.bounds[2] >= 60


def test_geo_window_across_the_dateline(ds):
    d = ds("regular_global.nc")
    geo = find_geo(d["sst"], d)
    base = geo_grid(d["sst"], geo, {}, max_size=30)
    window = geo_window(geo, base, (-10, 170, 10, 190))  # 170°E .. 170°W
    detail = geo_grid(d["sst"], geo, {}, 1000, window, like=base)
    # Both sides of the antimeridian are in the window (it wraps, so it spans the row).
    assert detail.lon.min() <= -170 and detail.lon.max() >= 170


def test_curvilinear_geo_window(ds):
    d = ds("curvilinear.nc")
    geo = find_geo(d["temp"], d)
    base = geo_grid(d["temp"], geo, {}, max_size=10)
    window = geo_window(geo, base, (45, 0, 50, 8))
    detail = geo_grid(d["temp"], geo, {}, 10_000, window, like=base)
    assert not detail.slice.downsampled
    inside = (detail.lat >= 45) & (detail.lat <= 50) & (detail.lon >= 0) & (detail.lon <= 8)
    full = d["lat"].values
    everything = ((full >= 45) & (full <= 50) & (d["lon"].values >= 0) & (d["lon"].values <= 8)).sum()
    assert inside.sum() == everything  # no cell in the box is missing from the detail


def test_point_window_finds_every_point(ds):
    d = ds("points.nc")
    geo = find_geo(d["elevation"], d)
    base = geo_grid(d["elevation"], geo, {}, max_size=5)  # only 5 of 50 stations
    window = geo_window(geo, base, (45, 0, 55, 15))
    lat, lon = d["lat"].values, d["lon"].values
    expected = np.flatnonzero((lat >= 45) & (lat <= 55) & (lon >= 0) & (lon <= 15))
    np.testing.assert_array_equal(window["station"], expected)
    detail = geo_grid(d["elevation"], geo, {}, 1000, window, like=base)
    np.testing.assert_allclose(detail.values, d["elevation"].values[expected])


def test_whole_world_in_view_needs_no_detail(ds):
    d = ds("regular_global.nc")
    geo = find_geo(d["sst"], d)
    base = geo_grid(d["sst"], geo, {}, max_size=30)
    window = geo_window(geo, base, (-135, -270, 135, 270))  # a padded view of the globe
    (lat0, lat1), (lon0, lon1) = window["lat"], window["lon"]
    assert lat0 <= 0 and lat1 >= 90 and lon0 <= 0 and lon1 >= 180
    assert not is_finer(base.slice, window, 30)


def test_nothing_in_view(ds):
    d = ds("four_d.nc")
    geo = find_geo(d["salinity"], d)
    base = geo_grid(d["salinity"], geo, {}, max_size=10)
    assert geo_window(geo, base, (-60, 100, -40, 120)) is None


def test_tracker():
    t = DetailTracker()
    assert not t.up_to_date((0, 10, 0, 10))
    t.remember((0, 10, 0, 10), padded((0, 10, 0, 10)))
    assert t.up_to_date((1, 11, 1, 11))  # small pan within the margin
    assert not t.up_to_date((5, 15, 0, 10))  # beyond the margin
    assert not t.up_to_date((2, 4, 2, 4))  # zoomed in a lot: finer data may fit now
    t.reset()
    assert not t.up_to_date((0, 10, 0, 10))


def test_resolution_note():
    arr = xr.DataArray(np.zeros((100, 100)), dims=("y", "x"))
    base = extract(arr, ("y", "x"), {}, max_size=10)
    full = extract(arr, ("y", "x"), {}, window={"y": (0, 10)})
    assert "zoom in" in resolution_note(base, None)
    assert resolution_note(base, full) == "  full resolution in view"
    assert resolution_note(full, None) == ""
