import numpy as np
import pytest
import xarray as xr

from netseedf.core.coords import GridLocator, find_geo, geo_grid, geo_window, normalize_lon, pick_point


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


def test_regular_grid(ds):
    d = ds("regular_global.nc")
    geo = find_geo(d["sst"], d)
    assert geo.kind == "regular"
    assert geo.dims == ("lat", "lon")


def test_curvilinear_grid(ds):
    d = ds("curvilinear.nc")
    geo = find_geo(d["temp"], d)
    assert geo.kind == "curvilinear"
    assert geo.dims == ("y", "x")
    assert (geo.lat.name, geo.lon.name) == ("lat", "lon")


def test_wrf_coordinates_follow_time_index(ds):
    d = ds("wrf_like.nc")
    geo = find_geo(d["T2"], d)
    assert geo.kind == "curvilinear"
    assert geo.dims == ("south_north", "west_east")
    grid = geo_grid(d["T2"], geo, {"Time": 1})
    assert grid.lat.shape == grid.values.shape == (20, 30)
    np.testing.assert_allclose(grid.values, d["T2"].isel(Time=1).values)


def test_points(ds):
    d = ds("points.nc")
    geo = find_geo(d["precip"], d)
    assert geo.kind == "points"
    assert geo.dims == ("station",)


def test_no_geo_for_timeseries(ds):
    d = ds("timeseries.nc")
    assert find_geo(d["discharge"], d) is None


def test_regular_grid_is_sorted_and_lon_normalized(ds):
    d = ds("regular_global.nc")
    grid = geo_grid(d["sst"], find_geo(d["sst"], d), {"time": 2})
    assert np.all(np.diff(grid.lat) > 0) and np.all(np.diff(grid.lon) > 0)
    assert grid.lon[0] == -180 and grid.lon[-1] == 178
    assert grid.is_global
    i, j = np.searchsorted(grid.lat, 89), np.searchsorted(grid.lon, 0)
    assert grid.values[i, j] == pytest.approx(float(d["sst"].sel(time=d.time[2], lat=89, lon=0)))


def test_dateline_box_stays_contiguous(ds):
    d = ds("pacific.nc")
    grid = geo_grid(d["sla"], find_geo(d["sla"], d), {})
    assert (grid.lon.min(), grid.lon.max()) == (150, 210)


def test_normalize_lon():
    lon, lon_0_360 = normalize_lon(np.array([0.0, 90, 180, 270]))
    np.testing.assert_array_equal(lon, [0, 90, -180, -90])
    assert not lon_0_360
    lon, lon_0_360 = normalize_lon(np.array([-170.0, 170]))
    np.testing.assert_array_equal(lon, [190, 170])
    assert lon_0_360
    np.testing.assert_array_equal(normalize_lon(np.array([-170.0, 170]), False)[0], [-170, 170])


def test_downsampled_grid(ds):
    d = ds("regular_global.nc")
    grid = geo_grid(d["sst"], find_geo(d["sst"], d), {}, max_size=45)
    assert grid.values.shape == (45, 45)
    assert grid.slice.steps == (2, 4)


def test_regular_locator(ds):
    d = ds("regular_global.nc")
    grid = geo_grid(d["sst"], find_geo(d["sst"], d), {})
    loc = GridLocator(grid)
    i, j = loc.nearest(45.4, 10.9)
    assert (grid.lat[i], grid.lon[j]) == (45, 10)
    i, j = loc.nearest(0, 179.5)  # wraps around to -180
    assert grid.lon[j] == -180
    assert loc.nearest(89.9, 0) is not None  # half a cell beyond the last centre


def test_regular_locator_outside_grid(ds):
    d = ds("four_d.nc")
    loc = GridLocator(geo_grid(d["salinity"], find_geo(d["salinity"], d), {}))
    assert loc.nearest(50, 0) is not None
    assert loc.nearest(10, 0) is None
    assert loc.nearest(50, 100) is None


def test_curvilinear_locator(ds):
    d = ds("curvilinear.nc")
    grid = geo_grid(d["temp"], find_geo(d["temp"], d), {})
    loc = GridLocator(grid)
    idx = loc.nearest(grid.lat[10, 20], grid.lon[10, 20])
    assert idx == (10, 20)
    assert loc.nearest(-30, 100) is None
    (_, value) = loc.value_at(grid.lat[5, 5], grid.lon[5, 5])
    assert value == grid.values[5, 5]


def test_pick_point_finds_the_full_resolution_cell(ds):
    d = ds("regular_global.nc")
    da, geo = d["sst"], find_geo(d["sst"], d)
    coarse = geo_grid(da, geo, {"time": 1}, max_size=45)  # every 2nd lat, 4th lon
    assert 47 not in coarse.lat and 14 not in coarse.lon
    picked = pick_point(da, geo, coarse, 47.3, 13.8, {"time": 1})
    assert (picked.lat, picked.lon) == (47, 14)
    assert picked.value == pytest.approx(float(da.isel(time=1, **picked.index)))
    assert float(da.lat[picked.index["lat"]]) == 47
    assert float(da.lon[picked.index["lon"]]) == 14


def test_pick_point_outside_data(ds):
    d = ds("four_d.nc")
    da, geo = d["salinity"], find_geo(d["salinity"], d)
    assert pick_point(da, geo, geo_grid(da, geo, {}), 10, 0, {}) is None


def test_pick_point_curvilinear_and_points(ds):
    d = ds("curvilinear.nc")
    da, geo = d["temp"], find_geo(d["temp"], d)
    grid = geo_grid(da, geo, {})
    picked = pick_point(da, geo, grid, grid.lat[7, 3], grid.lon[7, 3], {})
    assert picked.index == {"y": 7, "x": 3}
    d = ds("points.nc")
    da, geo = d["precip"], find_geo(d["precip"], d)
    lat, lon = float(d.lat[12]), float(d.lon[12])
    coarse = geo_grid(da, geo, {}, max_size=10)  # every 5th station
    assert 12 not in coarse.index[0]
    picked = pick_point(da, geo, coarse, lat, lon, {})
    assert picked.index == {"station": 12}
    assert (picked.lat, picked.lon) == pytest.approx((lat, lon))


def test_window_across_the_ends_of_stored_longitudes(ds):
    """lon is stored 0..358: a view around 0° has columns at both ends, not everything between."""
    d = ds("regular_global.nc")
    da, geo = d["sst"], find_geo(d["sst"], d)
    coarse = geo_grid(da, geo, {"time": 1}, max_size=45)  # every 4th column
    window = geo_window(geo, coarse, (40, -21, 60, 21))
    cols = window["lon"]
    assert isinstance(cols, np.ndarray) and len(cols) < 40  # of 180 columns
    assert cols.min() == 0 and cols.max() == 179
    detail = geo_grid(da, geo, {"time": 1}, max_size=45, window=window, like=coarse)
    assert detail.slice.steps[1] == 1  # the detail is at full resolution along lon
    assert np.all(np.diff(detail.lon) > 0) and detail.lon.min() < -20 and detail.lon.max() > 20
    for j, lon in enumerate(detail.lon):
        expected = da.isel(time=1).sel(lon=lon % 360).values[detail.index[0]]
        np.testing.assert_array_equal(detail.values[:, j], expected)


def test_window_within_stored_longitudes_is_a_range(ds):
    d = ds("regular_global.nc")
    geo = find_geo(d["sst"], d)
    coarse = geo_grid(d["sst"], geo, {}, max_size=45)
    window = geo_window(geo, coarse, (40, 30, 60, 60))
    assert isinstance(window["lon"], tuple) and isinstance(window["lat"], tuple)


def test_grid_mapping_gives_projection(ds):
    d = ds("polar_projected.nc")
    geo = find_geo(d["ice"], d)
    assert geo.kind == "curvilinear"
    assert geo.projection is not None and geo.projection.crs.is_projected
    np.testing.assert_allclose(geo.projection.x, d["x"].values * 1000)  # km to metres
    np.testing.assert_allclose(geo.projection.y, d["y"].values * 1000)


def test_projected_cell_edges(ds):
    d = ds("polar_projected.nc")
    geo = find_geo(d["ice"], d)
    grid = geo_grid(d["ice"], geo, {"time": 0})
    x_edges, y_edges = grid.xy_edges
    np.testing.assert_allclose(x_edges, np.arange(-30, 31) * 200_000.0)
    np.testing.assert_allclose(y_edges, np.arange(30, -31, -1) * 200_000.0)  # descending, like y
    # Thinned out, the drawn cells still have real cell edges, and cover the whole grid.
    coarse = geo_grid(d["ice"], geo, {"time": 0}, max_size=20)
    assert coarse.slice.downsampled
    cx, cy = coarse.xy_edges
    assert (cx.size, cy.size) == (coarse.values.shape[1] + 1, coarse.values.shape[0] + 1)
    assert set(cx) <= set(x_edges) and set(cy) <= set(y_edges)
    assert (cx[0], cx[-1], cy[0], cy[-1]) == (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])


def test_projected_grid_centred_on_its_pole(ds):
    d = ds("polar_projected.nc")
    lat, lon = find_geo(d["ice"], d).projection.centre()
    assert lat == pytest.approx(90)
    assert lon == pytest.approx(0, abs=1e-6)  # the meridian going down from the pole


@pytest.mark.parametrize("change", [
    {"grid_mapping_name": "latitude_longitude"},  # not a projection: coordinates in degrees
    {"grid_mapping_name": "no_such_projection"},
])
def test_no_projection_without_usable_mapping(ds, change):
    d = ds("polar_projected.nc").copy()
    d["crs"].attrs = change
    geo = find_geo(d["ice"], d)
    assert geo.kind == "curvilinear" and geo.projection is None
    assert geo_grid(d["ice"], geo, {"time": 0}).xy_edges is None


def test_no_projection_without_projection_coordinates(ds):
    d = ds("polar_projected.nc").copy()
    d["x"].attrs["units"] = "furlong"
    assert find_geo(d["ice"], d).projection is None
    d = ds("polar_projected.nc").copy()
    del d["ice"].attrs["grid_mapping"]
    assert find_geo(d["ice"], d).projection is None
