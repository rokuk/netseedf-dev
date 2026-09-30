import numpy as np
import pytest
import xarray as xr

from netseedf.core.slicing import (
    coord_values,
    extract,
    fixed_indices,
    is_time_dim,
    point_series,
    series_frame,
    time_dims,
)


@pytest.fixture
def salinity(samples):
    ds = xr.open_dataset(samples["four_d.nc"])
    yield ds["salinity"]
    ds.close()


def test_extract_2d_slice(salinity):
    sl = extract(salinity, ("lat", "lon"), {"time": 1, "depth": 2})
    assert sl.values.shape == (30, 40)
    assert sl.fixed == {"time": 1, "depth": 2}
    assert not sl.downsampled
    np.testing.assert_array_equal(sl.values, salinity.isel(time=1, depth=2).values)


def test_extract_respects_dim_order(salinity):
    sl = extract(salinity, ("lon", "lat"), {})
    assert sl.values.shape == (40, 30)
    np.testing.assert_array_equal(sl.values, salinity.isel(time=0, depth=0).values.T)


def test_out_of_range_indices_are_clamped(salinity):
    assert fixed_indices(salinity, ("lat", "lon"), {"time": 99, "depth": -3}) == {
        "time": 2, "depth": 0}


def test_downsampling(salinity):
    sl = extract(salinity, ("lat", "lon"), {}, max_size=10)
    assert sl.steps == (3, 4)
    assert sl.values.shape == (10, 10)
    assert sl.downsampled and sl.max_step == 4
    np.testing.assert_array_equal(sl.values, salinity.isel(time=0, depth=0).values[::3, ::4])


def test_coord_values(salinity):
    np.testing.assert_array_equal(coord_values(salinity, "depth"), [0.5, 5, 10, 50, 100])
    assert coord_values(salinity, "lat", slice(None, None, 10)).shape == (3,)


def test_coord_values_missing_for_plain_dimension(samples):
    with xr.open_dataset(samples["curvilinear.nc"]) as ds:
        assert coord_values(ds["temp"], "x") is None


def test_time_dims(samples):
    with xr.open_dataset(samples["four_d.nc"]) as ds:
        assert time_dims(ds["salinity"]) == ("time",)
        assert time_dims(ds["salinity"], exclude={"time": 0}) == ()
    with xr.open_dataset(samples["wrf_like.nc"]) as ds:  # "Time" has no coordinate
        assert time_dims(ds["T2"]) == ("Time",)
    with xr.open_dataset(samples["netcdf3.nc"], decode_times=False) as ds:  # units "days since"
        assert time_dims(ds["pr"]) == ("time",)
    with xr.open_dataset(samples["regular_global.nc"]) as ds:
        assert not is_time_dim(ds["sst"], "lat")


def test_point_series_keeps_other_dims_fixed(salinity):
    series = point_series(salinity, {"lat": 4, "lon": 7}, {"time": 2, "depth": 3})
    assert series.dims == ("time",)
    np.testing.assert_array_equal(series.values, salinity.isel(depth=3, lat=4, lon=7).values)


def test_series_frame(salinity):
    series = point_series(salinity, {"lat": 4, "lon": 7}, {"depth": 1})
    frame = series_frame(series, {"lat": 99.0, "station": "A"})
    assert frame.index.name == "time"
    assert len(frame) == salinity.sizes["time"]
    assert frame.columns[0] == "station"  # extra columns come first...
    assert (frame["lat"] == float(salinity.lat[4])).all()  # ...unless they're coordinates already
    assert (frame["depth"] == float(salinity.depth[1])).all()
    np.testing.assert_array_equal(frame["salinity"], series.values)


def test_load_selection_reads_distant_indices_as_blocks(salinity, monkeypatch):
    import xarray as xr

    from netseedf.core import slicing
    sub = salinity.isel(time=0, depth=0)
    read = []
    original = xr.DataArray.isel

    def spy(self, indexers=None, **kw):
        out = original(self, indexers, **kw)
        read.append(out.size)
        return out
    monkeypatch.setattr(xr.DataArray, "isel", spy)
    cols = np.array([0, 1, 2, 37, 38, 39])  # both ends of 40 longitudes
    values = slicing.load_selection(sub, (slice(0, 30, 1), cols))
    np.testing.assert_array_equal(values, sub.values[:, cols])
    assert max(read) <= 30 * 3  # two blocks of 3 columns, not all 40
