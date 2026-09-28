import numpy as np
import pytest
import xarray as xr

from netseedf.core.slicing import coord_values, extract, fixed_indices


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
