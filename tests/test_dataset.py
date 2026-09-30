import numpy as np
import pytest

from netseedf.core.dataset import (
    data_variables,
    describe,
    group_header,
    open_file,
    variable_header,
)
from netseedf.core.formatting import format_value


@pytest.fixture
def opened(samples):
    files = []

    def _open(name):
        f = open_file(samples[name])
        files.append(f)
        return f

    yield _open
    for f in files:
        f.close()


def test_regular_file(opened):
    f = opened("regular_global.nc")
    assert list(f.groups) == ["/"]
    assert [v.name for v in data_variables(f.dataset())] == ["sst"]
    assert f.warnings == []
    info = describe(f.variable("/", "sst"))
    assert info.dims == ("time", "lat", "lon")
    assert info.shape == (4, 90, 180)
    assert info.units == "K"
    assert info.long_name == "Sea surface temperature"
    assert info.dims_text == "(time=4, lat=90, lon=180)"


def test_groups_inherit_root_coordinates(opened):
    f = opened("groups.nc")
    assert list(f.groups) == ["/", "/observations", "/observations/station_b"]
    wind = f.variable("/observations/station_b", "wind_speed")
    assert "time" in wind.coords
    assert format_value(wind.coords["time"].values[1]) == "2021-03-01 01:00"


def test_undecodable_times_fall_back_to_raw_values(opened):
    f = opened("bad_time.nc")
    assert any("raw numbers" in w for w in f.warnings)
    assert f.variable("/", "time").dtype.kind == "f"


def test_packed_values_are_unpacked_and_masked(opened):
    t2m = opened("packed.nc").variable("/", "t2m").values
    assert np.isnan(t2m[0, 0])
    assert 250 < np.nanmean(t2m) < 300


def test_noleap_calendar(opened):
    time = opened("noleap.nc").variable("/", "time").values
    assert format_value(time[0]) == "1850-01-16"


def test_netcdf3_file(opened):
    assert opened("netcdf3.nc").variable("/", "pr").shape == (2, 25, 72)


def test_group_header(samples):
    header = group_header(samples["groups.nc"])
    assert header.attrs["title"] == "Grouped file"
    assert [(d.name, d.size) for d in header.dimensions] == [("time", 5)]
    obs = group_header(samples["groups.nc"], "/observations")
    temperature = next(v for v in obs.variables if v.name == "air_temperature")
    assert (temperature.dtype, temperature.dims) == ("float", ("time",))


def test_header_of_packed_file_shows_disk_types_and_all_attributes(samples):
    t2m = next(v for v in group_header(samples["packed.nc"]).variables if v.name == "t2m")
    assert t2m.dtype == "short"
    assert t2m.attrs["scale_factor"] == pytest.approx(0.01)  # xarray keeps this in encoding
    assert "_FillValue" in t2m.attrs


def test_variable_header_finds_inherited_coordinate(samples):
    header = variable_header(samples["groups.nc"], "/observations", "time")
    assert header.dims_text == "(time=5)"
    assert header.attrs["units"] == "hours since 2021-03-01 00:00"
    assert variable_header(samples["groups.nc"], "/observations", "no_such_variable") is None


def test_repeated_dimension_gets_its_own_name(tmp_path):
    import netCDF4

    path = tmp_path / "matrix.nc"
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("level", 3)
        nc.createVariable("level", "f8", ("level",))[:] = [1000, 850, 500]
        nc.createVariable("kernel", "f4", ("level", "level"))[:] = np.arange(9).reshape(3, 3)
    f = open_file(path)
    kernel = f.variable("/", "kernel")
    assert kernel.dims == ("level", "level_2")
    assert list(kernel.level_2.values) == [1000, 850, 500]
    assert float(kernel.isel(level=1, level_2=2)) == 5
    f.close()
