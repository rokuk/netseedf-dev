import numpy as np
import pytest

from netseedf.core.dataset import (
    data_variables,
    describe,
    header_text,
    open_file,
    variable_header_text,
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


def test_header_text(samples):
    text = header_text(samples["groups.nc"])
    assert text.startswith("netcdf groups {")
    assert "\ttime = 5 ;" in text
    assert "group: observations {" in text
    assert "group: station_b {" in text
    assert ':title = "Grouped file" ;' in text
    assert "float air_temperature(time) ;" in text


def test_header_of_packed_file_shows_disk_types(samples):
    text = header_text(samples["packed.nc"])
    assert "short t2m(lat, lon) ;" in text
    assert "t2m:scale_factor = 0.01" in text


def test_variable_header_finds_inherited_coordinate(samples):
    text = variable_header_text(samples["groups.nc"], "/observations", "time")
    assert text.startswith("double time(time) ;")
    assert 'time:units = "hours since 2021-03-01 00:00" ;' in text
