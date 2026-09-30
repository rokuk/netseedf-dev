"""Does NetSeeDF read files the way the CF conventions say they're meant to be read?

Section numbers refer to https://cfconventions.org/cf-conventions/cf-conventions.html
(1.14 draft at the time of writing). Each test writes a small file that uses
one feature of the conventions, opens it the way the app does and checks
what the viewer makes of it.
"""

import netCDF4
import numpy as np
import pandas as pd
import pytest

from netseedf.core.coords import GridLocator, fill_missing_positions, find_geo, geo_grid, pick_point
from netseedf.core.dataset import describe, open_file
from netseedf.core.formatting import format_value, index_label, value_text, variable_label
from netseedf.core.gridlines import grid_lines
from netseedf.core.render import cell_polygons, render_overlay
from netseedf.core.slicing import as_float, extract, is_time_dim, point_series, series_frame, time_dims


@pytest.fixture
def cf_file(tmp_path):
    """Write a file with `build(nc)` and open it like the app does."""
    files = []

    def _open(build):
        path = tmp_path / f"cf{len(files)}.nc"
        with netCDF4.Dataset(path, "w") as nc:
            build(nc)
        files.append(open_file(path))
        return files[-1]

    yield _open
    for f in files:
        f.close()


def var(nc, name, dtype, dims=(), values=None, fill_value=None, **attrs):
    """Variable with attributes; `values` are written as they should be stored (no packing)."""
    v = nc.createVariable(name, dtype, dims, fill_value=fill_value)
    v.set_auto_maskandscale(False)
    v.setncatts(attrs)
    if values is not None:
        v[...] = values
    return v


def dim(nc, name, values, dtype="f8", **attrs):
    """Dimension plus its coordinate variable."""
    nc.createDimension(name, len(values))
    return var(nc, name, dtype, (name,), values, **attrs)


def lat_lon_grid(nc, lat=(10.0, 20.0, 30.0), lon=(0.0, 5.0, 10.0, 15.0)):
    dim(nc, "lat", lat, units="degrees_north", standard_name="latitude")
    dim(nc, "lon", lon, units="degrees_east", standard_name="longitude")


def dates(values):
    return [format_value(v) for v in values]


# --- 2.2 Data types ---------------------------------------------------------------


def test_netcdf4_integer_and_string_types(cf_file):
    def build(nc):
        nc.createDimension("n", 3)
        var(nc, "u16", "u2", ("n",), [0, 40000, 65535])
        var(nc, "i64", "i8", ("n",), [-(2**40), 0, 2**40])
        var(nc, "u64", "u8", ("n",), [0, 1, 2**40])
        names = nc.createVariable("name", str, ("n",))
        names[:] = np.array(["a", "bb", "ccc"], dtype=object)

    f = cf_file(build)
    np.testing.assert_array_equal(as_float(f.variable("/", "u16").values), [0, 40000, 65535])
    np.testing.assert_array_equal(as_float(f.variable("/", "i64").values), [-(2**40), 0, 2**40])
    np.testing.assert_array_equal(as_float(f.variable("/", "u64").values), [0, 1, 2**40])
    assert list(f.variable("/", "name").values) == ["a", "bb", "ccc"]


# --- 2.4 Dimensions: T, Z, Y, X order is only recommended --------------------------


def test_dimension_order_is_not_assumed(cf_file):
    def build(nc):
        dim(nc, "lon", [30.0, 20.0, 10.0], units="degrees_east")  # decreasing is monotonic too
        dim(nc, "lat", [-10.0, 0.0], units="degrees_north")
        dim(nc, "time", [0.0, 1.0], units="days since 2000-01-01")
        values = np.arange(12, dtype="f4").reshape(3, 2, 2)
        var(nc, "v", "f4", ("lon", "lat", "time"), values)

    f = cf_file(build)
    da = f.variable("/", "v")
    geo = find_geo(da, f.dataset())
    assert (geo.kind, geo.dims) == ("regular", ("lat", "lon"))
    assert time_dims(da) == ("time",)
    grid = geo_grid(da, geo, {"time": 1})
    for i, lat in enumerate(grid.lat):
        for j, lon in enumerate(grid.lon):
            assert grid.values[i, j] == float(da.sel(lat=lat, lon=lon).isel(time=1))


# --- 2.5.1 Missing data, valid and actual range of data ----------------------------


def test_fill_value_and_missing_values_are_missing(cf_file):
    def build(nc):
        nc.createDimension("n", 5)
        var(nc, "scalar", "f4", ("n",), [1, -999, 3, 4, 5], missing_value=np.float32(-999))
        var(nc, "vector", "f4", ("n",), [1, -999, -998, 4, 5],
            missing_value=np.array([-999, -998], "f4"))
        var(nc, "both", "f4", ("n",), [1, -1, -999, 4, 5], fill_value=np.float32(-1),
            missing_value=np.float32(-999))

    f = cf_file(build)
    for name, missing in (("scalar", [1]), ("vector", [1, 2]), ("both", [1, 2])):
        values = f.variable("/", name).values
        assert list(np.flatnonzero(np.isnan(values))) == missing, name
    assert f.warnings == []  # several missing values are fine, nothing to warn about


@pytest.mark.parametrize("attrs", [
    {"valid_min": np.float32(0)},
    {"valid_max": np.float32(10)},
    {"valid_range": np.array([0, 10], "f4")},
], ids=["valid_min", "valid_max", "valid_range"])
def test_values_outside_valid_range_are_missing(cf_file, attrs):
    def build(nc):
        nc.createDimension("n", 4)
        var(nc, "v", "f4", ("n",), [-5, 1, 5, 20], **attrs)

    values = cf_file(build).variable("/", "v").values
    expected = {"valid_min": [0], "valid_max": [3], "valid_range": [0, 3]}[next(iter(attrs))]
    assert list(np.flatnonzero(np.isnan(values))) == expected


def test_valid_range_of_integers(cf_file):
    def build(nc):
        nc.createDimension("n", 4)
        var(nc, "v", "i2", ("n",), [-5, 1, 5, 20], valid_range=np.array([0, 10], "i2"))

    values = cf_file(build).variable("/", "v").values
    assert values.dtype.kind == "f"
    np.testing.assert_array_equal(values, [np.nan, 1, 5, np.nan])


def test_masking_is_lazy_and_follows_indexing(cf_file):
    def build(nc):
        nc.createDimension("y", 3)
        nc.createDimension("x", 4)
        var(nc, "v", "f4", ("y", "x"), np.arange(12).reshape(3, 4), valid_max=np.float32(6))

    da = cf_file(build).variable("/", "v")
    assert not da.variable._in_memory
    np.testing.assert_array_equal(da.isel(y=[0, 2], x=[1, 3]).values, [[1, 3], [np.nan, np.nan]])
    sl = extract(da, ("y", "x"), {}, max_size=2)  # strided read
    np.testing.assert_array_equal(sl.values, [[0, 2], [np.nan, np.nan]])


def test_default_fill_value_is_missing(cf_file):
    def build(nc):
        nc.createDimension("n", 4)
        v = nc.createVariable("v", "f4", ("n",))  # no _FillValue attribute
        v[:2] = [1, 2]
        w = nc.createVariable("packed", "i2", ("n",))
        w.set_auto_maskandscale(False)
        w.scale_factor = np.float32(0.5)
        w[:2] = [2, 4]
        nc.createVariable("count", "i4", ("n",))[:2] = [1, 2]

    f = cf_file(build)
    values = f.variable("/", "v").values
    assert values[0] == 1 and np.isnan(values[2:]).all()
    np.testing.assert_array_equal(f.variable("/", "packed").values, [1, 2, np.nan, np.nan])
    # Integers aren't turned into floats just in case some values were never written.
    assert f.variable("/", "count").dtype == np.int32


def test_missing_auxiliary_coordinates(cf_file):
    """Data can't be placed where its auxiliary coordinates are missing."""
    def build(nc):
        nc.createDimension("station", 4)
        var(nc, "lat", "f8", ("station",), [10, -999, 30, 40], fill_value=-999.0, units="degrees_north")
        var(nc, "lon", "f8", ("station",), [1, 2, 3, 4], fill_value=-999.0, units="degrees_east")
        var(nc, "p", "f4", ("station",), [1, 2, 3, 4], coordinates="lat lon")

    f = cf_file(build)
    da = f.variable("/", "p")
    geo = find_geo(da, f.dataset())
    grid = geo_grid(da, geo, {})
    assert np.isnan(grid.lat[1])
    assert np.isfinite(grid.bounds).all()
    assert GridLocator(grid).nearest(20, 2) is None
    assert pick_point(da, geo, grid, 20, 2, {}) is None


# --- 2.7 Groups ----------------------------------------------------------------------


@pytest.mark.filterwarnings("error:The input coordinates to pcolormesh:UserWarning")
def test_missing_curvilinear_coordinates_can_be_drawn(cf_file):
    """Swaths often have missing lat/lon at the edges; the rest of the grid is still drawn."""
    j, i = np.mgrid[0:6, 0:8]
    true_lat, true_lon = 40 + j + 0.1 * i, 10 + i + 0.1 * j

    def build(nc):
        nc.createDimension("y", 6)
        nc.createDimension("x", 8)
        lat, lon = true_lat.copy(), true_lon.copy()
        lat[0, :3] = lon[0, :3] = lat[:, 5:] = lon[:, 5:] = -999
        var(nc, "lat", "f8", ("y", "x"), lat, fill_value=-999.0, units="degrees_north")
        var(nc, "lon", "f8", ("y", "x"), lon, fill_value=-999.0, units="degrees_east")
        var(nc, "v", "f4", ("y", "x"), np.arange(48).reshape(6, 8), coordinates="lat lon")

    f = cf_file(build)
    da = f.variable("/", "v")
    grid = geo_grid(da, find_geo(da, f.dataset()), {})
    assert np.isnan(grid.lat[0, 0]) and np.isnan(grid.lat[:, 5:]).all()
    lat, lon, values = fill_missing_positions(grid.lat, grid.lon, grid.values)
    # The grid is continued where positions are missing (here: exactly, it's linear), so
    # the cells next to them keep their size, and a mesh its shape.
    np.testing.assert_allclose(lat, true_lat)
    np.testing.assert_allclose(lon, true_lon)
    assert np.isnan(values[0, :3]).all() and np.isnan(values[:, 5:]).all()
    np.testing.assert_array_equal(values[1:, :5], grid.values[1:, :5])  # the rest is unchanged
    overlay = render_overlay(grid, "viridis", 0, 47)  # (pcolormesh doesn't warn)
    # The image covers the cells with positions, not the made-up ones east of them.
    assert overlay.east == pytest.approx(true_lon[:, 4].max() + 0.5, abs=0.1)
    assert cell_polygons(grid, "viridis", 0, 47) is not None


def _curvilinear_in_root(nc):
    nc.createDimension("y", 3)
    nc.createDimension("x", 4)
    j, i = np.mgrid[0:3, 0:4]
    var(nc, "lat", "f8", ("y", "x"), 45 + j + 0.1 * i, units="degrees_north")
    var(nc, "lon", "f8", ("y", "x"), 10 + i + 0.1 * j, units="degrees_east")


def test_coordinates_found_in_parent_group(cf_file):
    def build(nc):
        _curvilinear_in_root(nc)
        var(nc.createGroup("model"), "tas", "f4", ("y", "x"), np.ones((3, 4)), coordinates="lat lon")

    f = cf_file(build)
    geo = find_geo(f.variable("/model", "tas"), f.dataset("/model"))
    assert geo is not None and (geo.kind, geo.lat.name, geo.lon.name) == ("curvilinear", "lat", "lon")


@pytest.mark.parametrize("coordinates", ["/lat /lon", "../../lat ../../lon"])
def test_coordinates_given_as_paths(cf_file, coordinates):
    def build(nc):
        _curvilinear_in_root(nc)
        group = nc.createGroup("model").createGroup("run1")
        var(group, "tas", "f4", ("y", "x"), np.ones((3, 4)), coordinates=coordinates)

    f = cf_file(build)
    geo = find_geo(f.variable("/model/run1", "tas"), f.dataset("/model/run1"))
    assert geo is not None and (geo.kind, geo.lat.name) == ("curvilinear", "lat")


# --- 3.1-3.3 units, long_name, standard_name ----------------------------------------------------


def test_labels_use_long_name_then_standard_name(cf_file):
    def build(nc):
        nc.createDimension("n", 2)
        var(nc, "a", "f4", ("n",), [1, 2], long_name="Air temperature", standard_name="air_temperature",
            units="K")
        var(nc, "b", "f4", ("n",), [1, 2], standard_name="air_temperature", units="K")
        var(nc, "c", "f4", ("n",), [1, 2])

    f = cf_file(build)
    assert variable_label(f.variable("/", "a")) == "Air temperature [K]"
    assert variable_label(f.variable("/", "b")) == "air_temperature [K]"
    assert variable_label(f.variable("/", "c")) == "c"
    assert describe(f.variable("/", "b")).long_name == "air_temperature"


def test_celsius_units_display_as_degree_sign(cf_file):
    def build(nc):
        nc.createDimension("n", 2)
        var(nc, "sst", "f4", ("n",), [1, 2], long_name="Sea temperature", units="degree_Celsius")

    da = cf_file(build).variable("/", "sst")
    assert variable_label(da) == "Sea temperature [°C]"
    assert value_text(da, 1.5) == "1.5 °C"


# --- 4.1 Latitude, 4.2 Longitude ---------------------------------------------------------------


@pytest.mark.parametrize(("lat_units", "lon_units"), [
    ("degrees_north", "degrees_east"),
    ("degree_north", "degree_east"),
    ("degree_N", "degree_E"),
    ("degrees_N", "degrees_E"),
    ("degreeN", "degreeE"),
    ("degreesN", "degreesE"),
])
def test_lat_lon_identified_by_units_alone(cf_file, lat_units, lon_units):
    def build(nc):
        dim(nc, "a", [10.0, 20.0, 30.0], units=lat_units)  # names that give nothing away
        dim(nc, "b", [0.0, 5.0], units=lon_units)
        var(nc, "v", "f4", ("a", "b"), np.ones((3, 2)))

    f = cf_file(build)
    geo = find_geo(f.variable("/", "v"), f.dataset())
    assert (geo.kind, geo.lat.name, geo.lon.name) == ("regular", "a", "b")


def test_lat_lon_identified_by_standard_name(cf_file):
    def build(nc):
        dim(nc, "phi", [10.0, 20.0], standard_name="latitude")
        dim(nc, "lam", [0.0, 5.0, 10.0], standard_name="longitude")
        var(nc, "v", "f4", ("phi", "lam"), np.ones((2, 3)))

    f = cf_file(build)
    geo = find_geo(f.variable("/", "v"), f.dataset())
    assert (geo.lat.name, geo.lon.name) == ("phi", "lam")


# --- 4.3 Vertical coordinate: see test_ui_smoke.py (plot orientation) --------------------------

# --- 4.4 Time coordinate --------------------------------------------------------------------


def _time_file(units, values=(0.0, 1.0), **attrs):
    def build(nc):
        dim(nc, "time", values, units=units, **attrs)
        var(nc, "x", "f4", ("time",), np.arange(len(values)))
    return build


@pytest.mark.parametrize(("calendar", "since", "expected"), [
    ("standard", "2000-02-28", ["2000-02-29", "2000-03-01"]),
    ("gregorian", "2000-02-28", ["2000-02-29", "2000-03-01"]),  # deprecated alias of standard
    ("proleptic_gregorian", "2000-02-28", ["2000-02-29", "2000-03-01"]),
    (None, "2000-02-28", ["2000-02-29", "2000-03-01"]),  # standard is the default
    ("noleap", "2000-02-28", ["2000-03-01", "2000-03-02"]),
    ("365_day", "2000-02-28", ["2000-03-01", "2000-03-02"]),
    ("all_leap", "1900-02-28", ["1900-02-29", "1900-03-01"]),
    ("366_day", "1900-02-28", ["1900-02-29", "1900-03-01"]),
    ("360_day", "1900-02-28", ["1900-02-29", "1900-02-30"]),
    ("julian", "1900-02-28", ["1900-02-29", "1900-03-01"]),
    ("standard", "1900-02-28", ["1900-03-01", "1900-03-02"]),
    ("utc", "2000-02-28", ["2000-02-29", "2000-03-01"]),  # CF 1.12
    ("tai", "2000-02-28", ["2000-02-29", "2000-03-01"]),
])
def test_calendars(cf_file, calendar, since, expected):
    attrs = {"calendar": calendar} if calendar else {}
    f = cf_file(_time_file(f"days since {since}", (1.0, 2.0), **attrs))
    assert f.warnings == []
    assert dates(f.variable("/", "time").values) == expected


def test_utc_calendar_counts_leap_seconds(cf_file):
    """Two seconds after 23:59:59 on 2016-12-31 is midnight: 23:59:60 was a leap second."""
    f = cf_file(_time_file("seconds since 2016-12-31 23:59:59", (0.0, 2.0, 3.0), calendar="utc"))
    assert dates(f.variable("/", "time").values) == ["2016-12-31 23:59:59", "2017-01-01",
                                                      "2017-01-01 00:00:01"]
    f = cf_file(_time_file("seconds since 2016-12-31 23:59:59", (0.0, 2.0), calendar="tai"))
    assert dates(f.variable("/", "time").values) == ["2016-12-31 23:59:59", "2017-01-01 00:00:01"]


def test_one_undecodable_time_leaves_the_others_decoded(cf_file):
    def build(nc):
        dim(nc, "time", [0.0, 1.0], units="days since 2000-01-01")
        var(nc, "issued", "f8", ("time",), [0, 1], units="days since yesterday")
        var(nc, "x", "f4", ("time",), [1, 2])

    f = cf_file(build)
    assert dates(f.variable("/", "time").values) == ["2000-01-01", "2000-01-02"]
    np.testing.assert_array_equal(f.variable("/", "issued").values, [0, 1])
    assert f.warnings == ["Time values could not be decoded and are shown as raw numbers (issued)."]


def test_standard_calendar_switches_from_julian_to_gregorian(cf_file):
    """1582-10-04 (Julian) is followed by 1582-10-15 (Gregorian)."""
    f = cf_file(_time_file("days since 1582-10-04", calendar="standard"))
    assert dates(f.variable("/", "time").values) == ["1582-10-04", "1582-10-15"]
    f = cf_file(_time_file("days since 1582-10-04", calendar="proleptic_gregorian"))
    assert dates(f.variable("/", "time").values) == ["1582-10-04", "1582-10-05"]


@pytest.mark.parametrize(("units", "expected"), [
    ("seconds since 1992-10-8 15:15:42 -6:00", "1992-10-08 21:15:43"),  # time zone, in UTC
    ("hours since 1990-01-01 00:00:00 +0", "1990-01-01 01:00"),
    ("days since 1990-1-1 0:0:0", "1990-01-02"),
    ("seconds since 1970-01-01T00:00:00Z", "1970-01-01 00:00:01"),
    ("minutes since 2000-01-01 12:00", "2000-01-01 12:01"),
])
def test_reference_time_formats(cf_file, units, expected):
    f = cf_file(_time_file(units))
    assert dates(f.variable("/", "time").values)[1] == expected


@pytest.mark.parametrize("calendar", ["standard", "proleptic_gregorian", "noleap", "360_day", "julian"])
@pytest.mark.parametrize(("offset", "expected"), [
    ("-6:00", "2002-05-07 17:24"),
    ("+0530", "2002-05-07 05:54"),
    ("0:00", "2002-05-07 11:24"),  # unsigned, as ARM writes UTC; not the time of day
    ("UTC", "2002-05-07 11:24"),
])
def test_utc_offset_in_every_calendar(cf_file, calendar, offset, expected):
    """4.4: the reference time may end in a UTC offset; times are shown in UTC."""
    f = cf_file(_time_file(f"seconds since 2002-05-07 11:24:00 {offset}", calendar=calendar))
    time = f.variable("/", "time")
    assert dates(time.values)[0] == expected
    assert time.encoding["units"].endswith(offset)  # shown in the info tab as written


@pytest.mark.parametrize("attrs", [
    {"units": "months since 2000-01-01"},  # allowed, but calendar months aren't what CF means
    {"units": "days since 2000-01-01", "calendar": "mars",
     "month_lengths": np.array([30] * 12, "i4"), "leap_year": np.int32(0)},
], ids=["months", "explicit calendar"])
def test_undecodable_times_are_still_a_time_axis(cf_file, attrs):
    f = cf_file(_time_file(**attrs))
    assert any("raw numbers" in w for w in f.warnings)
    np.testing.assert_array_equal(f.variable("/", "time").values, [0, 1])
    assert time_dims(f.variable("/", "x")) == ("time",)


@pytest.mark.parametrize("attrs", [{"axis": "T"}, {"standard_name": "time"}])
def test_time_axis_identified_by_attributes(cf_file, attrs):
    def build(nc):
        dim(nc, "step", [0.0, 1.0], **attrs)
        var(nc, "x", "f4", ("step",), [1, 2])

    f = cf_file(build)
    assert is_time_dim(f.variable("/", "x"), "step")


# --- 5 Coordinate systems and domain -------------------------------------------------------------


def _projected(nc, **data_attrs):
    dim(nc, "y", [0.0, 1000.0, 2000.0], units="m", standard_name="projection_y_coordinate", axis="Y")
    dim(nc, "x", [0.0, 1000.0], units="m", standard_name="projection_x_coordinate", axis="X")
    var(nc, "crs", "i4", grid_mapping_name="lambert_conformal_conic", standard_parallel=45.0,
        longitude_of_central_meridian=10.0, latitude_of_projection_origin=45.0)
    var(nc, "tas", "f4", ("y", "x"), np.ones((3, 2)), grid_mapping="crs", **data_attrs)


def test_projection_coordinates_with_auxiliary_lat_lon(cf_file):
    """5.2, 5.6: x/y are projection coordinates; lat/lon come from the coordinates attribute."""
    def build(nc):
        _projected(nc, coordinates="lat lon")
        j, i = np.mgrid[0:3, 0:2]
        var(nc, "lat", "f8", ("y", "x"), 45 + 0.01 * j, units="degrees_north", standard_name="latitude")
        var(nc, "lon", "f8", ("y", "x"), 10 + 0.01 * i, units="degrees_east", standard_name="longitude")

    f = cf_file(build)
    da = f.variable("/", "tas")
    assert {"lat", "lon"} <= set(da.coords)
    geo = find_geo(da, f.dataset())
    assert (geo.kind, geo.lat.name, geo.lon.name, geo.dims) == ("curvilinear", "lat", "lon", ("y", "x"))


def test_projection_coordinates_alone_are_not_lat_lon(cf_file):
    f = cf_file(_projected)
    assert find_geo(f.variable("/", "tas"), f.dataset()) is None


def _rotated_pole(nc, lat_name="rlat", lon_name="rlon", aux=False):
    dim(nc, lat_name, [0.0, 1.0, 2.0], standard_name="grid_latitude", units="degrees", axis="Y")
    dim(nc, lon_name, [0.0, 1.0], standard_name="grid_longitude", units="degrees", axis="X")
    var(nc, "rotated_pole", "c", grid_mapping_name="rotated_latitude_longitude",
        grid_north_pole_latitude=39.25, grid_north_pole_longitude=-162.0)
    attrs = {"coordinates": "lat lon"} if aux else {}
    var(nc, "tas", "f4", (lat_name, lon_name), np.ones((3, 2)), grid_mapping="rotated_pole", **attrs)
    if aux:
        j, i = np.mgrid[0:3, 0:2]
        var(nc, "lat", "f8", (lat_name, lon_name), 50 + j + 0.1 * i, units="degrees_north",
            standard_name="latitude")
        var(nc, "lon", "f8", (lat_name, lon_name), 10 + i + 0.1 * j, units="degrees_east",
            standard_name="longitude")


def test_rotated_pole_uses_true_lat_lon(cf_file):
    f = cf_file(lambda nc: _rotated_pole(nc, aux=True))
    geo = find_geo(f.variable("/", "tas"), f.dataset())
    assert (geo.kind, geo.lat.name, geo.lon.name, geo.dims) == ("curvilinear", "lat", "lon",
                                                               ("rlat", "rlon"))


@pytest.mark.parametrize("names", [("rlat", "rlon"), ("lat", "lon")])
def test_rotated_coordinates_are_not_lat_lon(cf_file, names):
    """grid_latitude/grid_longitude aren't geographic, whatever the variables are called."""
    f = cf_file(lambda nc: _rotated_pole(nc, *names))
    assert find_geo(f.variable("/", "tas"), f.dataset()) is None


def test_scalar_coordinate_variables(cf_file):
    """5.7: scalar coordinates apply to every value, e.g. in an exported time series."""
    def build(nc):
        dim(nc, "time", [0.0, 1.0, 2.0], units="days since 2000-01-01")
        var(nc, "height", "f8", (), 2.0, units="m", standard_name="height", positive="up")
        var(nc, "lat", "f8", (), 46.0, units="degrees_north")
        var(nc, "lon", "f8", (), 14.5, units="degrees_east")
        var(nc, "tas", "f4", ("time",), [1, 2, 3], coordinates="height lat lon")

    f = cf_file(build)
    da = f.variable("/", "tas")
    assert {"height", "lat", "lon"} <= set(da.coords)
    assert time_dims(da) == ("time",)
    frame = series_frame(point_series(da, {}, {}))
    assert list(frame.columns) == ["height", "lat", "lon", "tas"]
    assert (frame["height"] == 2).all() and (frame["lat"] == 46).all()


# --- 6.1 Labels -------------------------------------------------------------------------------


def test_string_labels(cf_file):
    def build(nc):
        nc.createDimension("region", 3)
        nc.createDimension("strlen", 5)
        var(nc, "region_name", "S1", ("region", "strlen"),
            np.array([list(s.ljust(5, "\0")) for s in ("north", "south", "east")], "S1"),
            standard_name="region")
        var(nc, "flux", "f4", ("region",), [1, 2, 3], coordinates="region_name")

    f = cf_file(build)
    da = f.variable("/", "flux")
    assert [format_value(v) for v in da.coords["region_name"].values] == ["north", "south", "east"]
    assert find_geo(da, f.dataset()) is None


def test_label_coordinate_variable_names_positions(cf_file):
    def build(nc):
        nc.createDimension("region", 2)
        names = nc.createVariable("region", str, ("region",))
        names[:] = np.array(["Alps", "Pannonia"], dtype=object)
        var(nc, "flux", "f4", ("region",), [1, 2])

    da = cf_file(build).variable("/", "flux")
    assert [index_label(da, "region", i) for i in range(2)] == ["Alps", "Pannonia"]


# --- 7.1 Cell boundaries ----------------------------------------------------------------------


def _bounded(nc):
    nc.createDimension("nv", 2)
    dim(nc, "lat", [0.5, 2.0, 3.5], units="degrees_north", bounds="lat_bnds")
    var(nc, "lat_bnds", "f8", ("lat", "nv"), [[0, 1], [1, 3], [3, 4]])
    dim(nc, "lon", [10.0, 20.0], units="degrees_east", bounds="lon_bnds")
    var(nc, "lon_bnds", "f8", ("lon", "nv"), [[5, 15], [15, 25]])
    dim(nc, "time", [0.5, 1.5], units="days since 2000-02-28", calendar="noleap", bounds="time_bnds")
    var(nc, "time_bnds", "f8", ("time", "nv"), [[0, 1], [1, 2]])  # units come from time
    var(nc, "v", "f4", ("time", "lat", "lon"), np.ones((2, 3, 2)))


def test_cell_bounds_define_map_cells(cf_file):
    f = cf_file(_bounded)
    da = f.variable("/", "v")
    geo = find_geo(da, f.dataset())
    grid = geo_grid(da, geo, {})
    assert grid.from_bounds
    np.testing.assert_array_equal(grid.lat_edges, [0, 1, 3, 4])
    np.testing.assert_array_equal(grid.lon_edges, [5, 15, 25])
    # The cell under a position is the one whose bounds hold it, not the nearest centre.
    assert GridLocator(grid).nearest(1.1, 10) == (1, 0)
    lines = grid_lines(da, geo, grid, {}, (-10, 0, 10, 30), max_lines=100)
    assert sorted({line[0][0] for line in lines if line[0][0] == line[1][0]}) == [0, 1, 3, 4]


def test_bounds_of_curvilinear_cells(cf_file):
    """2D bounds have 4 vertices per cell; CF doesn't fix which one comes first."""
    ny, nx = 3, 4
    j, i = np.mgrid[0:ny + 1, 0:nx + 1]
    corner_lat, corner_lon = 40 + 2.0 * j + 0.3 * i ** 2, 10 + 3.0 * i + 0.2 * j ** 2  # uneven cells

    def vertices(c):
        v = np.stack([c[:-1, :-1], c[:-1, 1:], c[1:, 1:], c[1:, :-1]], axis=-1)
        return np.roll(v, 1, axis=-1)  # starting at another corner

    def centres(c):
        return (c[:-1, :-1] + c[1:, :-1] + c[:-1, 1:] + c[1:, 1:]) / 4

    def build(nc):
        nc.createDimension("y", ny)
        nc.createDimension("x", nx)
        nc.createDimension("nv", 4)
        var(nc, "lat", "f8", ("y", "x"), centres(corner_lat), units="degrees_north", bounds="lat_b")
        var(nc, "lon", "f8", ("y", "x"), centres(corner_lon), units="degrees_east", bounds="lon_b")
        var(nc, "lat_b", "f8", ("y", "x", "nv"), vertices(corner_lat))
        var(nc, "lon_b", "f8", ("y", "x", "nv"), vertices(corner_lon))
        var(nc, "v", "f4", ("y", "x"), np.arange(ny * nx).reshape(ny, nx), coordinates="lat lon")

    f = cf_file(build)
    da = f.variable("/", "v")
    grid = geo_grid(da, find_geo(da, f.dataset()), {})
    assert grid.from_bounds
    np.testing.assert_allclose(grid.corners[0], corner_lat)
    np.testing.assert_allclose(grid.corners[1], corner_lon)
    cells = cell_polygons(grid, "viridis", 0, 11)
    np.testing.assert_allclose(cells.lat, corner_lat.ravel(), atol=1e-6)


def test_time_bounds_use_units_and_calendar_of_time(cf_file):
    bnds = cf_file(_bounded).variable("/", "time_bnds").values
    assert dates(bnds.ravel()) == ["2000-02-28", "2000-03-01", "2000-03-01", "2000-03-02"]


# --- 8.1 Packed data ----------------------------------------------------------------------------


@pytest.mark.parametrize(("scale_type", "expected"), [("f4", np.float32), ("f8", np.float64)])
def test_unpacked_type_is_that_of_scale_factor(cf_file, scale_type, expected):
    def build(nc):
        nc.createDimension("n", 3)
        var(nc, "v", "i2", ("n",), [1, 2, -32767], fill_value=np.int16(-32767),
            scale_factor=np.array(0.5, scale_type), add_offset=np.array(100, scale_type))

    values = cf_file(build).variable("/", "v").values
    assert values.dtype == expected
    np.testing.assert_array_equal(values, [100.5, 101, np.nan])  # _FillValue is in packed units


def test_unsigned_bytes(cf_file):
    def build(nc):
        nc.createDimension("n", 3)
        var(nc, "v", "i1", ("n",), np.array([1, -1, -56], "i1"), _Unsigned="true")

    np.testing.assert_array_equal(cf_file(build).variable("/", "v").values, [1, 255, 200])


def test_valid_range_of_packed_data_is_in_packed_units(cf_file):
    def build(nc):
        nc.createDimension("n", 3)
        var(nc, "v", "i2", ("n",), [10, 200, -3], scale_factor=np.float32(0.1),
            valid_range=np.array([0, 100], "i2"))

    values = cf_file(build).variable("/", "v").values
    assert values[0] == pytest.approx(1) and np.isnan(values[1:]).all()


# --- 9 Discrete sampling geometries --------------------------------------------------------------


def test_single_trajectory_is_mapped_as_points(cf_file):
    def build(nc):
        nc.featureType = "trajectory"
        nc.createDimension("obs", 4)
        var(nc, "time", "f8", ("obs",), [0, 1, 2, 3], units="hours since 2000-01-01",
            standard_name="time")
        var(nc, "lat", "f8", ("obs",), [40, 41, 42, 43], units="degrees_north")
        var(nc, "lon", "f8", ("obs",), [10, 11, 12, 13], units="degrees_east")
        var(nc, "temp", "f4", ("obs",), [1, 2, 3, 4], coordinates="time lat lon")

    f = cf_file(build)
    geo = find_geo(f.variable("/", "temp"), f.dataset())
    assert (geo.kind, geo.dims) == ("points", ("obs",))


def test_trajectories_are_mapped_one_at_a_time(cf_file):
    """9.3.2: incomplete multidimensional array of trajectories, padded with missing values."""
    def build(nc):
        nc.featureType = "trajectory"
        nc.createDimension("trajectory", 2)
        nc.createDimension("obs", 4)
        var(nc, "trajectory", "i4", ("trajectory",), [1, 2], cf_role="trajectory_id")
        var(nc, "time", "f8", ("trajectory", "obs"), [[0, 1, 2, 3], [0, 1, -1, -1]], fill_value=-1.0,
            units="hours since 2000-01-01", standard_name="time")
        var(nc, "lat", "f8", ("trajectory", "obs"), [[40, 41, 42, 43], [50, 51, -999, -999]],
            fill_value=-999.0, units="degrees_north")
        var(nc, "lon", "f8", ("trajectory", "obs"), [[10, 11, 12, 13], [20, 21, -999, -999]],
            fill_value=-999.0, units="degrees_east")
        var(nc, "temp", "f4", ("trajectory", "obs"), [[1, 2, 3, 4], [5, 6, -9, -9]],
            fill_value=np.float32(-9), coordinates="time lat lon")

    f = cf_file(build)
    da = f.variable("/", "temp")
    geo = find_geo(da, f.dataset())
    assert (geo.kind, geo.dims) == ("points", ("obs",))  # not a grid
    grid = geo_grid(da, geo, {"trajectory": 1})
    np.testing.assert_array_equal(grid.lat, [50, 51, np.nan, np.nan])
    np.testing.assert_array_equal(grid.values, [5, 6, np.nan, np.nan])


def test_time_series_of_stations(cf_file):
    """9.3.1: orthogonal multidimensional timeSeries; each station's series can be extracted."""
    def build(nc):
        nc.featureType = "timeSeries"
        nc.createDimension("station", 3)
        dim(nc, "time", [0.0, 24.0], units="hours since 2024-05-01", standard_name="time")
        var(nc, "lat", "f8", ("station",), [46, 45.5, 46.5], units="degrees_north")
        var(nc, "lon", "f8", ("station",), [14.5, 13.7, 15.6], units="degrees_east")
        var(nc, "precip", "f4", ("station", "time"), [[1, 2], [3, 4], [5, 6]], coordinates="lat lon")

    f = cf_file(build)
    da = f.variable("/", "precip")
    geo = find_geo(da, f.dataset())
    assert (geo.kind, geo.dims) == ("points", ("station",))
    frame = series_frame(point_series(da, {"station": 1}, {}))
    assert list(frame.index) == list(pd.to_datetime(["2024-05-01", "2024-05-02"]))
    assert list(frame["precip"]) == [3, 4]
