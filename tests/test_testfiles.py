"""CF compliance on real-world files: the ones in tests/testfiles.

The files are large and not part of the repository, so these tests are
skipped when the directory isn't there. They only read metadata and small
slices, never whole variables.

Two kinds of checks:
- every file: opens lazily without warnings, and every numeric variable
  decodes to the same values as netCDF4-python's own masking and unpacking
  (an independent implementation of the NUG/CF missing-data rules), on a
  small corner of the variable;
- per file: the CF features that file uses, with the answers worked out
  from its metadata.
"""

from pathlib import Path

import netCDF4
import numpy as np
import pytest

from netseedf.core.coords import find_geo, geo_grid
from netseedf.core.dataset import describe, group_header, open_file
from netseedf.core.formatting import format_value
from netseedf.core.render import render_overlay
from netseedf.core.slicing import is_downward, time_dims

TESTFILES = Path(__file__).parent / "testfiles"
FILES = sorted(p.name for p in TESTFILES.glob("*.nc")) if TESTFILES.is_dir() else []

pytestmark = pytest.mark.skipif(not FILES, reason="tests/testfiles isn't there")

# Where netCDF4-python doesn't follow CF, so NetSeeDF shouldn't match it.
ORACLE_WRONG = {
    # valid_range is float32, the type of scale_factor, so it's in unpacked units (CF 8.1);
    # netCDF4-python compares it with the packed values and masks everything.
    ("rhum.2003.nc", "/", "rhum"),
}


@pytest.fixture(scope="module")
def testfile():
    """Open a file from tests/testfiles once per module."""
    opened = {}

    def _open(name):
        if not (TESTFILES / name).exists():
            pytest.skip(f"{name} isn't in tests/testfiles")
        if name not in opened:
            opened[name] = open_file(TESTFILES / name)
        return opened[name]

    yield _open
    for f in opened.values():
        f.close()


def dates(values):
    return [format_value(v) for v in np.ravel(values)]


def corner(shape, n=200):
    """First index of all but the last two dims, up to n along those."""
    return tuple(slice(0, 1) if i < len(shape) - 2 else slice(0, n) for i in range(len(shape)))


def netcdf4_values(var, key):
    """What netCDF4-python makes of the values: masked, unpacked, NaN where missing."""
    var.set_auto_maskandscale(True)
    data = var[key]
    mask = np.ma.getmaskarray(data)
    data = np.array(np.ma.getdata(data), dtype=float)
    data[mask] = np.nan
    return data


def _numeric_variables(nc):
    def walk(node, path):
        yield path, node
        for name, child in node.groups.items():
            yield from walk(child, f"{path.rstrip('/')}/{name}")

    for group, node in walk(nc, "/"):
        for name, var in node.variables.items():
            if var.dtype is not str and var.dtype.kind in "fiu" \
                    and " since " not in str(getattr(var, "units", "")):  # times are compared elsewhere
                yield group, name, var


# --- Every file -----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", FILES)
def test_opens_lazily_without_warnings(testfile, name):
    f = testfile(name)
    assert f.warnings == []
    for ds in f.groups.values():
        for var in ds.data_vars.values():
            if var.size > 1_000_000:
                assert not var.variable._in_memory, var.name


@pytest.mark.parametrize("name", FILES)
def test_values_match_netcdf4(testfile, name):
    f = testfile(name)
    compared = 0
    with netCDF4.Dataset(TESTFILES / name) as nc:
        for group, var_name, var in _numeric_variables(nc):
            ours = f.variable(group, var_name).variable
            if ours.dtype.kind not in "fiu" or (name, group, var_name) in ORACLE_WRONG:
                continue  # e.g. time bounds, decoded to dates
            key = corner(var.shape)
            np.testing.assert_allclose(np.asarray(ours[key].values, dtype=float), netcdf4_values(var, key),
                                       rtol=1e-6, err_msg=f"{group} {var_name}")
            compared += 1
    assert compared > 0


@pytest.mark.parametrize("name", FILES)
def test_every_variable_described_and_located(testfile, name):
    f = testfile(name)
    group_header(TESTFILES / name)
    mapped = set()
    for ds in f.groups.values():
        for var_name in ds.data_vars:
            da = ds[var_name]
            describe(da)
            geo = find_geo(da, ds)
            if geo is None or da.dtype.kind not in "fiu":
                continue
            assert all(hasattr(c, "name") for c in (geo.lat, geo.lon)), "coordinates must be DataArrays"
            key = (geo.kind, str(geo.lat.name), str(geo.lon.name), geo.dims)
            if key in mapped:
                continue
            mapped.add(key)
            grid = geo_grid(da, geo, {}, max_size=40)
            lat = grid.lat[np.isfinite(grid.lat)]
            assert lat.size and np.all((lat >= -90) & (lat <= 90))


# --- CF 4.4 Time coordinates ----------------------------------------------------------------------


@pytest.mark.parametrize(("name", "var", "first", "last"), [
    ("ECMWF_ERA-40_subset.nc", "time", "2002-07-01 12:00", "2002-07-31 18:00"),
    ("ECMWF_utci_20260601_v1.1_con.nc", "time", "2026-06-01", "2026-06-01 23:00"),  # proleptic_gregorian
    ("tg_ens_mean_0.1deg_reg_2011-2025_v33.0e.nc", "time", "2011-01-01", "2025-12-31"),
    # "hours since 1-1-1": with the standard (mixed Julian/Gregorian) calendar, as CF says;
    # a proleptic Gregorian reading would be two days off.
    ("rhum.2003.nc", "time", "2003-01-01", "2003-12-31"),
    ("cami_0000-09-01_64x128_L26_c030918.nc", "time", "0000-09-01", "0000-09-01"),  # noleap, year 0
    ("sresa1b_ncar_ccsm3-example.nc", "time", "2000-05-16 12:00", "2000-05-16 12:00"),  # since 0000-1-1
    ("tos_O1_2001-2002.nc", "time", "2001-01-16", "2002-12-16"),  # 360_day
    ("GLASS.nc", "base_time", "2000-06-30 06:49:09", "2000-06-30 06:49:09"),  # "... 00:00:00 UTC"
    ("19981111_0045.nc", "validTime", "1998-11-11 00:45", "1998-11-11 00:45"),  # "... 00:00:00.00 0:00"
    ("OMI-Aura_L2-example.nc", "DATETIME", "2007-11-20 14:12:23", "2007-11-20 14:12:33"),
])
def test_times(testfile, name, var, first, last):
    values = dates(testfile(name).variable("/", var).values)
    assert (values[0], values[-1]) == (first, last)


def test_tos_360_day_months_and_bounds(testfile):
    ds = testfile("tos_O1_2001-2002.nc").dataset()
    assert len(ds.time) == 24  # two years of 12 30-day months
    assert dates(ds.time_bnds.values[0]) == ["2001-01-01", "2001-02-01"]
    assert dates(ds.time_bnds.values[1]) == ["2001-02-01", "2001-03-01"]  # no February 29/30 trouble
    assert dates(ds.time.values[1]) == ["2001-02-16"]


def test_unsigned_utc_offset(testfile):
    """ARM writes UTC as "... 11:24:00 0:00"; the offset mustn't be taken for the time of day."""
    ds = testfile("sgpsondewnpnC1.nc").dataset()
    assert ds.time_offset.encoding["units"] == "seconds since 2002-05-07 11:24:00 0:00"
    assert dates(ds.time_offset.values[:2]) == ["2002-05-07 11:24", "2002-05-07 11:24:02"]
    assert ds.time_offset.values[0] == ds.base_time.values


@pytest.mark.parametrize(("name", "dim"), [
    ("tg_ens_mean_0.1deg_reg_2011-2025_v33.0e.nc", "time"),
    ("rhum.2003.nc", "time"),
    ("GOTEX.C130-example.nc", "Time"),
])
def test_time_axes(testfile, name, dim):
    ds = testfile(name).dataset()
    da = next(ds[v] for v in ds.data_vars if dim in ds[v].dims)
    assert dim in time_dims(da)


# --- CF 2.5.1 Missing data, 8.1 packed data -----------------------------------------------------------


def test_packed_era40(testfile):
    tcw = testfile("ECMWF_ERA-40_subset.nc").variable("/", "tcw")
    assert tcw.dtype == np.float64  # the type of scale_factor
    assert tcw.encoding["dtype"] == np.int16
    values = tcw.isel(time=0).values
    assert np.isfinite(values).all() and 0 < values.min() < values.max() < 100  # kg m-2 of water


def test_packed_eobs_is_masked_at_sea(testfile):
    tg = testfile("tg_ens_mean_0.1deg_reg_2011-2025_v33.0e.nc").variable("/", "tg")
    assert tg.dtype == np.float32
    corner_values = tg.isel(time=0, latitude=slice(0, 20), longitude=slice(0, 20)).values  # Atlantic
    assert np.isnan(corner_values).all()
    alps = float(tg.isel(time=180).sel(latitude=46.5, longitude=8.0, method="nearest"))
    assert -10 < alps < 30  # a July day in the Alps, degrees Celsius


def test_valid_range_in_unpacked_units(testfile):
    """rhum: packed int16 with valid_range [-25, 125] given in percent (float, like scale_factor)."""
    rhum = testfile("rhum.2003.nc").variable("/", "rhum")
    values = rhum.isel(time=0, level=0).values
    assert np.isfinite(values).mean() > 0.99
    assert np.nanmin(values) >= -25 and np.nanmax(values) <= 125
    assert 50 < np.nanmean(values) < 90


def test_valid_range_in_packed_units(testfile):
    """ROSE: int16 valid_range [-32766, 32767] in packed units; missing_value -32767 is outside it."""
    da = testfile("smith_sandwell_topo_v8_2.nc").variable("/", "ROSE")
    assert da.dtype.kind == "f"
    grid = geo_grid(da, find_geo(da, testfile("smith_sandwell_topo_v8_2.nc").dataset()), {}, max_size=100)
    assert np.nanmin(grid.values) < -5000 and np.nanmax(grid.values) > 3000  # ocean floor, mountains


def test_fill_values_utci(testfile):
    utci = testfile("ECMWF_utci_20260601_v1.1_con.nc").variable("/", "utci")
    values = utci.isel(time=12, lat=slice(0, 601, 10), lon=slice(0, 1440, 10)).values
    assert np.isnan(values).any() and np.isfinite(values).any()  # sea is missing
    assert np.nanmin(values) > 150 and np.nanmax(values) < 350  # kelvin, not -9e33


def test_positions_off_the_globe_are_missing(testfile):
    """IMAGE0002 marks undefined pixels with lat/lon of about 2e9 instead of a fill value."""
    f = testfile("IMAGE0002.nc")
    assert f.variable("/", "latitude").max() > 1e9
    da = f.variable("/", "data")
    grid = geo_grid(da, find_geo(da, f.dataset()), {})
    lat = grid.lat[np.isfinite(grid.lat)]
    assert 0 < lat.size < grid.lat.size and np.abs(lat).max() <= 90
    assert np.isnan(grid.lon[np.isnan(grid.lat)]).all()
    south, west, north, east = grid.bounds
    assert -90 <= south < north <= 90 and -180 <= west < east <= 180


@pytest.mark.filterwarnings("error:The input coordinates to pcolormesh:UserWarning")
def test_satellite_image_with_missing_positions_is_drawn(testfile):
    """Beyond the Earth's limb IMAGE0002 has no positions; the web map image covers the rest."""
    f = testfile("IMAGE0002.nc")
    da = f.variable("/", "data")
    grid = geo_grid(da, find_geo(da, f.dataset()), {}, max_size=4096)
    overlay = render_overlay(grid, "viridis", float(np.nanmin(grid.values)), float(np.nanmax(grid.values)))
    south, west, north, east = grid.bounds
    assert south - 1 < overlay.south < south and north < overlay.north < north + 1
    assert west - 3 < overlay.west < west and east < overlay.east < east + 3


def test_ease2_grid_uses_its_projection(testfile):
    """OSI SAF ice concentration: EASE2 north (Lambert azimuthal) with x/y in km (CF 5.6)."""
    f = testfile("ice_conc_nh_ease2-250_cdr-v3p0-amsr_202009011200.nc")
    da = f.variable("/", "ice_conc")
    geo = find_geo(da, f.dataset())
    assert geo.projection is not None
    assert geo.projection.crs.to_cf()["grid_mapping_name"] == "lambert_azimuthal_equal_area"
    grid = geo_grid(da, geo, {})
    x_edges, y_edges = grid.xy_edges
    np.testing.assert_allclose(x_edges[[0, -1]], [-5_400_000, 5_400_000])  # 432 cells of 25 km
    np.testing.assert_allclose(y_edges[[0, -1]], [5_400_000, -5_400_000])
    lat, lon = geo.projection.centre()
    assert lat == pytest.approx(90) and lon == pytest.approx(0, abs=1e-6)


def test_infinite_valid_range_masks_nothing(testfile):
    o3 = testfile("OMI-Aura_L2-example.nc").variable("/", "ColumnAmountO3")
    assert np.isfinite(o3.values).all()


# --- CF 4.1-4.3 Coordinate types ------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "var", "kind", "lat", "lon"), [
    ("ECMWF_ERA-40_subset.nc", "tcw", "regular", "latitude", "longitude"),
    ("smith_sandwell_topo_v8_2.nc", "ROSE", "regular", "latitude", "longitude"),  # degrees_N / degrees_E
    ("rhum.2003.nc", "rhum", "regular", "lat", "lon"),
    ("OMI-Aura_L2-example.nc", "ColumnAmountO3", "points", "LATITUDE", "LONGITUDE"),  # degree_north
    ("madis-profiler.nc", "windSpeedSfc", "points", "staLat", "staLon"),  # degree_N
    ("GLASS.nc", "tdry", "points", "lat", "lon"),  # not coordinates: found in the dataset
    ("wrfout_v2_Lambert.nc", "T2", "curvilinear", "XLAT", "XLONG"),
    ("IMAGE0002.nc", "data", "curvilinear", "latitude", "longitude"),
])
def test_lat_lon(testfile, name, var, kind, lat, lon):
    f = testfile(name)
    geo = find_geo(f.variable("/", var), f.dataset())
    assert (geo.kind, geo.lat.name, geo.lon.name) == (kind, lat, lon)


@pytest.mark.parametrize(("name", "dim", "down"), [
    ("rhum.2003.nc", "level", True),  # millibar, positive: down
    ("sresa1b_ncar_ccsm3-example.nc", "plev", True),  # Pa
    ("OMI-Aura_L2-example.nc", "PRESSURE", True),  # hPa, no positive attribute
    ("cami_0000-09-01_64x128_L26_c030918.nc", "lev", True),  # hybrid level, positive: down
])
def test_vertical_direction(testfile, name, dim, down):
    ds = testfile(name).dataset()
    da = next(ds[v] for v in ds.data_vars if dim in ds[v].dims)
    assert is_downward(da, dim) == down


# --- CF 5.7 scalar coordinates, 7.1 cell bounds -----------------------------------------------------------


def test_gaussian_grid_bounds(testfile):
    """sresa1b: the first latitude cell is [-90, -88.23] around -88.93, not halfway between centres."""
    f = testfile("sresa1b_ncar_ccsm3-example.nc")
    tas = f.variable("/", "tas")
    geo = find_geo(tas, f.dataset())
    grid = geo_grid(tas, geo, {})
    assert grid.from_bounds
    bnds = np.sort(f.dataset().lat_bnds.values, axis=1)
    np.testing.assert_allclose(grid.lat_edges, np.append(bnds[:, 0], bnds[-1, 1]))
    assert grid.lat_edges[0] == -90 and grid.lat_edges[-1] == 90


def test_missing_scalar_coordinate_is_harmless(testfile):
    """tas names a `height` coordinate the file doesn't have."""
    tas = testfile("sresa1b_ncar_ccsm3-example.nc").variable("/", "tas")
    assert "height" not in tas.coords
    assert tas.attrs["cell_methods"] == "time: mean (interval: 1 month)"


# --- CF 2.7 Groups ----------------------------------------------------------------------------------------


def test_groups_use_dimensions_of_the_root(testfile):
    f = testfile("test_hgroups.nc")
    assert len(f.groups) == 8
    group = "/mozaic_flight_2012030403540535_ascent"
    co = f.variable(group, "CO")
    assert co.dims == ("recNum",) and co.shape == (74,)
    assert f.variable(group, "lat").ndim == 0  # one position per flight


# --- netCDF-4 compression is invisible ------------------------------------------------------------------


def test_deflated_file_reads_like_the_original(testfile):
    plain = testfile("test_echam_spectral.nc").dataset()
    deflated = testfile("test_echam_spectral-deflated.nc").dataset()
    assert set(plain.data_vars) == set(deflated.data_vars)
    for name in ("tpot", "aps", "st"):
        np.testing.assert_array_equal(plain[name].isel(time=-1).values, deflated[name].isel(time=-1).values)


# --- Beyond CF: netCDF features the viewer must cope with ------------------------------------------


def test_repeated_dimension_gets_its_own_name(testfile):
    """OMI's averaging kernel is a PRESSURE x PRESSURE matrix per time."""
    avk = testfile("OMI-Aura_L2-example.nc").variable("/", "O3.COLUMN.PARTIAL_AVK")
    assert avk.dims == ("DATETIME", "PRESSURE", "PRESSURE_2")
    np.testing.assert_array_equal(avk.PRESSURE_2.values, avk.PRESSURE.values)
    with netCDF4.Dataset(TESTFILES / "OMI-Aura_L2-example.nc") as nc:
        expected = netcdf4_values(nc["O3.COLUMN.PARTIAL_AVK"], 1)
    np.testing.assert_allclose(avk.isel(DATETIME=1).values, expected, rtol=1e-6)
