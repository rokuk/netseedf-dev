import numpy as np
import pytest
import xarray as xr

from netseedf.core.coords import find_geo, geo_grid
from netseedf.core.gridlines import cell_corners, grid_lines


@pytest.fixture
def lines_for(samples):
    handles = []

    def _lines(name, var, box, max_lines=1000, max_size=None):
        ds = xr.open_dataset(samples[name])
        handles.append(ds)
        da, geo = ds[var], find_geo(ds[var], ds)
        return grid_lines(da, geo, geo_grid(da, geo, {}, max_size), {}, box, max_lines)

    yield _lines
    for ds in handles:
        ds.close()


def _split(lines):
    """Vertical lines' longitudes and horizontal lines' latitudes."""
    lons = sorted({a[1] for a, b in lines if a[1] == b[1]})
    lats = sorted({a[0] for a, b in lines if a[0] == b[0]})
    return lons, lats


def test_regular_lines_are_the_cell_edges(lines_for):
    lines = lines_for("four_d.nc", "salinity", (-90, -180, 90, 180))
    lons, lats = _split(lines)
    assert len(lons) == 41 and len(lats) == 31  # one more edge than cells
    step_lon, step_lat = 50 / 39, 30 / 29
    assert lons[0] == pytest.approx(-20 - step_lon / 2) and lons[-1] == pytest.approx(30 + step_lon / 2)
    assert lats[0] == pytest.approx(35 - step_lat / 2) and lats[-1] == pytest.approx(65 + step_lat / 2)
    for (lat0, _), (lat1, _) in (line for line in lines if line[0][1] == line[1][1]):
        assert (lat0, lat1) == pytest.approx((lats[0], lats[-1]))  # vertical lines span the grid


def test_regular_lines_only_in_the_box(lines_for):
    # Centres at odd latitudes and even longitudes, 2° apart.
    lines = lines_for("regular_global.nc", "sst", (40, 0, 50, 10))
    lons, lats = _split(lines)
    assert lons == [1, 3, 5, 7, 9]
    assert lats == [40, 42, 44, 46, 48, 50]
    assert all(0 <= lon <= 10 and 40 <= lat <= 50 for line in lines for lat, lon in line)


def test_regular_lines_across_the_antimeridian(lines_for):
    lons, _ = _split(lines_for("regular_global.nc", "sst", (0, 170, 10, 190)))
    assert lons == [171, 173, 175, 177, 179, 181, 183, 185, 187, 189]


def test_too_many_lines(lines_for):
    assert lines_for("regular_global.nc", "sst", (-90, -180, 90, 180), max_lines=100) is None
    assert lines_for("regular_global.nc", "sst", (40, 0, 50, 10), max_lines=100) is not None


def test_curvilinear_lines_follow_the_grid(lines_for):
    lines = lines_for("curvilinear.nc", "temp", (-90, -180, 90, 180), max_size=10)
    assert len(lines) == 41 + 51  # a line along every row and column of corners, at full resolution
    first_row = np.array(lines[0])
    assert first_row.shape == (51, 2)
    # lat/lon are linear in the indices, so the corner of cell (0, 0) is at index (-0.5, -0.5).
    angle = np.deg2rad(20)
    lat = 40 + 0.5 * (-0.5 * np.cos(angle) - 0.5 * np.sin(angle) * 0.3)
    lon = -5 + 0.6 * (-0.5 * np.cos(angle) + 0.5 * np.sin(angle) * 0.3)
    assert first_row[0] == pytest.approx([lat, lon])


def test_curvilinear_too_many_lines(lines_for):
    assert lines_for("curvilinear.nc", "temp", (-90, -180, 90, 180), max_lines=50) is None


def test_points_have_no_lines(lines_for):
    assert lines_for("points.nc", "precip", (-90, -180, 90, 180)) == []


def test_cell_corners():
    y, x = np.meshgrid(np.arange(3.0), np.arange(4.0), indexing="ij")
    corners = cell_corners(10 * y + x)
    assert corners.shape == (4, 5)
    cy, cx = np.meshgrid(np.arange(-0.5, 3), np.arange(-0.5, 4), indexing="ij")
    np.testing.assert_allclose(corners, 10 * cy + cx)


def test_projected_lines_are_exact_and_cut_at_the_antimeridian(lines_for):
    lines = lines_for("polar_projected.nc", "ice", (0, -180, 90, 180))
    points = np.concatenate([np.array(line) for line in lines])
    assert np.abs(points[:, 1]).max() <= 180
    for line in lines:  # nothing drawn across the map
        assert np.abs(np.diff(np.array(line)[:, 1])).max(initial=0) <= 90
    # A corner of the grid (x, y = -6000 km, 6000 km), where the projection puts it.
    import pyproj
    crs = pyproj.CRS.from_cf({"grid_mapping_name": "lambert_azimuthal_equal_area",
                              "longitude_of_projection_origin": 0.0, "latitude_of_projection_origin": 90.0,
                              "semi_major_axis": 6378137.0, "inverse_flattening": 298.257223563})
    lon, lat = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(-6e6, 6e6)
    assert np.min(np.hypot(points[:, 0] - lat, points[:, 1] - lon)) < 1e-6
