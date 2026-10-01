"""Open every sample file, select every variable and look at every tab.

The web map needs a GPU-backed window, so it's left out here.
"""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from PySide6 import QtWebEngineWidgets  # noqa: E402, F401 (must precede the QApplication)
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtGui import QKeySequence  # noqa: E402
from PySide6.QtWidgets import QAbstractButton, QFileDialog, QLabel  # noqa: E402

from netseedf.context import SourceContext  # noqa: E402
from netseedf.core.coords import geo_grid  # noqa: E402
from netseedf.ui import web_map_view  # noqa: E402
from netseedf.ui.cartopy_map_view import configure_offline_data  # noqa: E402
from netseedf.ui.main_window import MainWindow, user_agent  # noqa: E402
from netseedf.ui.table_view import to_frame  # noqa: E402

RESOURCES = Path(__file__).resolve().parents[1] / "src" / "main" / "resources" / "base"
SETTINGS = {"app_name": "netseedf", "version": "0.0.0", "homepage": "https://example.org"}


def resource(*rel):
    return str(RESOURCES.joinpath(*rel))


@pytest.fixture
def window(qtbot, tmp_path):
    configure_offline_data(RESOURCES / "cartopy", tmp_path / "cartopy")
    w = MainWindow(resource, SETTINGS, QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(w)
    w.resize(1200, 800)
    w.show()
    return w


def _errors(w):
    """Views that failed (as opposed to explaining why they don't apply)."""
    return {v.title: v._message.text() for v in w.views
            if v._stack.currentWidget() is v._message and v._message.text().startswith("Couldn't")}


@pytest.mark.parametrize("name", ["regular_global.nc", "curvilinear.nc", "timeseries.nc", "four_d.nc",
                                  "groups.nc", "netcdf3.nc", "packed.nc", "noleap.nc", "bad_time.nc",
                                  "points.nc", "pacific.nc", "wrf_like.nc", "bounded.nc",
                                  "trajectories.nc", "polar_projected.nc"])
def test_every_variable_in_every_tab(window, samples, name):
    window.open_path(samples[name])
    file_id = window.tree.current_file_id()
    for item in window.tree.variable_items(file_id):
        window.tree.setCurrentItem(item)
        for tab in range(window.tabs.count() - 1):  # all but the web map
            window.tabs.setCurrentIndex(tab)
            da = window.state.da
            for dim in da.dims:  # jump to the last index of every dim
                window.state.set_indices({dim: da.sizes[dim] - 1})
            assert _errors(window) == {}, f"{item.text(0)} in tab {tab}"


def test_info_tab_shows_tables(window, samples):
    window.open_path(samples["packed.nc"])
    info = window.info
    assert info.heading.text().count("packed.nc") == 1
    assert "t2m" in info.subheading.text()
    rows = {info.attributes.table.item(r, 0).text(): info.attributes.table.item(r, 1).text()
            for r in range(info.attributes.table.rowCount())}
    assert rows["scale_factor"] == "0.01"  # not in da.attrs after decoding, but in the file
    assert info.dimensions.isHidden() and info.variables.isHidden()

    window.tree.setCurrentItem(window.tree.file_item(window.tree.current_file_id()))
    assert "packed.nc" in info.heading.text()
    assert not info.dimensions.isHidden() and not info.variables.isHidden()
    variables = [info.variables.table.item(r, 0).text() for r in range(info.variables.table.rowCount())]
    assert "t2m" in variables
    assert info.dimensions.table.rowCount() == 2


def test_maps_are_available_only_with_coordinates(window, samples):
    window.open_path(samples["timeseries.nc"])
    window.tabs.setCurrentWidget(window.map)
    assert "no latitude/longitude" in window.map._message.text()
    window.open_path(samples["regular_global.nc"])
    assert window.map._stack.currentWidget() is window.map.body


def test_sliders_follow_state(window, samples):
    window.open_path(samples["four_d.nc"])
    window.tabs.setCurrentWidget(window.table)
    rows = {r.dim: r for r in window.dims._rows}
    assert set(rows) == {"time", "depth"}
    window.state.set_indices({"depth": 3})
    assert rows["depth"].slider.value() == 3
    assert rows["depth"].value.text() == "50 m"
    assert window.table.model.value(0, 0) == pytest.approx(
        float(window.state.da.isel(time=0, depth=3, lat=0, lon=0)))


def test_labels_name_their_controls_and_mnemonics_are_unique(window):
    menus = [QKeySequence.mnemonic(a.text()) for a in window.menuBar().actions()]
    for tab in range(window.tabs.count()):
        page = window.tabs.widget(tab)
        labels = [w for w in page.findChildren(QLabel) if "&" in w.text()]
        assert all(label.buddy() is not None for label in labels), window.tabs.tabText(tab)
        texts = [w.text() for w in labels + page.findChildren(QAbstractButton)]
        keys = menus + [k for k in map(QKeySequence.mnemonic, texts) if not k.isEmpty()]
        assert len(keys) == len(set(keys)), window.tabs.tabText(tab)


def test_dimension_controls_are_named_for_screen_readers(window, samples):
    window.open_path(samples["four_d.nc"])
    window.tabs.setCurrentWidget(window.table)
    depth = {r.dim: r for r in window.dims._rows}["depth"]
    window.state.set_indices({"depth": 3})
    assert depth.slider.accessibleName() == "depth index"
    assert depth.slider.accessibleDescription() == "50 m"
    assert depth.play.accessibleName() == "Play depth"
    depth.play.setChecked(True)
    assert depth.play.accessibleName() == "Pause depth"
    depth.play.setChecked(False)
    window.dims.focus_first()
    assert window.dims._rows[0].slider.hasFocus()


def test_time_starts_at_first_step_when_switching_variables(window, samples):
    window.open_path(samples["points.nc"])
    items = {i.text(0): i for i in window.tree.variable_items(window.tree.current_file_id())}
    window.tree.setCurrentItem(items["precip"])
    window.state.set_indices({"time": 4})
    window.tree.setCurrentItem(items["elevation"])  # has no time dimension
    window.tree.setCurrentItem(items["precip"])
    window.tabs.setCurrentWidget(window.map)  # maps stations; time gets a slider
    assert window.state.indices["time"] == 0
    assert [r.dim for r in window.dims._rows] == ["time"]
    assert window.dims._rows[0].slider.value() == 0


def test_depth_kept_when_switching_variables(window, samples):
    window.open_path(samples["four_d.nc"])
    window.tabs.setCurrentWidget(window.table)
    window.state.set_indices({"time": 2, "depth": 3})
    item = window.tree.currentItem()
    window.tree.setCurrentItem(window.tree.file_item(window.tree.current_file_id()))
    window.tree.setCurrentItem(item)
    assert window.state.indices == {"time": 0, "depth": 3}


def test_opening_the_same_file_twice_selects_it(window, samples):
    window.open_path(samples["packed.nc"])
    window.open_path(samples["packed.nc"])
    assert window.tree.topLevelItemCount() == 1


def test_close_file(window, samples):
    window.open_path(samples["packed.nc"])
    window.close_current_file()
    assert window.tree.topLevelItemCount() == 0
    assert window.state.da is None


def test_table_export_frame(window, samples):
    window.open_path(samples["four_d.nc"])
    window.tabs.setCurrentWidget(window.table)
    frame = to_frame(window.table.model._da)
    assert frame.shape == (30, 40)
    assert frame.index.name == "lat" and frame.columns.name == "lon"
    np.testing.assert_allclose(frame.values, window.state.da.isel(time=0, depth=0).values)


def test_source_context_finds_resources():
    ctx = SourceContext.__new__(SourceContext)  # without creating a second QApplication
    assert Path(ctx.get_resource("web", "map.html")).is_file()
    assert Path(ctx.get_resource("Icon.ico")).is_file()
    assert ctx.build_settings["app_name"] == "NetSeeDF"


# --- more detail after zooming in -------------------------------------------------

def _select(window, name):
    for item in window.tree.variable_items(window.tree.current_file_id()):
        if item.text(0) == name:
            window.tree.setCurrentItem(item)


def test_heatmap_loads_detail_when_zoomed(window, samples, monkeypatch):
    from netseedf.ui import plot_view
    monkeypatch.setattr(plot_view, "MAX_IMAGE_SIZE", 30)
    window.open_path(samples["regular_global.nc"])
    window.tabs.setCurrentWidget(window.plot)
    view, sst = window.plot, window.state.da
    assert view._base_slice.steps == (3, 6)
    assert "zoom in" in view.note.text()
    view._ax.set_xlim(0, 30)
    view._ax.set_ylim(40, 60)
    view._update_detail()
    assert view._detail is not None and not view._detail.downsampled
    assert view.note.text().strip() == "full resolution in view"
    # lon 10 isn't in the overview (every 6th column: 0, 12, ...), only in the detail.
    text = view._ax.format_coord(10, 45)
    expected = float(sst.isel(time=0).sel(lat=45, lon=10))
    assert f"sst = {expected:.6g} K" in text
    view._ax.set_xlim(0, 358)
    view._ax.set_ylim(-89, 89)
    view._update_detail()
    assert view._detail is None


def test_line_plot_loads_detail_when_zoomed(window, samples, monkeypatch):
    import matplotlib.dates as mdates

    from netseedf.ui import plot_view
    monkeypatch.setattr(plot_view, "MAX_LINE_POINTS", 50)
    window.open_path(samples["timeseries.nc"])
    window.tabs.setCurrentWidget(window.plot)
    view, q = window.plot, window.state.da
    assert view._base_slice.steps == (8,)
    t0, t1 = np.datetime64("2023-05-01"), np.datetime64("2023-05-21")
    view._ax.set_xlim(mdates.date2num(t0), mdates.date2num(t1))
    view._update_detail()
    assert view._detail is not None and not view._detail.downsampled
    shown = view._line_index
    assert np.all(np.diff(shown) > 0)  # overview and detail spliced into one ordered line
    day = int(np.flatnonzero(q.time.values == np.datetime64("2023-05-10"))[0])
    assert day in shown and day % 8  # a day the overview skips
    assert f"discharge = {float(q[day]):.6g}" in view._ax.format_coord(
        mdates.date2num(np.datetime64("2023-05-10")), 0)


def test_map_loads_detail_when_zoomed(window, samples, monkeypatch):
    import cartopy.crs as ccrs

    from netseedf.ui import cartopy_map_view
    monkeypatch.setattr(cartopy_map_view, "MAX_SIZE_IMAGE", 30)
    monkeypatch.setattr(cartopy_map_view, "MAX_SIZE_MESH", 30)
    window.open_path(samples["regular_global.nc"])
    window.tabs.setCurrentWidget(window.map)
    view, sst = window.map, window.state.da
    assert view._grid.slice.downsampled
    view._ax.set_extent((0, 30, 40, 60), crs=ccrs.PlateCarree())
    view._update_detail()
    assert view._detail is not None and not view._detail.slice.downsampled
    assert view.note.text().strip() == "full resolution in view"
    x, y = view._ax.projection.transform_point(10, 45, ccrs.PlateCarree())
    expected = float(sst.isel(time=0).sel(lat=45, lon=10))
    assert f"sst = {expected:.6g} K" in view._ax.format_coord(x, y)


def test_map_draws_projected_grid_in_its_projection(window, samples):
    import cartopy.crs as ccrs
    from matplotlib.collections import PolyQuadMesh

    window.open_path(samples["polar_projected.nc"])
    window.tabs.setCurrentWidget(window.map)
    view, ice = window.map, window.state.da
    view.projection.setCurrentText("Orthographic")
    params = view._ax.projection.proj4_params
    assert (params["lat_0"], params["lon_0"]) == pytest.approx((90, 0))  # looking down on the pole
    assert not hasattr(view._artist, "_wrapped_collection_fix")  # nothing crosses an edge there

    # In Plate Carrée the cells across the antimeridian are cut there, not smeared across the map.
    view.projection.setCurrentText("Plate Carrée")
    artist = view._artist
    fix = artist._wrapped_collection_fix
    assert isinstance(fix, PolyQuadMesh) and fix.axes is view._ax
    assert artist._wrapped_mask.any() and not artist._wrapped_mask.all()
    view.mpl.canvas.draw()

    window.state.set_indices({"time": 1})  # new values go into both parts of the same mesh
    assert view._artist is artist
    shown = np.ma.filled(artist.get_array(), np.nan)
    np.testing.assert_allclose(shown, ice.isel(time=1).values)
    x, y = view._ax.projection.transform_point(30, 70, ccrs.PlateCarree())
    assert "ice = " in view._ax.format_coord(x, y)
    assert _errors(window) == {}


def test_map_detail_for_stations(window, samples, monkeypatch):
    import cartopy.crs as ccrs

    from netseedf.ui import cartopy_map_view
    monkeypatch.setattr(cartopy_map_view, "MAX_POINTS", 25)
    window.open_path(samples["points.nc"])
    _select(window, "elevation")
    window.tabs.setCurrentWidget(window.map)
    view = window.map
    assert view._grid.slice.steps == (2,)  # the overview has every other station
    view._ax.set_extent((0, 15, 45, 55), crs=ccrs.PlateCarree())
    view._update_detail()
    assert view._detail is not None and not view._detail.slice.downsampled
    lat, lon = window.state.dataset["lat"].values, window.state.dataset["lon"].values
    x0, x1, y0, y1 = view._ax.get_extent(crs=ccrs.PlateCarree())
    visible = np.flatnonzero((lat >= y0) & (lat <= y1) & (lon >= x0) & (lon <= x1))
    assert any(i % 2 for i in visible)  # stations the overview skipped...
    assert set(visible) <= set(view._detail.index[0])  # ...are all shown now


def _show_on_web_map(window, samples, name):
    """What the web map's refresh() sets up, without the page (it needs a GPU)."""
    window.open_path(samples[name])
    view, state = window.web_map, window.state
    view._grid = geo_grid(state.da, state.geo, state.indices, max_size=10)
    return view


def test_web_map_click_popup(window, samples):
    view = _show_on_web_map(window, samples, "four_d.nc")
    window.state.set_indices({"time": 1, "depth": 2})
    info = view.pick(50.3, 5.2)
    assert info["title"] == "salinity"
    assert info["value"].endswith(" psu")
    assert "time = 2022-02-01" in info["at"] and "depth = " in info["at"]
    assert info["export"] == "Export time series (3 values)…"
    da = window.state.da
    lat, lon = da.lat.sel(lat=50.3, method="nearest"), da.lon.sel(lon=5.2, method="nearest")
    assert (info["lat"], info["lon"]) == pytest.approx((float(lat), float(lon)))  # cell centre
    assert view.pick(10, 0) is None  # outside the data
    assert view._picked is None


def test_web_map_export_time_series(window, samples, tmp_path, monkeypatch):
    view = _show_on_web_map(window, samples, "four_d.nc")
    window.state.set_indices({"depth": 2})
    view.pick(50.3, 5.2)
    path = tmp_path / "point.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))
    view.export_point()
    frame = pd.read_csv(path)
    da, index = window.state.da, view._picked.index
    assert list(frame["time"]) == ["2022-01-01", "2022-02-01", "2022-03-01"]
    np.testing.assert_allclose(frame["salinity"], da.isel(depth=2, **index).values, rtol=1e-6)
    assert (frame["lat"] == view._picked.lat).all()


def test_web_map_no_export_without_time(window, samples):
    view = _show_on_web_map(window, samples, "points.nc")
    items = {i.text(0): i for i in window.tree.variable_items(window.tree.current_file_id())}
    window.tree.setCurrentItem(items["elevation"])  # has no time dimension
    view._grid = geo_grid(window.state.da, window.state.geo, {}, max_size=10)
    station = window.state.da.station.size // 2
    info = view.pick(float(window.state.da.lat[station]), float(window.state.da.lon[station]))
    assert info["value"].endswith(" m")
    assert info["export"] is None


def test_user_agent_names_app_homepage_and_contact():
    settings = {"app_name": "NetSeeDF", "version": "0.1.0", "homepage": "https://example.org",
                "contact": "maps@example.org"}
    assert user_agent(settings) == "NetSeeDF/0.1.0 (+https://example.org; contact: maps@example.org)"
    assert user_agent(SETTINGS) == "netseedf/0.0.0 (+https://example.org)"
    assert user_agent({"app_name": "a", "version": "1"}) == "a/1"


def _drawn_cmaps(view):
    """Names of the colormaps of everything colour-mapped in a view's figure."""
    return {m.get_cmap().name for ax in view.mpl.figure.axes for m in [*ax.images, *ax.collections]
            if m.get_array() is not None}


def test_colormap_is_shared_between_views(window, samples):
    window.open_path(samples["regular_global.nc"])
    window.tabs.setCurrentWidget(window.plot)
    window.plot.kind.setCurrentText("Heatmap")
    window.plot.style_bar.cmap.setCurrentText("magma")
    assert _drawn_cmaps(window.plot) == {"magma"}
    assert window.map.style_bar.style().cmap == window.web_map.style_bar.style().cmap == "magma"

    window.tabs.setCurrentWidget(window.map)  # hidden while the colormap changed: redrawn now
    assert _drawn_cmaps(window.map) == {"magma"}
    window.map.style_bar.cmap.setCurrentText("turbo")
    assert _drawn_cmaps(window.map) == {"turbo"}
    window.tabs.setCurrentWidget(window.plot)
    assert window.plot.style_bar.style().cmap == "turbo"
    assert _drawn_cmaps(window.plot) == {"turbo"}


def test_colour_range_auto_or_typed(window, samples):
    window.open_path(samples["four_d.nc"])
    window.tabs.setCurrentWidget(window.map)
    view, bar = window.map, window.map.style_bar
    sal = window.state.da

    def slice_range():
        sl = sal.isel({d: window.state.indices.get(d, 0) for d in ("time", "depth")})
        return pytest.approx((float(sl.min()), float(sl.max())), rel=1e-5)

    assert bar.auto.isChecked() and view._limits == slice_range()
    bar.vmin.editingFinished.emit()  # left the box without typing: still automatic
    assert bar.auto.isChecked()

    bar.vmin.setText("30")
    bar.vmin.setModified(True)  # as if typed
    bar.vmin.editingFinished.emit()
    assert not bar.auto.isChecked()
    typed = view._limits
    assert typed[0] == 30
    window.state.set_indices({"time": 2})  # the typed range stays while stepping through time
    assert view._limits == typed and view._limits != slice_range()

    bar.auto.setChecked(True)
    assert view._limits == slice_range()

    bar.auto.setChecked(False)
    window.open_path(samples["regular_global.nc"])  # another variable: automatic again
    assert bar.auto.isChecked()


def test_web_map_grid_lines(window, samples, monkeypatch):
    view = _show_on_web_map(window, samples, "four_d.nc")
    assert not view.grid_lines.isChecked()  # off by default
    view._view_box = (-180, 180, -85, 85)  # (west, east, south, north): whole grid in view
    view.grid_lines.setChecked(True)
    sent = view._pending["setGridLines"]  # the page isn't there, so the call waits
    assert sent.startswith("netseedf.setGridLines([[[") and "grid lines" not in view.note.text()

    monkeypatch.setattr(web_map_view, "MAX_GRID_LINES", 10)  # as if zoomed far out
    view._update_grid_lines()
    assert view._pending["setGridLines"] == "netseedf.setGridLines(null);"
    assert "zoom in to see grid lines" in view.note.text()

    view.grid_lines.setChecked(False)
    assert view._pending["setGridLines"] == "netseedf.setGridLines(null);"
    assert "grid lines" not in view.note.text()



@pytest.mark.parametrize(("name", "units", "attrs", "down"), [
    ("depth", "m", {"positive": "down"}, True),
    ("plev", "hPa", {}, True),  # pressure is down without `positive`
    ("height", "m", {"positive": "up"}, False),
])
def test_vertical_axis_follows_cf_direction(window, tmp_path, name, units, attrs, down):
    """CF 4.3: `positive` (or pressure units) says which way a vertical coordinate increases."""
    import netCDF4

    path = tmp_path / "vertical.nc"
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension(name, 3)
        nc.createDimension("lat", 4)
        z = nc.createVariable(name, "f8", (name,))
        z.setncatts({"units": units, **attrs})
        z[:] = [10, 500, 1000]
        lat = nc.createVariable("lat", "f8", ("lat",))
        lat.units = "degrees_north"
        lat[:] = [0, 10, 20, 30]
        nc.createVariable("v", "f4", (name, "lat"))[:] = np.arange(12).reshape(3, 4)
    window.open_path(path)
    window.tabs.setCurrentWidget(window.plot)
    assert window.plot.y.currentText() == name
    assert window.plot._ax.yaxis_inverted() == down


@pytest.mark.filterwarnings("error:The input coordinates to pcolormesh:UserWarning")
@pytest.mark.parametrize("projection", ["Equal Earth", "Plate Carrée"])
def test_map_of_swath_with_missing_positions(window, tmp_path, projection):
    """Missing lat/lon (CF 2.5.1) mustn't make the mesh's cell edges go wrong (matplotlib warns)."""
    import netCDF4

    path = tmp_path / "swath.nc"
    j, i = np.mgrid[0:30, 0:40]
    lat, lon = 40 + 0.5 * j + 0.05 * i, 5 + 0.5 * i - 0.05 * j
    lat[:, 32:] = lon[:, 32:] = lat[:4, :10] = lon[:4, :10] = -999  # e.g. beyond the limb
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("y", 30)
        nc.createDimension("x", 40)
        for name, values, units in (("lat", lat, "degrees_north"), ("lon", lon, "degrees_east")):
            v = nc.createVariable(name, "f8", ("y", "x"), fill_value=-999.0)
            v.units = units
            v[:] = values
        v = nc.createVariable("radiance", "f4", ("y", "x"))
        v.coordinates = "lat lon"
        v[:] = np.arange(1200).reshape(30, 40)
    window.open_path(path)
    window.tabs.setCurrentWidget(window.map)
    window.map.projection.setCurrentText(projection)
    assert window.state.geo.kind == "curvilinear"
    assert _errors(window) == {}
    window.map.mpl.canvas.draw()


TESTFILES = Path(__file__).parent / "testfiles"
REAL_FILES = sorted(p.name for p in TESTFILES.glob("*.nc")) if TESTFILES.is_dir() else []


@pytest.mark.skipif(not REAL_FILES, reason="tests/testfiles isn't there")
@pytest.mark.parametrize("name", REAL_FILES)
def test_real_files_in_every_tab(window, name):
    """Every variable of the (large, not in git) files in tests/testfiles, in every tab."""
    window.open_path(TESTFILES / name)
    file_id = window.tree.current_file_id()
    for item in window.tree.variable_items(file_id):
        window.tree.setCurrentItem(item)
        for tab in range(window.tabs.count() - 1):  # all but the web map
            window.tabs.setCurrentIndex(tab)
            assert _errors(window) == {}, f"{item.text(0)} in tab {tab}"


@pytest.mark.parametrize("n_points", [2, 3])  # 2: an index array that unpacks like (start, stop)
def test_web_map_zoom_on_stations_shown_in_full(window, samples, n_points):
    """Stations all on the map already: zooming in needs no detail, and mustn't fail."""
    view = _show_on_web_map(window, samples, "points.nc")
    view._max_size = web_map_view.MAX_POINTS
    view._grid = geo_grid(window.state.da, window.state.geo, window.state.indices, view._max_size)
    assert not view._grid.slice.downsampled
    lat, lon = window.state.dataset["lat"].values, window.state.dataset["lon"].values
    order = np.argsort(lon)
    near = order[:n_points]
    view._view_box = (float(lon[near].min()) - 0.01, float(lon[near].max()) + 0.01,
                      float(lat[near].min()) - 0.01, float(lat[near].max()) + 0.01)
    view._load_detail()  # raises, unlike _update_detail, which only prints the error
    assert view._detail is None


def _map_pixels(view):
    view.mpl.canvas.draw()
    return np.asarray(view.mpl.canvas.buffer_rgba()).copy()


@pytest.mark.parametrize("projection", ["Equal Earth", "Plate Carrée"])
def test_map_time_step_updates_in_place(window, samples, projection):
    """Another time step only swaps the values: the map looks exactly as if drawn anew."""
    window.open_path(samples["regular_global.nc"])
    window.tabs.setCurrentWidget(window.map)
    view = window.map
    view.projection.setCurrentText(projection)
    ax, colorbar = view._ax, view._colorbar
    window.state.set_indices({"time": 2})
    assert view._ax is ax and view._colorbar is colorbar  # nothing was rebuilt
    sst = window.state.da.isel(time=2).values
    assert colorbar.norm.vmin == pytest.approx(np.nanmin(sst))  # the automatic range follows
    assert "2024-01-03" in view._title.get_text()
    in_place = _map_pixels(view)
    view._layout = None  # draw it anew
    view.update_view()
    assert view._ax is not ax
    np.testing.assert_array_equal(in_place, _map_pixels(view))


def test_map_time_step_reloads_zoomed_detail(window, samples, monkeypatch):
    import cartopy.crs as ccrs

    from netseedf.ui import cartopy_map_view
    monkeypatch.setattr(cartopy_map_view, "MAX_SIZE_MESH", 30)
    window.open_path(samples["regular_global.nc"])
    window.tabs.setCurrentWidget(window.map)
    view = window.map
    view._ax.set_extent((0, 30, 40, 60), crs=ccrs.PlateCarree())
    view._update_detail()
    assert view._detail is not None
    window.state.set_indices({"time": 3})
    assert view._detail is not None and view._detail.slice.fixed == {"time": 3}
    x, y = view._ax.projection.transform_point(10, 45, ccrs.PlateCarree())
    expected = float(window.state.da.isel(time=3).sel(lat=45, lon=10))
    assert f"sst = {expected:.6g} K" in view._ax.format_coord(x, y)


def test_map_time_step_redraws_stations(window, samples):
    """Which stations have no value changes with time, so the points are drawn again."""
    window.open_path(samples["points.nc"])
    _select(window, "precip")
    window.tabs.setCurrentWidget(window.map)
    view = window.map
    ax, points = view._ax, view._artist
    window.state.set_indices({"time": 3})
    assert view._ax is ax and view._artist is not points
    assert view._colorbar.mappable is view._artist
    precip = window.state.da.isel(time=3).values
    np.testing.assert_array_equal(np.sort(view._artist.get_array()), np.sort(precip[np.isfinite(precip)]))


def test_map_drawn_anew_when_positions_move(window, samples):
    """Another trajectory (or WRF's XLAT at another time) puts the cells elsewhere."""
    window.open_path(samples["trajectories.nc"])
    window.tabs.setCurrentWidget(window.map)
    view = window.map
    ax = view._ax
    window.state.set_indices({"trajectory": 1})
    assert view._ax is not ax


def test_gridlines_placed_once_per_view(window, samples, monkeypatch):
    """Cartopy would place gridlines and labels on every draw; only a new view needs them."""
    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import Gridliner

    from netseedf.ui.cartopy_map_view import ViewGridliner
    placed = []
    original = Gridliner._draw_gridliner
    monkeypatch.setattr(Gridliner, "_draw_gridliner",
                        lambda self, *a, **k: (placed.append(1), original(self, *a, **k))[1])
    window.open_path(samples["regular_global.nc"])
    window.tabs.setCurrentWidget(window.map)
    view = window.map
    view.mpl.canvas.draw()
    gridliner = next(a for a in view._ax.artists if isinstance(a, ViewGridliner))
    assert gridliner.label_artists and gridliner.xline_artists
    assert all(lines.get_transform() == view._ax.transData  # projected once, not on every draw
               for lines in gridliner.xline_artists + gridliner.yline_artists)
    placed.clear()
    for t in (1, 2, 3):
        window.state.set_indices({"time": t})
        view.mpl.canvas.draw()
    assert placed == []
    view._ax.set_extent((0, 30, 40, 60), crs=ccrs.PlateCarree())
    view.mpl.canvas.draw()
    assert len(placed) == 1
    labels = {label.get_text() for label in gridliner.label_artists if label.get_visible()}
    assert "10°E" in labels and "50°N" in labels  # labels of the zoomed-in view


def test_web_map_outlines_for_the_none_basemap():
    """The offline "None" basemap: coastlines and borders from the bundled Natural Earth files."""
    outlines = web_map_view.outlines_geojson(resource)
    assert set(outlines) == {"coastlines", "borders"}
    for name, collection in outlines.items():
        assert collection is not None, name
        assert collection["type"] == "FeatureCollection" and len(collection["features"]) > 100, name
        assert collection["features"][0]["geometry"]["type"] in ("LineString", "MultiLineString"), name
