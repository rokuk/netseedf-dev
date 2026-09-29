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
from PySide6.QtWidgets import QFileDialog  # noqa: E402

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
                                  "points.nc", "pacific.nc", "wrf_like.nc"])
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

