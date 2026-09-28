"""Web map tab: Leaflet slippy map in QtWebEngine with the data drawn on top."""

import json
import sys
import traceback

import numpy as np
from PySide6.QtCore import QFile, QIODevice, QObject, Qt, QTimer, QUrl, Slot
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import (
    QWebEnginePage,
    QWebEngineProfile,
    QWebEngineScript,
    QWebEngineSettings,
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout

from netseedf.core.coords import GridLocator, geo_grid, geo_window, with_cyclic_column
from netseedf.core.detail import DetailTracker, padded, resolution_note
from netseedf.core.formatting import format_value, position_text, value_text, variable_label
from netseedf.core.render import legend_colors, point_colors, render_overlay
from netseedf.core.slicing import is_finer
from netseedf.ui.base_view import DataView, wait_cursor
from netseedf.ui.cartopy_map_view import NO_GEO_MESSAGE
from netseedf.ui.style_bar import StyleBar

MAX_SIZE_REGULAR = 2000
MAX_SIZE_CURVILINEAR = 800
MAX_POINTS = 20_000
COASTLINE_SCALE = "110m"
DETAIL_DELAY_MS = 150


class _Bridge(QObject):
    """Exposed to the page's JavaScript as `bridge`."""

    def __init__(self, view):
        super().__init__(view)
        self._view = view

    @Slot()
    def ready(self):
        self._view._page_ready()

    @Slot(float, float, result=str)
    def valueAt(self, lat, lon):  # noqa: N802 (called from JavaScript)
        return self._view.hover_text(lat, lon)

    @Slot(float, float, float, float)
    def viewChanged(self, south, west, north, east):  # noqa: N802 (called from JavaScript)
        self._view._view_changed(south, west, north, east)


class _Page(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, line, source):  # noqa: N802
        print(f"[web map] {source}:{line}: {message}", file=sys.stderr)


class WebMapView(DataView):
    title = "Web map"

    def __init__(self, state, resources, app_id, homepage, parent=None):
        super().__init__(state, parent)
        self._resources = resources
        self._app_id = app_id
        self._homepage = homepage
        self.style_bar = StyleBar()
        self.opacity = QSlider(Qt.Orientation.Horizontal, minimum=0, maximum=100, value=75,
                               maximumWidth=120, toolTip="Opacity of the data layer")
        self.note = QLabel()
        bar = QHBoxLayout()
        for w in (self.style_bar, QLabel("Opacity:"), self.opacity):
            bar.addWidget(w)
        bar.addWidget(self.note, 1)
        self._layout = QVBoxLayout(self.body)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.addLayout(bar)
        self.web = None  # created on first use; QtWebEngine is slow to start
        self._ready = False
        self._pending: dict[str, str] = {}
        self._locator = self._detail_locator = None
        self._grid = self._detail = None
        self._view_box = None  # visible (west, east, south, north), from the page
        self._fit = True
        self._tracker = DetailTracker()
        self._detail_timer = QTimer(self, singleShot=True, interval=DETAIL_DELAY_MS)
        self._detail_timer.timeout.connect(self._update_detail)
        self.style_bar.changed.connect(self._redraw)
        self.opacity.valueChanged.connect(lambda v: self._call("setOpacity", v / 100))

    def unavailable_reason(self):
        da = self.state.da
        if self.state.geo is None:
            return NO_GEO_MESSAGE.format(name=da.name)
        if da.dtype.kind not in "fiub":
            return f"'{da.name}' holds {da.dtype} values, which can't be coloured on a map."
        return None

    def free_dims(self):
        geo = self.state.geo
        return geo.dims if geo is not None else super().free_dims()

    def variable_changed(self):
        self._fit = True
        self._locator = self._detail_locator = self._grid = self._detail = None
        self.style_bar.reset_lock()

    def _redraw(self):
        if self.state.da is not None and self.isVisible():
            self.update_view()

    def refresh(self):
        self._ensure_web()
        self._detail_timer.stop()
        self._tracker.reset()
        self._detail = self._detail_locator = None
        da, geo = self.state.da, self.state.geo
        self._max_size = {"points": MAX_POINTS, "curvilinear": MAX_SIZE_CURVILINEAR}.get(
            geo.kind, MAX_SIZE_REGULAR)
        grid = geo_grid(da, geo, self.state.indices, self._max_size)
        self._limits = self.style_bar.limits_for(grid.values)
        self._cmap = self.style_bar.style().cmap
        payload = self._payload(with_cyclic_column(grid))
        if payload is None:
            raise ValueError("All of the data is closer to a pole than 85°, which web maps "
                             "can't show. Use the Map tab instead.")
        vmin, vmax = self._limits
        payload["fit"] = self._fit
        payload["legend"] = {"title": variable_label(da), "colors": legend_colors(self._cmap),
                             "min": format_value(vmin, 4), "max": format_value(vmax, 4)}
        self._fit = False
        self._grid, self._locator = grid, GridLocator(grid)
        self._call("setData", payload)
        self._update_note()
        self._detail_timer.start()  # the map may be zoomed in already

    def _payload(self, grid):
        vmin, vmax = self._limits
        if grid.kind == "points":
            ok = np.isfinite(grid.lat) & np.isfinite(grid.lon)
            colors = point_colors(grid.values[ok], self._cmap, vmin, vmax)
            return {"kind": "points", "points": [
                [float(la), float(lo), c] for la, lo, c in zip(grid.lat[ok], grid.lon[ok], colors,
                                                               strict=True)]}
        overlay = render_overlay(grid, self._cmap, vmin, vmax)
        if overlay is None:
            return None
        return {"kind": "image", "url": overlay.data_url,
                "bounds": [[overlay.south, overlay.west], [overlay.north, overlay.east]]}

    def _update_note(self):
        if self._grid is not None:
            detail = self._detail.slice if self._detail is not None else None
            self.note.setText(resolution_note(self._grid.slice, detail))

    def hover_text(self, lat, lon):
        if self._locator is None:
            return ""
        text = position_text(lat, lon)
        found = self._detail_locator.value_at(lat, lon) if self._detail_locator else None
        found = found or self._locator.value_at(lat, lon)
        if found is not None:
            text += f"   {self.state.da.name} = {value_text(self.state.da, found[1])}"
        self.status.emit(text)
        return text

    # --- more detail when zoomed in ------------------------------------------

    def _view_changed(self, south, west, north, east):
        self._view_box = (west, east, south, north)
        self._detail_timer.start()

    def _update_detail(self):
        if self._grid is None or self._view_box is None or not self.isVisible():
            return
        try:
            with wait_cursor():
                self._load_detail()
        except Exception:  # the overview is still there; don't lose it over the detail
            traceback.print_exc()

    def _load_detail(self):
        visible = self._view_box
        if self._detail is not None and self._tracker.up_to_date(visible):
            return
        x0, x1, y0, y1 = padded(visible)
        box = (max(y0, -90), x0, min(y1, 90), x1)
        da, geo = self.state.da, self.state.geo
        window = geo_window(geo, self._grid, box)
        detail = payload = None
        if window is not None and is_finer(self._grid.slice, window, self._max_size):
            detail = geo_grid(da, geo, self.state.indices, self._max_size, window, like=self._grid)
            payload = self._payload(detail)
        if payload is None:
            if self._detail is not None:
                self._call("setDetail", None)
            self._detail = self._detail_locator = None
            self._tracker.reset()
        else:
            payload["covers"] = [[y0, x0], [y1, x1]]
            self._call("setDetail", payload)
            self._detail, self._detail_locator = detail, GridLocator(detail)
            self._tracker.remember(visible, (x0, x1, y0, y1))
        self._update_note()

    # --- talking to the page ---------------------------------------------

    def _call(self, function, *args):
        js = f"netseedf.{function}({', '.join(json.dumps(a) for a in args)});"
        if self._ready:
            self.web.page().runJavaScript(js)
        else:
            self._pending[function] = js  # only the latest call of each kind matters

    def _page_ready(self):
        self._ready = True
        coastlines = _coastlines_geojson(self._resources)
        if coastlines is not None:
            self.web.page().runJavaScript(f"netseedf.setCoastlines({json.dumps(coastlines)});")
        self._call("setOpacity", self.opacity.value() / 100)
        pending, self._pending = self._pending, {}
        for js in pending.values():
            self.web.page().runJavaScript(js)

    def _ensure_web(self):
        if self.web is not None:
            return
        profile = QWebEngineProfile.defaultProfile()
        # Tile servers (OpenStreetMap's usage policy) want to know which app is asking.
        profile.setHttpUserAgent(f"{profile.httpUserAgent()} {self._app_id} (+{self._homepage})")

        self.web = QWebEngineView()
        page = _Page(profile, self.web)
        page.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        page.scripts().insert(_qwebchannel_script())
        channel = QWebChannel(page)
        self._bridge = _Bridge(self)
        channel.registerObject("bridge", self._bridge)
        page.setWebChannel(channel)
        self.web.setPage(page)
        self.web.setUrl(QUrl.fromLocalFile(self._resources("web", "map.html")))
        self._layout.addWidget(self.web, 1)


def _qwebchannel_script():
    f = QFile(":/qtwebchannel/qwebchannel.js")
    if not f.open(QIODevice.OpenModeFlag.ReadOnly):
        raise RuntimeError("qwebchannel.js is missing from the Qt installation")
    source = bytes(f.readAll()).decode()
    f.close()
    script = QWebEngineScript()
    script.setName("qwebchannel")
    script.setSourceCode(source)
    script.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentCreation)
    script.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
    script.setRunsOnSubFrames(False)
    return script


_coastlines_cache = {}


def _coastlines_geojson(resources):
    """Natural Earth coastlines as GeoJSON, for the offline "Coastlines" layer."""
    if "data" not in _coastlines_cache:
        try:
            import cartopy.io.shapereader as shpreader
            from shapely.geometry import mapping

            path = resources("cartopy", "shapefiles", "natural_earth", "physical",
                             f"ne_{COASTLINE_SCALE}_coastline.shp")
            features = [{"type": "Feature", "properties": {}, "geometry": mapping(g)}
                        for g in shpreader.Reader(path).geometries()]
            _coastlines_cache["data"] = {"type": "FeatureCollection", "features": features}
        except Exception as e:  # the web map still works without them
            print(f"Couldn't load coastlines for the web map: {e}", file=sys.stderr)
            _coastlines_cache["data"] = None
    return _coastlines_cache["data"]
