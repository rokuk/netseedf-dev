"""Interactive map tab: Leaflet slippy map in QtWebEngine with the data drawn on top."""

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
    QWebEngineUrlRequestInterceptor,
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSlider,
    QVBoxLayout,
)

from netseedf.core.coords import (
    GridLocator,
    geo_grid,
    geo_window,
    pick_point,
    with_cyclic_column,
)
from netseedf.core.detail import DetailTracker, padded, resolution_note
from netseedf.core.formatting import (
    format_value,
    position_text,
    selection_text,
    value_text,
    variable_label,
)
from netseedf.core.gridlines import grid_lines
from netseedf.core.render import (
    MAX_POLYGON_CELLS,
    cell_polygons,
    legend_colors,
    point_colors,
    render_overlay,
)
from netseedf.core.slicing import fixed_indices, is_finer, point_series, series_frame, time_dims
from netseedf.ui.base_view import DataView, wait_cursor
from netseedf.ui.cartopy_map_view import NO_GEO_MESSAGE, OUTLINE_SCALE
from netseedf.ui.style_bar import StyleBar

MAX_SIZE_REGULAR = 2000
MAX_SIZE_CURVILINEAR = 800
MAX_POINTS = 20_000
DETAIL_DELAY_MS = 150
MAX_GRID_LINES = 400  # more than this in view and the lines are hidden
WEB_PROFILE_NAME = "webmap"  # names the on-disk cache and storage folders


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

    @Slot(float, float, result=str)
    def pick(self, lat, lon):
        return json.dumps(self._view.pick(lat, lon))

    @Slot()
    def exportPoint(self):  # noqa: N802 (called from JavaScript)
        # Not from inside the call from the page: the file dialog runs its own event loop.
        QTimer.singleShot(0, self._view.export_point)

    @Slot(float, float, float, float)
    def viewChanged(self, south, west, north, east):  # noqa: N802 (called from JavaScript)
        self._view._view_changed(south, west, north, east)


class _Page(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, line, source):  # noqa: N802
        print(f"[web map] {source}:{line}: {message}", file=sys.stderr)


class WebMapView(DataView):
    title = "Interactive map"

    def __init__(self, state, resources, user_agent, parent=None):
        super().__init__(state, parent)
        self._resources = resources
        self._user_agent = user_agent  # identifies the app to tile servers
        self.style_bar = StyleBar()
        self.opacity = QSlider(Qt.Orientation.Horizontal, minimum=0, maximum=100, value=75,
                               maximumWidth=120, toolTip="Opacity of the data layer")
        self.grid_lines = QCheckBox("Grid lines", toolTip="Outline the cells of the data grid "
                                    "(when zoomed in far enough to draw them all)")
        self.note = QLabel(minimumWidth=1)  # a long note mustn't widen the window
        bar = QHBoxLayout()
        for w in (self.style_bar, QLabel("Opacity:"), self.opacity, self.grid_lines):
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
        self._overview_cells = False  # the overview is drawn as polygons, see _wants_detail
        self._picked = None  # the cell last clicked on, see pick()
        self._view_box = None  # visible (west, east, south, north), from the page
        self._grid_note = ""  # why grid lines aren't shown, if they aren't
        self._fit = True
        self._tracker = DetailTracker()
        self._detail_timer = QTimer(self, singleShot=True, interval=DETAIL_DELAY_MS)
        self._detail_timer.timeout.connect(self._update_detail)
        self.style_bar.changed.connect(self._redraw)
        self.opacity.valueChanged.connect(lambda v: self._call("setOpacity", v / 100))
        self.grid_lines.toggled.connect(self._update_grid_lines)

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
        self._picked = None
        self._call("closePopup")
        self.style_bar.reset_range()

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
        self.grid_lines.setEnabled(grid.kind != "points")
        self.grid_lines.setToolTip("Stations have no grid cells to outline." if grid.kind == "points"
                                   else "Outline the cells of the data grid "
                                        "(when zoomed in far enough to draw them all)")
        self._overview_cells = payload["kind"] == "cells"
        self._call("setData", payload)
        self._update_note()
        self._update_detail()  # now: the map may be zoomed in already

    def _payload(self, grid):
        vmin, vmax = self._limits
        if grid.kind == "points":
            ok = np.isfinite(grid.lat) & np.isfinite(grid.lon)
            colors = point_colors(grid.values[ok], self._cmap, vmin, vmax)
            return {"kind": "points", "points": [
                [float(la), float(lo), c] for la, lo, c in zip(grid.lat[ok], grid.lon[ok], colors,
                                                               strict=True)]}
        cells = cell_polygons(grid, self._cmap, vmin, vmax)
        if cells is not None:  # few enough cells to draw each one exactly
            return {"kind": "cells", "shape": cells.shape, "lat": cells.lat, "lon": cells.lon,
                    "colors": cells.colors,
                    "bounds": [[cells.south, cells.west], [cells.north, cells.east]]}
        overlay = render_overlay(grid, self._cmap, vmin, vmax)
        if overlay is None:
            return None
        payload = {"kind": "image", "url": overlay.data_url,
                   "bounds": [[overlay.south, overlay.west], [overlay.north, overlay.east]]}
        if overlay.rows is not None:
            payload.update(kind="grid", rows=overlay.rows, cols=overlay.cols)
        return payload

    def _update_note(self):
        if self._grid is not None:
            detail = self._detail.slice if self._detail is not None else None
            self.note.setText(resolution_note(self._grid.slice, detail) + self._grid_note)

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

    # --- clicking on a cell --------------------------------------------------

    def pick(self, lat, lon) -> dict | None:
        """What the popup shows for the cell at lat/lon (None when there's no data)."""
        self._picked = None
        if self._grid is None or self.state.da is None:
            return None
        da, geo, indices = self.state.da, self.state.geo, self.state.indices
        try:
            picked = None
            if self._detail is not None:
                picked = pick_point(da, geo, self._detail, lat, lon, indices)
            picked = picked or pick_point(da, geo, self._grid, lat, lon, indices)
        except Exception:
            traceback.print_exc()
            return None
        if picked is None:
            return None
        self._picked = picked
        series_dims = time_dims(da, exclude=picked.index)
        steps = int(np.prod([da.sizes[d] for d in series_dims]))
        return {
            "title": str(da.name),
            "value": value_text(da, picked.value),
            "position": position_text(picked.lat, picked.lon),
            "lat": float(picked.lat), "lon": float(picked.lon),  # the popup points at the cell centre
            "cell": ", ".join(f"{d} #{i}" for d, i in picked.index.items()),
            "at": selection_text(da, fixed_indices(da, geo.dims, indices)),
            "export": f"Export time series ({steps} values)…" if series_dims else None,
        }

    def export_point(self):
        """Save the values of the clicked cell for every time step as CSV."""
        picked, da, geo = self._picked, self.state.da, self.state.geo
        if picked is None or da is None or not time_dims(da, exclude=picked.index):
            return
        default = f"{da.name}_lat{picked.lat:.3f}_lon{picked.lon:.3f}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export time series", default,
                                              "CSV files (*.csv)")
        if not path:
            return
        try:
            with wait_cursor():
                series = point_series(da, picked.index, self.state.indices)
                extra = {str(geo.lat.name): picked.lat, str(geo.lon.name): picked.lon}
                series_frame(series, extra).to_csv(path)
        except Exception as e:
            traceback.print_exc()
            QMessageBox.warning(self, "Export failed", f"{path}\n\n{type(e).__name__}: {e}")

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
        self._update_grid_lines()

    def _load_detail(self):
        visible = self._view_box
        if self._detail is not None and self._tracker.up_to_date(visible):
            return
        x0, x1, y0, y1 = padded(visible)
        box = (max(y0, -90), x0, min(y1, 90), x1)
        da, geo = self.state.da, self.state.geo
        window = geo_window(geo, self._grid, box)
        detail = payload = None
        if window is not None and self._wants_detail(window):
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

    def _wants_detail(self, window):
        """Would loading `window` show more than the overview: finer, or cells as polygons?"""
        if is_finer(self._grid.slice, window, self._max_size):
            return True
        # A full-resolution curvilinear overview with too many cells for polygons is an
        # image; zoomed in, few enough of them may be in view to draw exactly.
        if self._grid.kind != "curvilinear" or self._overview_cells:
            return False  # (points have index arrays for windows, not start and stop)
        sizes = [stop - start for start, stop in window.values()]
        return max(sizes) <= self._max_size and int(np.prod(sizes)) <= MAX_POLYGON_CELLS

    # --- grid lines ---------------------------------------------------------------

    def _update_grid_lines(self):
        """Draw (or remove) the cell outlines for what's visible."""
        lines, self._grid_note = None, ""
        wanted = self.grid_lines.isChecked() and self.grid_lines.isEnabled()
        if wanted and self._grid is not None and self._view_box is not None:
            x0, x1, y0, y1 = padded(self._view_box)
            box = (max(y0, -90), x0, min(y1, 90), x1)
            try:
                lines = grid_lines(self.state.da, self.state.geo, self._grid, self.state.indices,
                                   box, MAX_GRID_LINES)
            except Exception:
                traceback.print_exc()
                lines = []
            if lines is None:
                self._grid_note = "  zoom in to see grid lines"
        self._call("setGridLines", lines)
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
        self.web.page().runJavaScript(f"netseedf.setOutlines({json.dumps(outlines_geojson(self._resources))});")
        self._call("setOpacity", self.opacity.value() / 100)
        pending, self._pending = self._pending, {}
        for js in pending.values():
            self.web.page().runJavaScript(js)

    def _ensure_web(self):
        if self.web is not None:
            return
        profile = _profile(self, self._user_agent)
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


class _TileInterceptor(QWebEngineUrlRequestInterceptor):
    """Identifies the app to tile servers that ask for it."""

    # OpenStreetMap's tile usage policy wants a User-Agent naming the app, not a browser's.
    APP_ONLY_HOSTS = {"tile.openstreetmap.org"}

    def __init__(self, user_agent, parent=None):
        super().__init__(parent)
        self._user_agent = user_agent.encode()

    def interceptRequest(self, info):  # noqa: N802 (Qt override)
        if info.requestUrl().host() in self.APP_ONLY_HOSTS:
            info.setHttpHeader(b"User-Agent", self._user_agent)


def _profile(parent, user_agent) -> QWebEngineProfile:
    """Web profile that keeps downloaded tiles on disk between runs.

    Tile servers ask clients to honour their caching headers (OpenStreetMap's
    tiles stay fresh for about a week), which Qt's default, in-memory-only
    profile can't do across restarts. The cache lives in the per-user cache
    folder (from the application and organization names), never next to the
    program, so it also works when the app is frozen and installed read-only.
    """
    profile = QWebEngineProfile(WEB_PROFILE_NAME, parent)
    profile.setHttpCacheType(QWebEngineProfile.HttpCacheType.DiskHttpCache)
    profile.setPersistentCookiesPolicy(QWebEngineProfile.PersistentCookiesPolicy.NoPersistentCookies)
    # Other tile servers get the browser's User-Agent with the app appended.
    profile.setHttpUserAgent(f"{profile.httpUserAgent()} {user_agent}")
    # Kept referenced from Python: a Qt parent alone keeps the C++ object, but
    # not the Python subclass whose interceptRequest does the work.
    profile.interceptor = _TileInterceptor(user_agent, profile)
    profile.setUrlRequestInterceptor(profile.interceptor)
    return profile


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


OUTLINES = {  # name: (Natural Earth category, file)
    "coastlines": ("physical", "coastline"),
    "borders": ("cultural", "admin_0_boundary_lines_land"),
}
_outlines_cache = {}


def outlines_geojson(resources) -> dict:
    """Natural Earth coastlines and borders as GeoJSON, for the offline "None" basemap.

    {name: FeatureCollection, or None if that file couldn't be read}.
    """
    for name, (category, file) in OUTLINES.items():
        if name not in _outlines_cache:
            _outlines_cache[name] = _shapes_geojson(resources, category, f"ne_{OUTLINE_SCALE}_{file}.shp")
    return dict(_outlines_cache)


def _shapes_geojson(resources, category, file):
    try:
        import cartopy.io.shapereader as shpreader
        from shapely.geometry import mapping

        path = resources("cartopy", "shapefiles", "natural_earth", category, file)
        features = [{"type": "Feature", "properties": {}, "geometry": mapping(g)}
                    for g in shpreader.Reader(path).geometries()]
        return {"type": "FeatureCollection", "features": features}
    except Exception as e:  # the web map still works without them
        print(f"Couldn't load {file} for the web map: {e}", file=sys.stderr)
        return None
