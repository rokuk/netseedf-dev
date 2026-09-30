"""Map tab: Cartopy map with coastlines and a choice of projections (works offline)."""

import traceback
from pathlib import Path

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from cartopy.mpl.gridliner import Gridliner
from matplotlib.image import AxesImage
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout

from netseedf.core.coords import (
    GridLocator,
    cells_from_bounds,
    fill_missing_positions,
    geo_grid,
    geo_window,
    with_cyclic_column,
)
from netseedf.core.detail import DetailTracker, padded, resolution_note
from netseedf.core.formatting import position_text, selection_text, value_text, variable_label
from netseedf.core.gridlines import cell_corners
from netseedf.core.render import cell_edges
from netseedf.core.slicing import is_finer
from netseedf.ui.base_view import DataView, wait_cursor
from netseedf.ui.mpl_canvas import MplWidget
from netseedf.ui.style_bar import StyleBar

# Grids are thinned to at most this many cells per side; reprojecting
# pcolormesh cells is much slower than showing an image in Plate Carrée.
MAX_SIZE_IMAGE = 2000
MAX_SIZE_MESH = 700
MAX_POINTS = 50_000

NO_GEO_MESSAGE = (
    "'{name}' has no latitude/longitude coordinates, so it can't be put on a map.\n\n"
    "Maps need coordinates with standard_name latitude/longitude, units degrees_north/"
    "degrees_east, or names like lat/lon."
)


def configure_offline_data(bundled_dir, cache_dir):
    """Use the bundled Natural Earth files and never write into the app bundle."""
    cartopy.config["pre_existing_data_dir"] = Path(bundled_dir)
    cartopy.config["data_dir"] = Path(cache_dir)


def _plate_carree(lon0, lat0):
    return ccrs.PlateCarree(central_longitude=lon0)


PROJECTIONS = {
    "Plate Carrée": _plate_carree,
    "Robinson": lambda lon0, lat0: ccrs.Robinson(central_longitude=lon0),
    "Mollweide": lambda lon0, lat0: ccrs.Mollweide(central_longitude=lon0),
    "Equal Earth": lambda lon0, lat0: ccrs.EqualEarth(central_longitude=lon0),
    "Mercator": lambda lon0, lat0: ccrs.Mercator(central_longitude=lon0),
    "Orthographic": lambda lon0, lat0: ccrs.Orthographic(lon0, lat0),
    "Lambert Conformal": lambda lon0, lat0: ccrs.LambertConformal(
        lon0, lat0, standard_parallels=(lat0 - 5, lat0 + 5) if abs(lat0) > 6 else (30, 60)),
    "North Polar Stereographic": lambda lon0, lat0: ccrs.NorthPolarStereo(central_longitude=lon0),
    "South Polar Stereographic": lambda lon0, lat0: ccrs.SouthPolarStereo(central_longitude=lon0),
}
POLAR = {"North Polar Stereographic", "South Polar Stereographic"}


# Coastlines and borders get more detailed as you zoom in (only bundled scales).
_SCALER = cfeature.AdaptiveScaler("110m", (("50m", 60), ("10m", 8)))
COASTLINES = cfeature.NaturalEarthFeature("physical", "coastline", _SCALER)
BORDERS = cfeature.NaturalEarthFeature("cultural", "admin_0_boundary_lines_land", _SCALER)
DETAIL_DELAY_MS = 300

# Drawing order: data, then the zoomed-in detail, then coastlines; stations on top.
Z_DATA, Z_DETAIL, Z_FEATURES, Z_POINTS = 1, 1.2, 2, 3


class ViewGridliner(Gridliner):
    """Gridlines and their labels, placed anew only when the view changes.

    Cartopy's Gridliner places them on every draw, which with a reprojected map
    takes most of the drawing time (~0.4 s), e.g. for each time step. They only
    depend on what's in view and how large the figure is.

    The placing is done by _draw_gridliner, which GeoAxes also calls directly (to
    put the title above the labels), so that's where it's skipped. Should cartopy
    rename it, the gridlines are simply placed on every draw again.
    """

    def __init__(self, axes, **kwargs):
        super().__init__(axes, crs=ccrs.PlateCarree(), **kwargs)
        self._placed_for = None

    def _draw_gridliner(self, *args, **kwargs):
        # (The figure's size, not the axes': the layout moves the axes about while it works.)
        view = (tuple(self.axes.viewLim.bounds), tuple(self.axes.figure.bbox.bounds))
        if view == self._placed_for:
            return
        self._placed_for = view
        super()._draw_gridliner(*args, **kwargs)
        # The lines are in degrees, so cartopy would project them on every draw and to
        # measure them for the layout (~20 ms each). Project them once, here instead.
        # (Cartopy sets both the lines and their transform again for the next view.)
        for lines in self.xline_artists + self.yline_artists:
            to_map = lines.get_transform() - self.axes.transData
            lines.set_segments([part for path in lines.get_paths()  # split where they leave the map
                                for part in to_map.transform_path(path).to_polygons(closed_only=False)])
            lines.set_transform(self.axes.transData)


class CartopyMapView(DataView):
    title = "Map"

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self.projection = QComboBox()
        self.projection.addItems(PROJECTIONS)
        self.projection.setCurrentText("Equal Earth")
        self.style_bar = StyleBar()
        self.note = QLabel(minimumWidth=1)  # a long note mustn't widen the window
        bar = QHBoxLayout()
        for w in (QLabel("Projection:"), self.projection):
            bar.addWidget(w)
        bar.addWidget(self.note, 1)
        self.mpl = MplWidget()
        self.mpl.add_to_toolbar_row(self.style_bar)
        self.mpl.hover.connect(self.status)
        layout = QVBoxLayout(self.body)
        layout.setContentsMargins(4, 4, 4, 0)
        layout.addLayout(bar)
        layout.addWidget(self.mpl, 1)
        self.projection.currentTextChanged.connect(self._redraw)
        self.style_bar.changed.connect(self._redraw)
        self._detail_timer = QTimer(self, singleShot=True, interval=DETAIL_DELAY_MS)
        self._detail_timer.timeout.connect(self._update_detail)
        self._tracker = DetailTracker()
        self._ax = self._detail = self._detail_artist = self._detail_locator = None
        self._layout = None  # what the map on screen was made for, see _layout_key

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
        self.style_bar.reset_range()
        self.mpl.reset()

    def _redraw(self):
        if self.state.da is not None and self.isVisible():
            self.update_view()

    def _layout_key(self):
        """What the map depends on apart from the values shown (GeoInfo compares by identity)."""
        return self.state.ref, self.state.geo, self.projection.currentText(), self.style_bar.style().cmap

    def refresh(self):
        self._detail_timer.stop()
        if self._update_values():
            return
        self._tracker.reset()
        self._ax = self._detail = self._detail_artist = self._detail_locator = None
        self._layout = self._layout_key()
        da, geo = self.state.da, self.state.geo
        name = self.projection.currentText()
        self._mesh = geo.kind == "curvilinear" or name != "Plate Carrée" or cells_from_bounds(geo)
        if geo.kind == "points":
            self._max_size = MAX_POINTS
        else:
            self._max_size = MAX_SIZE_MESH if self._mesh else MAX_SIZE_IMAGE
        self._grid = grid = geo_grid(da, geo, self.state.indices, self._max_size)
        self._locator = GridLocator(grid)
        self._limits = self.style_bar.limits_for(grid.values)

        south, west, north, east = grid.bounds
        self._lon0 = lon0 = 0.0 if grid.is_global else round((west + east) / 2)
        lat0 = float(np.clip((south + north) / 2, -80, 80))
        proj = PROJECTIONS[name](lon0, lat0)
        data_crs = ccrs.PlateCarree()

        previous = self.mpl.begin((self.state.ref, name))
        ax = self.mpl.figure.add_subplot(projection=proj)
        self._artist = artist = self._draw(ax, with_cyclic_column(grid), detail=False)
        self._set_extent(ax, name, grid, lon0)
        for feature, style in ((COASTLINES, dict(edgecolor="black", linewidth=0.6)),
                               (BORDERS, dict(edgecolor="0.3", linewidth=0.4))):
            _outside_layout(ax.add_feature(feature, facecolor="none", zorder=Z_FEATURES, **style))
        ax.add_artist(ViewGridliner(ax, draw_labels=True, collection_kwargs=dict(
            linewidth=0.3, color="gray", alpha=0.6, linestyle="--")))

        self._colorbar = self.mpl.figure.colorbar(artist, ax=ax, orientation="horizontal", shrink=0.7,
                                                  pad=0.02, aspect=40, label=variable_label(da))
        self._title = ax.set_title(self._title_text(grid), fontsize="medium")

        def format_coord(x, y):
            lon, lat = data_crs.transform_point(x, y, proj)
            if not (np.isfinite(lat) and np.isfinite(lon)):
                return ""
            found = self._detail_locator.value_at(lat, lon) if self._detail_locator else None
            found = found or self._locator.value_at(lat, lon)
            text = position_text(lat, lon)
            return text if found is None else f"{text}   {da.name} = {value_text(da, found[1])}"
        ax.format_coord = format_coord
        self._ax = ax
        self._update_note()
        self.mpl.finish(ax, previous)
        # Zooming or panning (including the zoom kept from before) may call for more detail.
        for event in ("xlim_changed", "ylim_changed"):
            ax.callbacks.connect(event, lambda _ax: self._detail_timer.start())
        self._update_detail()  # now, so a kept zoom never shows the overview alone

    def _title_text(self, grid):
        where = selection_text(self.state.da, grid.slice.fixed)
        return f"{self.state.da.name}" + (f"   {where}" if where else "")

    def _update_values(self) -> bool:
        """Show new values (another time step, colour range) in the map that's there.

        Much faster than drawing the map again. Only possible when nothing but
        the values changed: same variable, projection and colormap, and the
        cells in the same places (they may move with time, e.g. WRF's XLAT(Time, ...)).
        """
        if self._ax is None or self._layout != self._layout_key():
            return False
        da, geo = self.state.da, self.state.geo
        grid = geo_grid(da, geo, self.state.indices, self._max_size)
        old = self._grid
        if grid.kind != old.kind or not (_same(grid.lat, old.lat) and _same(grid.lon, old.lon)):
            return False
        self._grid = grid
        self._locator = GridLocator(grid)
        self._limits = self.style_bar.limits_for(grid.values)
        drawn = with_cyclic_column(grid)
        if grid.kind == "points":
            # Stations with no value aren't in a scatter plot, and which ones those are
            # changes, so the points are drawn again (the rest of the map stays).
            self._artist.remove()
            self._artist = self._draw(self._ax, drawn, detail=False)
            self._colorbar.update_normal(self._artist)
        else:
            values = drawn.values
            if grid.kind == "curvilinear" and not grid.from_bounds:
                values = fill_missing_positions(grid.lat, grid.lon, values)[2]
            values = np.ma.masked_invalid(values)
            if isinstance(self._artist, AxesImage):
                self._artist.set_data(values)
            else:
                self._artist.set_array(values)
            self._artist.set_clim(*self._limits)
        self._title.set_text(self._title_text(grid))
        self._remove_detail()  # of the old values; a fresh one is loaded if zoomed in
        self._update_note()
        self._update_detail()
        self.mpl.canvas.draw_idle()
        return True

    def _draw(self, ax, grid, detail):
        return _outside_layout(self._draw_values(ax, grid, detail))

    def _draw_values(self, ax, grid, detail):
        vmin, vmax = self._limits
        cmap = self.style_bar.style().cmap
        values = np.ma.masked_invalid(grid.values)
        if grid.kind == "points":
            return ax.scatter(grid.lon, grid.lat, c=grid.values, cmap=cmap, vmin=vmin, vmax=vmax,
                              s=18, edgecolors="black", linewidths=0.3, transform=ccrs.PlateCarree(),
                              zorder=Z_POINTS + 0.1 * detail)
        zorder = Z_DETAIL if detail else Z_DATA
        if grid.from_bounds:  # cells as given by the file's bounds (CF 7.1)
            lat, lon = (grid.lat_edges, grid.lon_edges) if grid.kind == "regular" else grid.corners
            return ax.pcolormesh(lon, lat, values, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat",
                                 transform=ccrs.PlateCarree(), zorder=zorder)
        if not self._mesh and _uniform(grid.lat) and _uniform(grid.lon):
            # Plate Carrée coordinates are (shifted) degrees, so the image can go straight
            # into data coordinates. Cartopy would otherwise reproject it (needs scipy).
            x_edges, y_edges = cell_edges(grid.lon) - self._lon0, cell_edges(grid.lat)
            return ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower",
                             interpolation="nearest", transform=ax.transData, zorder=zorder,
                             extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]))
        lat, lon, shading = grid.lat, grid.lon, "auto"
        if grid.kind == "curvilinear":
            lat, lon, values = fill_missing_positions(lat, lon, values)
            if min(values.shape) >= 2:
                # The corners halfway between the centres (as shading="auto" would guess them,
                # but that guess doesn't allow for missing positions).
                lat, lon, shading = cell_corners(lat), cell_corners(lon), "flat"
        return ax.pcolormesh(lon, lat, np.ma.masked_invalid(values), cmap=cmap, vmin=vmin, vmax=vmax,
                             shading=shading, transform=ccrs.PlateCarree(), zorder=zorder)

    def _update_note(self):
        detail = self._detail.slice if self._detail is not None else None
        self.note.setText(resolution_note(self._grid.slice, detail))

    def _update_detail(self):
        if self._ax is None or self.state.da is None or not self.isVisible():
            return
        try:
            with wait_cursor():
                self._load_detail()
        except Exception:  # the overview is still there; don't lose it over the detail
            traceback.print_exc()

    def _load_detail(self):
        """Load the visible part in more detail if the zoom allows it."""
        ax, lon0 = self._ax, self._lon0
        visible = ax.get_extent(crs=ccrs.PlateCarree(central_longitude=lon0))
        if self._detail is not None and self._tracker.up_to_date(visible):
            return
        x0, x1, y0, y1 = padded(visible)
        box = (max(y0, -90), x0 + lon0, min(y1, 90), x1 + lon0)
        da, geo = self.state.da, self.state.geo
        window = geo_window(geo, self._grid, box)
        if window is None or not is_finer(self._grid.slice, window, self._max_size):
            self._remove_detail()
            return
        detail = geo_grid(da, geo, self.state.indices, self._max_size, window, like=self._grid)
        ax.set_autoscale_on(False)  # adding the detail mustn't move the view
        artist = self._draw(ax, detail, detail=True)
        self._remove_detail()
        self._detail, self._detail_artist = detail, artist
        self._detail_locator = GridLocator(detail)
        self._tracker.remember(visible, (x0, x1, y0, y1))
        self._update_note()
        self.mpl.canvas.draw_idle()

    def _remove_detail(self):
        if self._detail_artist is not None:
            self._detail_artist.remove()
            self._detail = self._detail_artist = self._detail_locator = None
            self._tracker.reset()
            self._update_note()
            self.mpl.canvas.draw_idle()

    @staticmethod
    def _set_extent(ax, name, grid, lon0):
        south, west, north, east = grid.bounds
        if name in POLAR:  # a cap around the pole, down to the data (at most 20°)
            if name.startswith("North"):
                extent = (-180, 180, min(max(south, 20.0), 80.0), 90)
            else:
                extent = (-180, 180, -90, max(min(north, -20.0), -80.0))
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            return
        # Orthographic shows the whole disk; very wide boxes degenerate in conic projections.
        if grid.is_global or name == "Orthographic" or east - west > 150:
            ax.set_global()
            return
        # In a CRS centred on the data, so regions crossing the dateline stay one box.
        west, east = west - lon0, east - lon0
        margin = max(0.5, 0.03 * max(east - west, north - south))
        if name == "Mercator":
            south, north = max(south, -79), min(north, 83)
        extent = (max(west - margin, -180), min(east + margin, 180),
                  max(south - margin, -90), min(north + margin, 90))
        ax.set_extent(extent, crs=ccrs.PlateCarree(central_longitude=lon0))


def _outside_layout(artist):
    """Leave a map artist out of the figure layout, which then needn't reproject it on every draw.

    It's inside the map axes anyway (unlike the gridline labels, which stay in).
    """
    artist.set_in_layout(False)
    wrapped = getattr(artist, "_wrapped_collection_fix", None)  # cells cut at the map's edge
    if wrapped is not None:
        wrapped.set_in_layout(False)
    return artist


def _same(a, b):
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))


def _uniform(c):
    d = np.diff(c)
    return len(d) == 0 or bool(np.allclose(d, d[0], rtol=1e-3, atol=0))
