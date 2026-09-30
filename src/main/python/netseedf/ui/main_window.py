"""Main window: file tree on the left, views of the selected variable on the right."""

import traceback
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from netseedf.core.dataset import FILE_FILTER, OpenedFile, VariableRef, open_file
from netseedf.ui.base_view import DataView, wait_cursor
from netseedf.ui.cartopy_map_view import CartopyMapView
from netseedf.ui.dimension_panel import DimensionPanel
from netseedf.ui.info_view import InfoView
from netseedf.ui.plot_view import PlotView
from netseedf.ui.state import SelectionState
from netseedf.ui.table_view import TableView
from netseedf.ui.variable_tree import ROLE, VariableTree
from netseedf.ui.web_map_view import WebMapView

MAX_RECENT = 10


def user_agent(build_settings) -> str:
    """How the app identifies itself to tile servers, e.g.
    'NetSeeDF/0.1.0 (+https://example.org; contact: maps@example.org)'.
    """
    about = [f"+{build_settings['homepage']}"] if build_settings.get("homepage") else []
    if build_settings.get("contact"):
        about.append(f"contact: {build_settings['contact']}")
    app_id = f"{build_settings['app_name']}/{build_settings['version']}"
    return f"{app_id} ({'; '.join(about)})" if about else app_id


class MainWindow(QMainWindow):
    def __init__(self, resources, build_settings, settings: QSettings | None = None):
        super().__init__()
        self._build_settings = build_settings
        self._settings = settings if settings is not None else QSettings()
        self._files: dict[int, OpenedFile] = {}
        self._next_id = 1
        self._dirty: set[DataView] = set()
        self.state = SelectionState(self)

        self.tree = VariableTree()
        self.info = InfoView()
        self.table = TableView(self.state)
        self.plot = PlotView(self.state)
        self.map = CartopyMapView(self.state)
        self.web_map = WebMapView(self.state, resources, user_agent(build_settings))
        self.views: list[DataView] = [self.table, self.plot, self.map, self.web_map]
        self.tabs = QTabWidget(documentMode=True)
        self.tabs.addTab(self.info, self.info.title)
        for view in self.views:
            self.tabs.addTab(view, view.title)
            view.freeDimsChanged.connect(lambda v=view: self._free_dims_changed(v))
            view.status.connect(self._show_hover)
            if hasattr(view, "style_bar"):
                view.style_bar.cmapPicked.connect(lambda name, v=view: self._cmap_picked(v, name))
        self.dims = DimensionPanel(self.state)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.tabs, 1)
        right_layout.addWidget(self.dims)

        splitter = QSplitter()
        splitter.addWidget(self.tree)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 960])
        self.setCentralWidget(splitter)
        self.hover = QLabel()
        self.statusBar().addPermanentWidget(self.hover)

        self.tree.variableSelected.connect(self._select_variable)
        self.tree.fileSelected.connect(self._select_file)
        self.state.variableChanged.connect(self._variable_changed)
        self.state.indicesChanged.connect(self._indices_changed)
        self.tabs.currentChanged.connect(self._tab_changed)

        self._build_menus()
        self.setAcceptDrops(True)
        self._update_title()
        self.resize(1280, 800)
        geometry = self._settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

    # --- menus -------------------------------------------------------------

    def _build_menus(self):
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self._action("&Open…", self.open_dialog, QKeySequence.StandardKey.Open))
        self.recent_menu = file_menu.addMenu("Open &Recent")
        self.recent_menu.aboutToShow.connect(self._fill_recent_menu)
        file_menu.addAction(self._action("&Close File", self.close_current_file,
                                         QKeySequence.StandardKey.Close))
        file_menu.addSeparator()
        file_menu.addAction(self._action("&Quit", self.close, QKeySequence.StandardKey.Quit))
        view_menu = self.menuBar().addMenu("&View")
        for i in range(self.tabs.count()):
            view_menu.addAction(self._action(self.tabs.tabText(i),
                                             lambda i=i: self.tabs.setCurrentIndex(i),
                                             QKeySequence(f"Ctrl+{i + 1}")))
        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(self._action("&About", self._about))

    def _action(self, text, slot, shortcut=None):
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        return action

    def _fill_recent_menu(self):
        self.recent_menu.clear()
        recent = self._recent_files()
        for path in recent:
            self.recent_menu.addAction(self._action(path, lambda p=path: self.open_path(p)))
        if not recent:
            self.recent_menu.addAction("(none)").setEnabled(False)

    def _recent_files(self):
        value = self._settings.value("recent_files", [])
        return [value] if isinstance(value, str) else list(value or [])

    def _remember(self, path):
        recent = [p for p in self._recent_files() if p != path]
        self._settings.setValue("recent_files", [path, *recent][:MAX_RECENT])

    def _about(self):
        s = self._build_settings
        homepage = s.get("homepage", "")
        QMessageBox.about(self, f"About {s['app_name']}", f"""
            <h3>{s['app_name']} {s['version']}</h3>
            <p>Quick and simple viewer for NetCDF files.</p>
            <p><a href="{homepage}">{homepage}</a></p>
            <p>License: GNU General Public License v3.0.<br>
            Coastlines and borders: Natural Earth (public domain).<br>
            Interactive map: Leaflet; basemaps &copy; OpenStreetMap contributors, Esri,
            OpenTopoMap.</p>""")

    # --- files ------------------------------------------------------------

    def open_dialog(self):
        start = str(Path(self._recent_files()[0]).parent) if self._recent_files() else ""
        paths, _ = QFileDialog.getOpenFileNames(self, "Open NetCDF file", start, FILE_FILTER)
        for path in paths:
            self.open_path(path)

    def open_path(self, path):
        path = str(Path(path).resolve())
        for file_id, opened in self._files.items():
            if str(opened.path.resolve()) == path:
                self.tree.setCurrentItem(self.tree.file_item(file_id))
                return
        try:
            with wait_cursor():
                opened = open_file(path)
        except Exception as e:
            traceback.print_exc()
            QMessageBox.warning(self, "Can't open file", f"{path}\n\n{type(e).__name__}: {e}")
            return
        file_id, self._next_id = self._next_id, self._next_id + 1
        self._files[file_id] = opened
        self._remember(path)
        self.tree.add_file(file_id, opened)
        if opened.warnings:
            self.statusBar().showMessage(f"{opened.name}: " + " ".join(opened.warnings), 15000)
        self._select_default(file_id)

    def _select_default(self, file_id):
        """Select the data variable with the most dimensions (usually the interesting one)."""
        items = [i for i in self.tree.variable_items(file_id)
                 if i.parent().data(0, ROLE).kind != "folder"]
        if not items:
            self.tree.setCurrentItem(self.tree.file_item(file_id))
            return
        opened = self._files[file_id]

        def ndim(item):
            node = item.data(0, ROLE)
            return opened.variable(node.group, node.name).ndim
        self.tree.setCurrentItem(max(items, key=ndim))

    def close_current_file(self):
        file_id = self.tree.current_file_id()
        if file_id is None:
            return
        if self.state.ref is not None and self.state.ref.file_id == file_id:
            self.state.clear()
        self.tree.remove_file(file_id)
        self._files.pop(file_id).close()
        if not self._files:
            self.info.show_text("")
        self._update_title()

    # --- selection ----------------------------------------------------------

    def _select_variable(self, ref: VariableRef):
        opened = self._files[ref.file_id]
        self.state.set_variable(opened, ref)
        da = self.state.da
        try:
            self.info.show_variable(opened, ref.group, da)
        except Exception as e:
            traceback.print_exc()
            self.info.show_text(f"Couldn't read the header of {ref.name}: {e}")
        self._update_title()

    def _select_file(self, file_id, group):
        opened = self._files[file_id]
        try:
            self.info.show_file(opened, group)
        except Exception as e:
            traceback.print_exc()
            self.info.show_text(f"Couldn't read the header of {opened.path}: {e}")
        self._update_title(opened)

    def _update_title(self, opened=None):
        name = self._build_settings["app_name"]
        opened = opened or self.state.file
        if opened is None:
            self.setWindowTitle(name)
        elif self.state.ref is not None and self.state.file is opened:
            self.setWindowTitle(f"{self.state.ref.name} — {opened.name} — {name}")
        else:
            self.setWindowTitle(f"{opened.name} — {name}")

    # --- keeping views up to date ------------------------------------------------

    def current_view(self) -> DataView | None:
        widget = self.tabs.currentWidget()
        return widget if isinstance(widget, DataView) else None

    def _variable_changed(self):
        for view in self.views:
            view.variable_changed()
        self._dirty = set(self.views)
        self._show_current(rebuild_dims=True)

    def _indices_changed(self):
        self._dirty = set(self.views)
        self._show_current(rebuild_dims=False)

    def _tab_changed(self):
        self._show_current(rebuild_dims=True)

    def _free_dims_changed(self, view):
        if view is self.current_view():
            self._dirty.add(view)
            self._show_current(rebuild_dims=True)

    def _cmap_picked(self, source, name):
        """Use the same colormap in every view; the others redraw when next shown."""
        for view in self.views:
            if view is not source and hasattr(view, "style_bar"):
                view.style_bar.set_cmap(name)
                self._dirty.add(view)

    def _show_current(self, rebuild_dims):
        view = self.current_view()
        self.hover.clear()
        if view is None:  # Info tab
            self.dims.show_dims(self.state.da.dims if self.state.da is not None else ())
            return
        if rebuild_dims:
            self.dims.show_dims(view.displayed_dims())
        if view in self._dirty:
            self._dirty.discard(view)
            view.update_view()

    def _show_hover(self, text):
        self.hover.setText(text)

    # --- drag and drop, closing -------------------------------------------------

    def dragEnterEvent(self, event):  # noqa: N802 (Qt override)
        if any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):  # noqa: N802 (Qt override)
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self.open_path(url.toLocalFile())

    def closeEvent(self, event):  # noqa: N802 (Qt override)
        self._settings.setValue("window/geometry", self.saveGeometry())
        for opened in self._files.values():
            opened.close()
        super().closeEvent(event)
