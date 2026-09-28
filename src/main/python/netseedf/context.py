"""Application start-up and wiring.

Uses fbs Pro's ApplicationContext when fbs is installed. Without it (e.g. in
CI or before fbs Pro is set up) a small stand-in with the same interface
runs the app straight from the source tree.
"""

import json
import sys
from functools import cached_property
from pathlib import Path

from PySide6.QtCore import QEvent, QStandardPaths, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

try:
    from fbs_runtime import PUBLIC_SETTINGS
    from fbs_runtime.application_context.PySide6 import ApplicationContext
except ImportError:
    ApplicationContext = PUBLIC_SETTINGS = None

# The settings the app reads (all listed in "public_settings" in base.json).
SETTING_KEYS = ("app_name", "author", "version", "homepage")

PROJECT_DIR = Path(__file__).resolve().parents[4]  # only meaningful when running from source


class NetseedfApplication(QApplication):
    """Reports files opened from Finder ("Open With", double-click) on macOS."""

    fileOpenRequested = Signal(str)

    def event(self, e):
        if e.type() == QEvent.Type.FileOpen:
            self.fileOpenRequested.emit(e.file())
            return True
        return super().event(e)


def _platform():
    if sys.platform == "win32":
        return "windows"
    return "mac" if sys.platform == "darwin" else "linux"


class SourceContext:
    """Stand-in for fbs's ApplicationContext when running from source without fbs."""

    def __init__(self):
        self.app  # noqa: B018 (Qt objects need the QApplication to exist first)
        if self.app_icon is not None:
            self.app.setWindowIcon(self.app_icon)

    @cached_property
    def app(self):
        return QApplication(sys.argv)

    @cached_property
    def build_settings(self):
        settings = {}
        for profile in ("base", "secret", _platform()):
            path = PROJECT_DIR / "src" / "build" / "settings" / f"{profile}.json"
            if path.exists():
                settings.update(json.loads(path.read_text(encoding="utf-8")))
        return settings

    def get_resource(self, *rel_path):
        dirs = [PROJECT_DIR / "src" / "main" / "icons",
                PROJECT_DIR / "src" / "main" / "resources" / _platform(),
                PROJECT_DIR / "src" / "main" / "resources" / "base"]
        for d in dirs:
            path = d.joinpath(*rel_path)
            if path.exists():
                return str(path)
        raise FileNotFoundError(f"Could not locate resource {'/'.join(rel_path)}")

    @cached_property
    def app_icon(self):
        if sys.platform == "darwin":
            return None  # macOS takes the icon from the app bundle
        try:
            return QIcon(self.get_resource("Icon.ico"))
        except FileNotFoundError:
            return None


class AppContext(ApplicationContext or SourceContext):
    if PUBLIC_SETTINGS is not None:
        @cached_property
        def build_settings(self):
            # fbs Pro replaced ApplicationContext.build_settings with PUBLIC_SETTINGS,
            # which only supports [key] lookups.
            return {key: PUBLIC_SETTINGS[key] for key in SETTING_KEYS}

    @cached_property
    def app(self):
        app = NetseedfApplication(sys.argv)
        app.setApplicationName(self.build_settings["app_name"])
        app.setApplicationVersion(self.build_settings["version"])
        app.setOrganizationName(self.build_settings["app_name"])
        return app

    @cached_property
    def main_window(self):
        from netseedf.ui.main_window import MainWindow

        return MainWindow(self.get_resource, self.build_settings)

    def run(self):
        from netseedf.ui.cartopy_map_view import configure_offline_data

        cache = Path(QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppDataLocation)) / "cartopy"
        configure_offline_data(self.get_resource("cartopy"), cache)
        window = self.main_window
        self.app.fileOpenRequested.connect(window.open_path)
        window.show()
        for arg in sys.argv[1:]:
            if Path(arg).is_file():
                window.open_path(arg)
        return self.app.exec()
