import os
import sys

if sys.platform == "linux" and getattr(sys, "frozen", False):
    # On Ubuntu, QtWebEngine's GPU rendering can fail to start ("did not find extension
    # DRI_Mesa", "EGL: Failed to initialize GBM device") and the web map stays blank.
    # Render in software instead.
    os.environ.setdefault("QT_WIDGETS_RHI", "0")
    os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")
else:
    # Composite the window on the GPU from the start. Otherwise Qt has to recreate the
    # native window when the web map's QWebEngineView first appears, so it closes and reopens.
    os.environ.setdefault("QT_WIDGETS_RHI", "1")

# QtWebEngine must be imported before the QApplication is created.
from PySide6 import QtWebEngineWidgets  # noqa: E402, F401

from netseedf.context import AppContext  # noqa: E402

if __name__ == "__main__":
    sys.exit(AppContext().run())
