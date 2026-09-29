import os
import sys

# Composite the window on the GPU from the start. Otherwise Qt has to recreate the
# native window when the web map's QWebEngineView first appears, so it closes and reopens.
os.environ.setdefault("QT_WIDGETS_RHI", "1")

# QtWebEngine must be imported before the QApplication is created.
from PySide6 import QtWebEngineWidgets  # noqa: E402, F401

from netseedf.context import AppContext  # noqa: E402

if __name__ == "__main__":
    sys.exit(AppContext().run())
