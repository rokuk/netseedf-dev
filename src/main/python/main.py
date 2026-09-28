import sys

# QtWebEngine must be imported before the QApplication is created.
from PySide6 import QtWebEngineWidgets  # noqa: F401

from netseedf.context import AppContext

if __name__ == "__main__":
    sys.exit(AppContext().run())
