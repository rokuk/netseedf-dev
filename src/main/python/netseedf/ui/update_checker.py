"""Asks whether a newer version is out (at start-up, or from the Help menu), and offers to download it."""

from PySide6.QtCore import QObject, QSettings, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import QMessageBox, QWidget

from netseedf.core.update import DOWNLOAD_URL, VERSION_URL, is_newer, latest_version

TIMEOUT_MS = 10_000
IGNORED_KEY = "updates/ignored_version"


class UpdateChecker(QObject):
    """Reads version.json in the background and offers a newer version if there is one.

    "Ignore" skips that version: the next one is offered again. Checking by hand
    (manual=True) offers ignored versions too, and also says when there's no newer
    version or the check failed.
    """

    def __init__(self, parent: QWidget, settings: QSettings, build_settings, user_agent,
                 url=VERSION_URL):
        super().__init__(parent)
        self._window = parent
        self._settings = settings
        self._app_name = build_settings["app_name"]
        self._current = build_settings["version"]
        self._user_agent = user_agent
        self._url = url
        self._network = QNetworkAccessManager(self)
        self.dialog: QMessageBox | None = None

    def check(self, manual=False):
        request = QNetworkRequest(QUrl(self._url))
        request.setHeader(QNetworkRequest.KnownHeaders.UserAgentHeader, self._user_agent)
        request.setTransferTimeout(TIMEOUT_MS)
        request.setAttribute(QNetworkRequest.Attribute.CacheLoadControlAttribute,
                             QNetworkRequest.CacheLoadControl.AlwaysNetwork)
        reply = self._network.get(request)
        reply.finished.connect(lambda: self._finished(reply, manual))

    def _finished(self, reply: QNetworkReply, manual=False):
        reply.deleteLater()
        if reply.error() != QNetworkReply.NetworkError.NoError:
            # offline, or the server is down: at start-up, try again next time
            if manual:
                self._tell("Couldn't check for updates",
                           f"Couldn't reach the update server:\n{reply.errorString()}",
                           QMessageBox.Icon.Warning)
            return
        self.consider(latest_version(bytes(reply.readAll().data())), manual)

    def consider(self, latest: str | None, manual=False):
        """Offer `latest` if it's newer than this version and not ignored (unless asked by hand)."""
        if latest is None:
            if manual:
                self._tell("Couldn't check for updates", "The update server's answer couldn't be read.",
                           QMessageBox.Icon.Warning)
        elif not is_newer(latest, self._current):
            if manual:
                self._tell("No update available",
                           f"{self._app_name} {self._current} is the newest version.")
        elif manual or self._settings.value(IGNORED_KEY) != latest:
            self._offer(latest)

    def _tell(self, title, text, icon=QMessageBox.Icon.Information):
        box = QMessageBox(icon, title, text, QMessageBox.StandardButton.Ok, self._window)
        self._show(box)

    def _offer(self, latest):
        box = QMessageBox(self._window)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Update available")
        box.setText(f"{self._app_name} {latest} is available.")
        box.setInformativeText(f"You have version {self._current}.")
        download = box.addButton("&Download", QMessageBox.ButtonRole.AcceptRole)
        ignore = box.addButton("&Ignore", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(download)
        box.setEscapeButton(ignore)

        def clicked(button):
            if button is download:
                QDesktopServices.openUrl(QUrl(DOWNLOAD_URL))
            else:
                self._settings.setValue(IGNORED_KEY, latest)

        box.buttonClicked.connect(clicked)
        self._show(box)

    def _show(self, box: QMessageBox):
        box.finished.connect(box.deleteLater)
        self.dialog = box
        box.open()
