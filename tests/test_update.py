"""The start-up check for a newer version."""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import QSettings, QUrl  # noqa: E402
from PySide6.QtWidgets import QWidget  # noqa: E402

from netseedf.core.update import DOWNLOAD_URL, is_newer, latest_version, parse_version  # noqa: E402
from netseedf.ui import update_checker  # noqa: E402
from netseedf.ui.update_checker import IGNORED_KEY, UpdateChecker  # noqa: E402

BUILD = {"app_name": "NetSeeDF", "version": "2.1.0"}


class Checker(UpdateChecker):
    """With what the tests look at."""

    opened: list[str]  # URLs opened in the browser
    settings: QSettings
    version_file: Path  # what the checker reads, as if from the server


@pytest.mark.parametrize("text, expected", [
    ("2.1.0", (2, 1)),
    ("2.1", (2, 1)),
    ("v3", (3,)),
    ("2.10.1-beta", (2, 10, 1)),
    (" 1.0.0\n", (1,)),
    ("latest", None),
    ("", None),
])
def test_parse_version(text, expected):
    assert parse_version(text) == expected


@pytest.mark.parametrize("latest, current, newer", [
    ("2.2.0", "2.1.0", True),
    ("2.10.0", "2.9.0", True),  # numerically, not as text
    ("3", "2.9.9", True),
    ("2.1", "2.1.0", False),
    ("2.0.9", "2.1.0", False),
    ("nonsense", "2.1.0", False),
])
def test_is_newer(latest, current, newer):
    assert is_newer(latest, current) is newer


@pytest.mark.parametrize("payload, expected", [
    (b'{"version": "2.2.0"}', "2.2.0"),
    (b'{"version": "2.2.0", "date": "2026-10-01"}', "2.2.0"),
    (b'"2.2.0"', "2.2.0"),
    (b'{"version": 3}', "3"),
    (b'{"version": true}', None),
    (b'{"other": "2.2.0"}', None),
    (b'{"version": "soon"}', None),
    (b"<html>Not found</html>", None),
    (b"\xff\xfe", None),
])
def test_latest_version(payload, expected):
    assert latest_version(payload) == expected


@pytest.fixture
def checker(qtbot, tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(update_checker.QDesktopServices, "openUrl", lambda url: opened.append(url.toString()))
    window = QWidget()
    qtbot.addWidget(window)
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    version_file = tmp_path / "version.json"
    url = QUrl.fromLocalFile(str(version_file)).toString()
    c = Checker(window, settings, BUILD, "NetSeeDF/2.1.0", url)
    c.opened, c.settings, c.version_file = opened, settings, version_file
    return c


def _button(dialog, text):
    return next(b for b in dialog.buttons() if b.text().replace("&", "") == text)


def _check_and_wait(qtbot, checker, manual=False):
    """check(), and wait until the reply has been dealt with."""
    handled = []
    finished = checker._finished
    checker._finished = lambda reply, *args: (finished(reply, *args), handled.append(reply))
    checker.check(manual)
    qtbot.waitUntil(lambda: bool(handled), timeout=5000)


def test_newer_version_offers_download(qtbot, checker):
    checker.version_file.write_text('{"version": "2.2.0"}')
    checker.check()
    qtbot.waitUntil(lambda: checker.dialog is not None)
    assert "2.2.0" in checker.dialog.text() and "2.1.0" in checker.dialog.informativeText()
    _button(checker.dialog, "Download").click()
    assert checker.opened == [DOWNLOAD_URL]
    assert checker.settings.value(IGNORED_KEY) is None


def test_ignore_skips_that_version_only(qtbot, checker):
    checker.consider("2.2.0")
    _button(checker.dialog, "Ignore").click()
    assert checker.settings.value(IGNORED_KEY) == "2.2.0"
    assert checker.opened == []

    checker.dialog = None
    checker.consider("2.2.0")
    assert checker.dialog is None  # not asked again
    checker.consider("2.3.0")
    assert checker.dialog is not None  # but the next one is offered


@pytest.mark.parametrize("content", ['{"version": "2.1.0"}', '{"version": "2.0.0"}', "garbage"])
def test_nothing_shown_unless_newer(qtbot, checker, content):
    checker.version_file.write_text(content)
    _check_and_wait(qtbot, checker)
    assert checker.dialog is None


def test_nothing_shown_when_unreachable(qtbot, checker):
    # version_file was never written
    _check_and_wait(qtbot, checker)
    assert checker.dialog is None


@pytest.mark.parametrize("content, title", [
    ('{"version": "2.1.0"}', "No update available"),
    ("garbage", "Couldn't check for updates"),
    (None, "Couldn't check for updates"),  # unreachable
])
def test_checking_by_hand_always_answers(qtbot, checker, content, title):
    if content is not None:
        checker.version_file.write_text(content)
    _check_and_wait(qtbot, checker, manual=True)
    assert checker.dialog is not None and checker.dialog.windowTitle() == title


def test_checking_by_hand_offers_an_ignored_version(qtbot, checker):
    checker.settings.setValue(IGNORED_KEY, "2.2.0")
    checker.version_file.write_text('{"version": "2.2.0"}')
    _check_and_wait(qtbot, checker, manual=True)
    assert checker.dialog is not None and checker.dialog.windowTitle() == "Update available"
