"""Whether a newer version of the app has been released."""

import json
import re

VERSION_URL = "https://storage.rokuk.org/netseedf/latest/version.json"
DOWNLOAD_URL = "https://rokuk.org/projects/netseedf"

_VERSION = re.compile(r"v?(\d+(?:\.\d+)*)")


def parse_version(text) -> tuple[int, ...] | None:
    """'2.1.0' -> (2, 1, 0); anything after the numbers (e.g. '-beta') is ignored."""
    match = _VERSION.match(str(text).strip())
    if match is None:
        return None
    parts = [int(p) for p in match.group(1).split(".")]
    while len(parts) > 1 and parts[-1] == 0:  # 2.1 == 2.1.0
        parts.pop()
    return tuple(parts)


def latest_version(payload: bytes) -> str | None:
    """The version in version.json: {"version": "2.2.0"}, or just "2.2.0"."""
    try:
        data = json.loads(payload)
    except (ValueError, UnicodeDecodeError):
        return None
    if isinstance(data, dict):
        data = data.get("version")
    if not isinstance(data, str | int | float) or isinstance(data, bool):
        return None
    return str(data) if parse_version(data) is not None else None


def is_newer(latest, current) -> bool:
    new, old = parse_version(latest), parse_version(current)
    return new is not None and old is not None and new > old
