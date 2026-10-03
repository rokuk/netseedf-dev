"""Download the third-party files bundled with the app into src/main/resources.

    python scripts/fetch_resources.py

- Natural Earth coastlines and borders (public domain) so the Cartopy map
  works offline.
- Leaflet (BSD-2-Clause) for the web map.

The results are committed, so this only needs re-running to change versions.
"""

import shutil
import tempfile
import urllib.request
from pathlib import Path

import cartopy
import cartopy.io.shapereader as shpreader

RESOURCES = Path(__file__).resolve().parents[1] / "src" / "main" / "resources" / "base"

NATURAL_EARTH = [
    ("physical", "coastline"),
    ("cultural", "admin_0_boundary_lines_land"),
]
SCALES = ["50m"]

LEAFLET_VERSION = "1.9.4"
LEAFLET_FILES = [
    "leaflet.js",
    "leaflet.css",
    "images/layers.png",
    "images/layers-2x.png",
    "images/marker-icon.png",
    "images/marker-icon-2x.png",
    "images/marker-shadow.png",
]


def fetch_natural_earth():
    target = RESOURCES / "cartopy_data"
    with tempfile.TemporaryDirectory() as tmp:
        cartopy.config["data_dir"] = tmp
        cartopy.config["pre_existing_data_dir"] = tmp
        for category, name in NATURAL_EARTH:
            for scale in SCALES:
                shp = Path(shpreader.natural_earth(resolution=scale, category=category, name=name))
                dest = target / shp.parent.relative_to(tmp)
                dest.mkdir(parents=True, exist_ok=True)
                for part in shp.parent.glob(shp.stem + ".*"):
                    shutil.copy2(part, dest / part.name)
                    print(dest / part.name)


def fetch_leaflet():
    target = RESOURCES / "web" / "leaflet"
    base = f"https://cdn.jsdelivr.net/npm/leaflet@{LEAFLET_VERSION}/"
    for rel in [*(f"dist/{f}" for f in LEAFLET_FILES), "LICENSE"]:
        dest = target / rel.removeprefix("dist/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(base + rel) as response:
            dest.write_bytes(response.read())
        print(dest)


if __name__ == "__main__":
    fetch_natural_earth()
    fetch_leaflet()
