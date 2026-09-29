# netseedf

Quick and simple viewer for NetCDF files, for Windows and macOS.

- **Tree** of the file's groups, variables and coordinates, with attributes.
- **Info**: `ncdump -h`-style header, how the variable was decoded, statistics on demand.
- **Table**: any 2D slice, read lazily while scrolling; copy (Ctrl+C) or export to CSV.
- **Plot**: line plot or heatmap of any slice, with zoom, pan and PNG export.
- **Map**: Cartopy map with coastlines and borders and a choice of projections. Works offline.
- **Web map**: Leaflet map on OpenStreetMap / satellite / topo tiles (needs internet), with the value under the mouse.

Sliders below the tabs step through the remaining dimensions (time, depth, …); ▶ animates.
Regular lat/lon grids, curvilinear grids with 2D lat/lon (including WRF output) and
station data (lat/lon along one dimension) can all be mapped.

## Development

Needs [uv](https://docs.astral.sh/uv/) and Python 3.13.

```sh
uv sync                                   # create .venv with all dependencies
uv run python tests/sample_data.py        # write sample files to samples/
uv run python src/main/python/main.py samples/regular_global.nc
uv run pytest
uv run ruff check src tests scripts
```

The app runs from source without fbs: `netseedf/context.py` uses fbs Pro's
`ApplicationContext` when it's installed and a small stand-in otherwise.

## Building installers with fbs Pro

fbs doesn't cross-compile: build the Windows installer on Windows and the macOS one on a Mac.

1. Install fbs Pro into the venv with from saved file or the link from your purchase email, e.g.
   `uv pip install <url>`. Keep it out of `pyproject.toml`: the URL contains your license key.
2. `uv run fbs run` starts the app the way fbs will package it.
3. `uv run fbs freeze` builds `target/netseedf/` (Windows) or `target/netseedf.app` (macOS).
   Start it and open a file before building the installer.
4. `uv run fbs installer` builds `target/netseedfSetup.exe` (needs [NSIS](https://nsis.sourceforge.io)
   on `PATH`) or `target/netseedf.dmg`.

Settings live in `src/build/settings/*.json`. `extra_pyinstaller_args` there already makes
PyInstaller bundle what xarray, Cartopy and pyproj need at run time. If a frozen build fails
to start, `fbs freeze --debug` shows the missing module; add it to `hidden_imports`.

Notes:

- The project targets Python 3.13 (`uv venv --python 3.13 && uv sync`).
- On macOS, the build is for the Mac's own architecture (Apple Silicon or Intel). fbs doesn't
  sign or notarize; do that with `codesign` / `notarytool` before distributing.
- `src/freeze/mac/Contents/Info.plist` registers the app for `.nc`, `.nc4`, `.cdf` and `.netcdf`
  files in Finder's "Open With".

## Layout

```
src/main/python/main.py            entry point (fbs main_module)
src/main/python/netseedf/core/     reading, slicing, lat/lon detection, rendering (no Qt)
src/main/python/netseedf/ui/       main window and the tabs
src/main/resources/base/           Leaflet page + library, Natural Earth shapefiles
src/main/icons/                    app icons (scripts/make_icons.py)
src/build/settings/                fbs settings
tests/                             pytest suite and the sample file generator
scripts/fetch_resources.py         re-download Natural Earth data and Leaflet
```

## Credits

Coastlines and borders from [Natural Earth](https://www.naturalearthdata.com) (public domain).
Web map by [Leaflet](https://leafletjs.com) (BSD-2-Clause); basemaps © OpenStreetMap
contributors, CARTO, Esri and OpenTopoMap. Licensed under the GNU GPL v3 (see `LICENSE`).
