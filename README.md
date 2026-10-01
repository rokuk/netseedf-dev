# NetSeeDF

Quick and simple viewer for NetCDF files. Available for Windows and macOS.

Inspired by Panoply, NetSeeDF is designed to be simpler tool that does not require a Java installation. Designed with students and researchers in mind, it lets you quickly explore available variables, their shapes, and visualize grid point values on a map.

- **Tree** of the file's groups, variables and coordinates, with attributes.
- **Table**: any 2D slice, copy (Ctrl+C) or export to CSV.
- **Plot**: line plot or heatmap of any slice, with zoom, pan and PNG export.
- **Map**: Cartopy map with coastlines and borders and a choice of projections. Works offline.
- **Interactive map**: Leaflet map on OpenStreetMap / satellite / topo tiles (needs internet), with the value under the mouse.

Sliders below the tabs step through the remaining dimensions (time, depth, …).

Regular lat/lon grids, curvilinear grids with 2D lat/lon and
station data (lat/lon along one dimension) can all be mapped. NetSeeDF complies with the [CF conventions](https://cfconventions.org/cf-conventions/cf-conventions.html).

NetSeeDF is developed by [Rok Kuk](https://rokuk.org) and licensed under the [GNU GPL v3](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Development

Needs [uv](https://docs.astral.sh/uv/) and Python 3.13.

```sh
uv sync                                   # create .venv with all dependencies
uv run python scripts/fetch_resources.py  # download Natural Earth and Leaflet
uv run python tests/sample_data.py        # write sample files to samples/
uv run python src/main/python/main.py samples/regular_global.nc
uv run pytest
uv run ruff check src tests scripts
```

The app runs from source without fbs: `netseedf/context.py` uses fbs Pro's
`ApplicationContext` when it's installed and a small stand-in otherwise.

## Building installers with fbs Pro

Build the Windows installer on Windows and the macOS one on a Mac. 
On macOS, the build is for the Mac's own architecture (Apple Silicon or Intel).

1. Install fbs Pro into the venv with from the fbs file, e.g.
   `uv pip install <file>`.
2. `uv run fbs run` starts the app the way fbs will package it.
3. `uv run fbs freeze` builds `target/netseedf/` (Windows) or `target/netseedf.app` (macOS).
   Start it and open a file before building the installer.
4. `uv run fbs installer` builds `target/netseedfSetup.exe` (needs [NSIS](https://nsis.sourceforge.io)
   on `PATH`) or `target/netseedf.dmg`.

Build settings are in `src/build/settings/*.json`. If a frozen build fails to start, `fbs freeze --debug`.

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
Interactive map by [Leaflet](https://leafletjs.com) (BSD-2-Clause); basemaps © OpenStreetMap
contributors and OpenTopoMap; [Sentinel-2 cloudless](https://s2maps.eu) by EOX IT Services GmbH
(contains modified Copernicus Sentinel data 2016, CC BY 4.0). Licensed under the GNU GPL v3 (see `LICENSE`).

## License

Code in this repository is licensed under the [GNU General Public License v3](https://www.gnu.org/licenses/gpl-3.0.en.html). See `LICENSE` for details.