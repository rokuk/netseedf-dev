# NetSeeDF

A simple application for viewing and visualizing data in NetCDF files.

![NetSeeDF stepping through time on a map](https://storage.rokuk.org/netseedf/foto/rec1.gif)

## Download

| Platform | Installer |
| --- | --- |
| Windows | [NetSeeDFSetup.exe](https://storage.rokuk.org/netseedf/latest/windows/NetSeeDFSetup.exe) |
| macOS (Apple Silicon) | [netseedf-macos-arm64.dmg](https://storage.rokuk.org/netseedf/latest/macos-arm64/netseedf-macos-arm64.dmg) |
| macOS (Intel) | [netseedf-macos-intel.dmg](https://storage.rokuk.org/netseedf/latest/macos-intel/netseedf-macos-intel.dmg) |
| Linux (.deb) | [netseedf-linux.deb](https://storage.rokuk.org/netseedf/latest/linux-deb/netseedf-linux.deb) |

The links always point to the latest version. More on the [project page](https://rokuk.org/projects/netseedf/).

## About

Inspired by Panoply, NetSeeDF is a simpler alternative that does not require Java.
Designed with students and researchers in mind, it lets you quickly explore the variables in a file,
their shapes, and visualize grid point values on a map. You can also export the time series of
a chosen grid cell.

## Features

- **Tree** of the file's groups, variables and coordinates, with attributes.
- **Table** of any 2D slice, with copy (Ctrl+C) and export to CSV.
- **Plot**: line plot or heatmap of any slice, with zoom, pan and PNG export.
- **Map** with coastlines, borders and a choice of projections. Works offline.
- **Interactive map** on OpenStreetMap, satellite or topographic tiles (needs internet), showing
  the value under the mouse and exporting the time series of a grid cell.
- **Sliders** to step through the remaining dimensions (time, depth, …).

Regular lat/lon grids, curvilinear grids with 2D lat/lon and station data (lat/lon along one
dimension) can all be mapped. NetSeeDF follows the [CF conventions](https://cfconventions.org/cf-conventions/cf-conventions.html).

## Screenshots

![Stepping through a dimension](https://storage.rokuk.org/netseedf/foto/rec2.gif)

![Table view](https://storage.rokuk.org/netseedf/foto/pic2.png)

![Map view](https://storage.rokuk.org/netseedf/foto/pic1.png)

## Feedback

Found a bug or have a suggestion? [Open an issue](https://github.com/rokuk/netseedf-dev/issues)
or write to [kontakt@rokuk.org](mailto:kontakt@rokuk.org).

NetSeeDF is developed by [Rok Kuk](https://rokuk.org) and licensed under the
[GNU GPL v3](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Development

Needs [uv](https://docs.astral.sh/uv/) and Python 3.13.

```sh
uv sync                                   # create .venv with all dependencies
uv run python scripts/fetch_resources.py  # download Natural Earth and Leaflet
uv run python tests/sample_data.py        # write sample files to samples/
uv run python src/main/python/main.py samples/regular_global.nc
uv run pytest
uv run pytest -m perf                     # performance budgets (writes a large sample file)
uv run ruff check src tests scripts
uv run pyright                            # type checking
```

The app runs from source without fbs: `netseedf/context.py` uses fbs Pro's
`ApplicationContext` when it's installed and a small stand-in otherwise.

### Building installers with fbs Pro

Build each installer on its own platform. On macOS, the build is for the Mac's own
architecture (Apple Silicon or Intel).

1. Install fbs Pro into the venv from the fbs file, e.g. `uv pip install <file>`.
2. `uv run fbs run` starts the app the way fbs will package it.
3. `uv run fbs freeze` builds `target/netseedf/` (Windows, Linux) or `target/netseedf.app` (macOS).
   Start it and open a file before building the installer.
4. `uv run fbs installer` builds `target/netseedfSetup.exe` (needs [NSIS](https://nsis.sourceforge.io)
   on `PATH`), `target/netseedf.dmg` or `target/netseedf.deb`.

Build settings are in `src/build/settings/*.json`. If a frozen build fails to start, `fbs freeze --debug`.

### Layout

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
(contains modified Copernicus Sentinel data 2016, CC BY 4.0).

## License

Code in this repository is licensed under the [GNU General Public License v3](https://www.gnu.org/licenses/gpl-3.0.en.html).
See [`LICENSE`](LICENSE) for details.
