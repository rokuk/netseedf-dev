"""Small synthetic NetCDF files covering the layouts the viewer has to handle.

Used by the tests; run ``python tests/sample_data.py [outdir]`` to write them
for trying the app by hand.
"""

import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import pyproj
import xarray as xr

RNG = np.random.default_rng(42)


def _field(lat, lon, t=0.0):
    """Smooth, plausible-looking field on a lat/lon grid (broadcasts)."""
    lat_r, lon_r = np.deg2rad(lat), np.deg2rad(lon)
    return (
        288.0
        - 30 * np.sin(lat_r) ** 2
        + 5 * np.cos(2 * lon_r + t) * np.cos(lat_r)
        + 2 * np.sin(3 * lat_r + t)
    )


def regular_global(path):
    """(time, lat, lon) global grid, lon 0..360, lat descending, NaN "land"."""
    time = pd.date_range("2024-01-01", periods=4, freq="D")
    lat = np.arange(89, -90, -2.0)
    lon = np.arange(0, 360, 2.0)
    data = np.stack(
        [_field(lat[:, None], lon[None, :], t) for t in range(len(time))]
    ).astype("float32")
    data[:, 40:50, 20:40] = np.nan
    ds = xr.Dataset(
        {"sst": (("time", "lat", "lon"), data, {"units": "K", "long_name": "Sea surface temperature"})},
        coords={
            "time": time,
            "lat": ("lat", lat, {"units": "degrees_north", "standard_name": "latitude"}),
            "lon": ("lon", lon, {"units": "degrees_east", "standard_name": "longitude"}),
        },
        attrs={"title": "Synthetic global SST", "Conventions": "CF-1.8"},
    )
    ds.to_netcdf(path, engine="netcdf4")


def curvilinear(path):
    """(time, y, x) grid with 2D auxiliary lat/lon coordinates (rotated box over Europe)."""
    ny, nx = 40, 50
    j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    angle = np.deg2rad(20)
    lat = 40 + 0.5 * (j * np.cos(angle) + i * np.sin(angle) * 0.3)
    lon = -5 + 0.6 * (i * np.cos(angle) - j * np.sin(angle) * 0.3)
    data = np.stack([_field(lat, lon, t) for t in range(3)]).astype("float32")
    ds = xr.Dataset(
        {"temp": (("time", "y", "x"), data, {"units": "K", "coordinates": "lat lon"})},
        coords={"time": pd.date_range("2020-06-01", periods=3, freq="6h")},
    )
    ds["lat"] = (("y", "x"), lat, {"units": "degrees_north"})
    ds["lon"] = (("y", "x"), lon, {"units": "degrees_east"})
    ds.to_netcdf(path, engine="netcdf4")


def timeseries(path):
    time = pd.date_range("2023-01-01", periods=365, freq="D")
    q = 50 + 30 * np.sin(np.arange(365) / 58.0) + RNG.normal(0, 3, 365)
    ds = xr.Dataset(
        {"discharge": ("time", q.astype("float32"), {"units": "m3 s-1", "long_name": "River discharge"})},
        coords={"time": time},
    )
    ds.to_netcdf(path, engine="netcdf4")


def four_d(path):
    """(time, depth, lat, lon) regional ocean box."""
    time = pd.date_range("2022-01-01", periods=3, freq="MS")
    depth = np.array([0.5, 5, 10, 50, 100])
    lat = np.linspace(35, 65, 30)
    lon = np.linspace(-20, 30, 40)
    base = 35 + 0.02 * lat[:, None] + 0.01 * lon[None, :]
    data = np.stack(
        [np.stack([base + 0.01 * d + 0.1 * t for d in depth]) for t in range(len(time))]
    ).astype("float32")
    ds = xr.Dataset(
        {"salinity": (("time", "depth", "lat", "lon"), data, {"units": "psu"})},
        coords={
            "time": time,
            "depth": ("depth", depth, {"units": "m", "positive": "down"}),
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "lon": ("lon", lon, {"units": "degrees_east"}),
        },
    )
    ds.to_netcdf(path, engine="netcdf4")


def groups(path):
    """NetCDF4 file with nested groups; the time coordinate lives in the root group."""
    with netCDF4.Dataset(path, "w") as nc:
        nc.title = "Grouped file"
        nc.createDimension("time", 5)
        t = nc.createVariable("time", "f8", ("time",))
        t.units = "hours since 2021-03-01 00:00"
        t[:] = np.arange(5)
        obs = nc.createGroup("observations")
        obs.instrument = "thermometer"
        v = obs.createVariable("air_temperature", "f4", ("time",))
        v.units = "degC"
        v[:] = [1.5, 2.0, 2.5, 1.0, 0.5]
        sub = obs.createGroup("station_b")
        w = sub.createVariable("wind_speed", "f4", ("time",))
        w.units = "m s-1"
        w[:] = [3, 4, 5, 4, 3]


def netcdf3(path):
    lat = np.linspace(-60, 60, 25)
    lon = np.linspace(-180, 175, 72)
    data = np.stack([_field(lat[:, None], lon[None, :], t) for t in range(2)]) - 280
    ds = xr.Dataset(
        {"pr": (("time", "lat", "lon"), data.astype("float32"), {"units": "mm/day"})},
        coords={"time": [0.0, 1.0], "lat": lat, "lon": lon},
    )
    ds["time"].attrs["units"] = "days since 2010-01-01"
    ds["lat"].attrs["units"] = "degrees_north"
    ds["lon"].attrs["units"] = "degrees_east"
    ds.to_netcdf(path, format="NETCDF3_CLASSIC", engine="netcdf4")


def packed(path):
    """int16 on disk with scale_factor / add_offset / _FillValue."""
    lat = np.linspace(-45, 45, 19)
    lon = np.linspace(0, 90, 31)
    real = _field(lat[:, None], lon[None, :])
    fill = -32767
    raw = np.round((real - 280) / 0.01).astype("int16")
    raw[0, 0] = fill
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("lat", len(lat))
        nc.createDimension("lon", len(lon))
        nc.createVariable("lat", "f4", ("lat",))[:] = lat
        nc["lat"].units = "degrees_north"
        nc.createVariable("lon", "f4", ("lon",))[:] = lon
        nc["lon"].units = "degrees_east"
        v = nc.createVariable("t2m", "i2", ("lat", "lon"), fill_value=fill)
        v.set_auto_maskandscale(False)
        v.scale_factor = 0.01
        v.add_offset = 280.0
        v.units = "K"
        v[:] = raw


def noleap(path):
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("time", 12)
        t = nc.createVariable("time", "f8", ("time",))
        t.units = "days since 1850-01-01"
        t.calendar = "noleap"
        t[:] = np.arange(12) * 30 + 15
        v = nc.createVariable("tas", "f4", ("time",))
        v.units = "K"
        v[:] = 280 + 10 * np.sin(np.arange(12) / 12 * 2 * np.pi)


def bad_time(path):
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("time", 3)
        t = nc.createVariable("time", "f8", ("time",))
        t.units = "days since yesterday"
        t[:] = [0, 1, 2]
        v = nc.createVariable("x", "f4", ("time",))
        v[:] = [1, 2, 3]


def points(path):
    """Station data: lat/lon are 1D coordinates along the same dimension."""
    n = 50
    lat = RNG.uniform(35, 60, n)
    lon = RNG.uniform(-10, 30, n)
    time = pd.date_range("2024-05-01", periods=6, freq="D")
    ds = xr.Dataset(
        {
            "elevation": ("station", RNG.uniform(0, 2500, n).astype("float32"), {"units": "m"}),
            "precip": (("time", "station"), RNG.gamma(1.0, 2.0, (6, n)).astype("float32"),
                       {"units": "mm"}),
            "station_name": ("station", np.array([f"Station {i:02d}" for i in range(n)], dtype=object)),
        },
        coords={
            "time": time,
            "lat": ("station", lat, {"units": "degrees_north"}),
            "lon": ("station", lon, {"units": "degrees_east"}),
        },
    )
    ds.to_netcdf(path, engine="netcdf4")


def pacific(path):
    """Regional grid crossing the dateline, stored as 150..210 degrees east."""
    lat = np.linspace(-20, 20, 21)
    lon = np.linspace(150, 210, 31)
    ds = xr.Dataset(
        {"sla": (("lat", "lon"), _field(lat[:, None], lon[None, :]) - 285, {"units": "m"})},
        coords={
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "lon": ("lon", lon, {"units": "degrees_east"}),
        },
    )
    ds.to_netcdf(path, engine="netcdf4")


def wrf_like(path):
    """WRF-style output: XLAT/XLONG carry a Time dimension and aren't dimension coords."""
    nt, ny, nx = 2, 20, 30
    j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    xlat = np.broadcast_to(44 + 0.1 * j + 0.01 * i, (nt, ny, nx)).astype("float32")
    xlong = np.broadcast_to(12 + 0.15 * i - 0.01 * j, (nt, ny, nx)).astype("float32")
    t2 = np.stack([_field(xlat[k], xlong[k], k) for k in range(nt)]).astype("float32")
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("Time", None)
        nc.createDimension("south_north", ny)
        nc.createDimension("west_east", nx)
        for name, arr, units in (("XLAT", xlat, "degree_north"), ("XLONG", xlong, "degree_east")):
            v = nc.createVariable(name, "f4", ("Time", "south_north", "west_east"))
            v.units = units
            v[:] = arr
        v = nc.createVariable("T2", "f4", ("Time", "south_north", "west_east"))
        v.units = "K"
        v.description = "TEMP at 2 M"
        v.coordinates = "XLONG XLAT"
        v[:] = t2


def bounded(path):
    """Regular grid whose cell bounds aren't halfway between the centres (CF 7.1)."""
    lat_b = np.array([30, 34, 40, 42, 50, 60], dtype=float)
    lon_b = np.array([-10, -4, 0, 10, 12, 20, 30], dtype=float)
    lat = (lat_b[:-1] + lat_b[1:]) / 2
    lon = lon_b[:-1] + 0.25 * np.diff(lon_b)  # centres off-centre in their cells
    with netCDF4.Dataset(path, "w") as nc:
        nc.Conventions = "CF-1.12"
        nc.createDimension("nv", 2)
        for name, centres, edges, units in (("lat", lat, lat_b, "degrees_north"),
                                            ("lon", lon, lon_b, "degrees_east")):
            nc.createDimension(name, len(centres))
            v = nc.createVariable(name, "f8", (name,))
            v.units, v.bounds = units, f"{name}_bnds"
            v[:] = centres
            bnds = nc.createVariable(f"{name}_bnds", "f8", (name, "nv"))
            bnds[:] = np.column_stack([edges[:-1], edges[1:]])
        v = nc.createVariable("tas", "f4", ("lat", "lon"))
        v.units = "K"
        v[:] = _field(lat[:, None], lon[None, :])


def trajectories(path):
    """CF 9 trajectories of different lengths, padded with missing values."""
    n_traj, n_obs = 3, 20
    lengths = [20, 12, 7]
    with netCDF4.Dataset(path, "w") as nc:
        nc.Conventions = "CF-1.12"
        nc.featureType = "trajectory"
        nc.createDimension("trajectory", n_traj)
        nc.createDimension("obs", n_obs)
        tid = nc.createVariable("trajectory", "i4", ("trajectory",))
        tid.cf_role = "trajectory_id"
        tid[:] = np.arange(n_traj)
        arrays = {name: np.full((n_traj, n_obs), -999.0) for name in ("time", "lat", "lon", "temp")}
        for k, n in enumerate(lengths):
            s = np.arange(n)
            arrays["time"][k, :n] = s
            arrays["lat"][k, :n] = 40 + 3 * k + 0.2 * s
            arrays["lon"][k, :n] = -5 + 4 * k + 0.3 * s + np.sin(s / 3)
            arrays["temp"][k, :n] = 15 + RNG.normal(0, 1, n)
        attrs = {"time": {"units": "hours since 2024-07-01", "standard_name": "time"},
                 "lat": {"units": "degrees_north", "standard_name": "latitude"},
                 "lon": {"units": "degrees_east", "standard_name": "longitude"},
                 "temp": {"units": "degC", "coordinates": "time lat lon"}}
        for name, values in arrays.items():
            v = nc.createVariable(name, "f4" if name == "temp" else "f8", ("trajectory", "obs"),
                                  fill_value=-999.0)
            v.setncatts(attrs[name])
            v[:] = values


def polar_projected(path):
    """North polar Lambert azimuthal grid (like EASE2), x/y in km, with 2D lat/lon and a
    grid mapping (CF 5.6). Its longitudes go all the way round the pole."""
    step = 200.0  # km
    x = np.arange(-30, 30) * step + step / 2  # the pole at the corner of the middle four cells
    y = x[::-1]  # north to south, as such grids usually are
    mapping = {"grid_mapping_name": "lambert_azimuthal_equal_area", "longitude_of_projection_origin": 0.0,
               "latitude_of_projection_origin": 90.0, "false_easting": 0.0, "false_northing": 0.0,
               "semi_major_axis": 6378137.0, "inverse_flattening": 298.257223563}
    to_lonlat = pyproj.Transformer.from_crs(pyproj.CRS.from_cf(mapping), "EPSG:4326", always_xy=True)
    lon, lat = to_lonlat.transform(*np.meshgrid(x * 1000, y * 1000))
    data = np.stack([_field(lat, lon, t) for t in range(2)]).astype("float32")
    data[:, 5:15, 40:55] = np.nan  # "land"
    with netCDF4.Dataset(path, "w") as nc:
        nc.Conventions = "CF-1.12"
        for name in ("time", "y", "x"):
            nc.createDimension(name, {"time": None, "y": y.size, "x": x.size}[name])
        crs = nc.createVariable("crs", "i4")
        crs.setncatts(mapping)
        t = nc.createVariable("time", "f8", ("time",))
        t.units, t.standard_name = "days since 2020-09-01", "time"
        t[:] = [0, 1]
        for name, values, axis in (("x", x, "x"), ("y", y, "y")):
            v = nc.createVariable(name, "f8", (name,))
            v.units, v.standard_name = "km", f"projection_{axis}_coordinate"
            v[:] = values
        for name, values, units in (("lat", lat, "degrees_north"), ("lon", lon, "degrees_east")):
            v = nc.createVariable(name, "f4", ("y", "x"))
            v.units, v.standard_name = units, {"lat": "latitude", "lon": "longitude"}[name]
            v[:] = values
        v = nc.createVariable("ice", "f4", ("time", "y", "x"), fill_value=np.float32(np.nan))
        v.units, v.grid_mapping, v.coordinates = "K", "crs", "time lat lon"
        v[:] = data


WRITERS = {
    "regular_global.nc": regular_global,
    "curvilinear.nc": curvilinear,
    "timeseries.nc": timeseries,
    "four_d.nc": four_d,
    "groups.nc": groups,
    "netcdf3.nc": netcdf3,
    "packed.nc": packed,
    "noleap.nc": noleap,
    "bad_time.nc": bad_time,
    "points.nc": points,
    "pacific.nc": pacific,
    "wrf_like.nc": wrf_like,
    "bounded.nc": bounded,
    "trajectories.nc": trajectories,
    "polar_projected.nc": polar_projected,
}


def write_all(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, writer in WRITERS.items():
        path = directory / name
        writer(path)
        paths[name] = path
    return paths


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "samples")
    for p in write_all(out).values():
        print(p)
