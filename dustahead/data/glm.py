"""GOES Geostationary Lightning Mapper (GLM): flash density per grid cell per 10 minutes.

s3://noaa-goes16|19/GLM-L2-LCFA/YYYY/DDD/HH/  (GOES-East, same satellite as the imagery)  - one ~300 KB file every 20 s with flash lat/lon/energy
(verified Oct 2026). A 10-minute window spans 30 files; we read only flash_lat/flash_lon remotely
and cache the binned density (one 200 KB .npy per window) in data/cache/glm/. Lightning is the
earliest sign of the thunderstorm whose outflow becomes a haboob, often 30-90 min before dust reaches a road.
"""
from __future__ import annotations

import datetime as dt
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..grid import Grid
from ._cache import keyed_lock
from .goes import fetch_tmp, list_files, scan_start


def files_in_window(bucket: str | None, product: str, t_end: dt.datetime, window_min: int) -> list[str]:
    if t_end.tzinfo is None:
        t_end = t_end.replace(tzinfo=dt.timezone.utc)
    t_start = t_end - dt.timedelta(minutes=window_min)
    cands = list_files(bucket, product, t_start) + ([] if t_start.hour == t_end.hour else list_files(bucket, product, t_end))
    return [f for f in cands if t_start <= scan_start(f) < t_end]


def _flash_lonlat(remote: str) -> tuple[np.ndarray, np.ndarray]:
    import xarray as xr
    tmp = fetch_tmp(remote)
    try:
        with xr.open_dataset(tmp, engine="h5netcdf") as ds:
            if ds.sizes.get("number_of_flashes", 0) == 0:
                return np.empty(0), np.empty(0)
            return ds["flash_lon"].values.astype(float), ds["flash_lat"].values.astype(float)
    finally:
        tmp.unlink(missing_ok=True)


def flash_density(files: list[str], grid: Grid, workers: int = 16) -> np.ndarray:
    """Flashes per cell over the window, (ny, nx) float32."""
    if not files:
        return np.zeros(grid.shape, np.float32)
    with ThreadPoolExecutor(workers) as ex:
        parts = list(ex.map(_flash_lonlat, files))
    lon = np.concatenate([p[0] for p in parts])
    lat = np.concatenate([p[1] for p in parts])
    return grid.bin_points(lon, lat)


def flash_density_at(t_end: dt.datetime, grid: Grid, cfg: dict) -> np.ndarray:
    L = cfg["lightning"]
    cache = Path(cfg["paths"]["cache"]) / "glm" / f"{t_end:%Y%m%dT%H%M}_{L['window_min']}m.npy"
    with keyed_lock(cache):
        if cache.exists():
            return np.load(cache)
        files = files_in_window(L["bucket"], L["product"], t_end, L["window_min"])
        dens = flash_density(files, grid)
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(f".{os.getpid()}_{threading.get_ident()}.tmp.npy")
        np.save(tmp, dens)
        os.replace(tmp, cache)
    return dens
