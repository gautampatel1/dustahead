"""NEXRAD composite reflectivity from the Iowa Environmental Mesonet (IEM) mosaics, as an INPUT to the sky branch.

Why radar: a haboob is the dust front of a thunderstorm outflow, and the outflow boundary shows on radar as a thin
line of weak echo ("fine line") minutes to an hour before any dust is lifted - the precursor NWS Phoenix forecasters
watch. No satellite channel shows it directly. The IEM N0Q mosaic (NEXRAD base reflectivity composite, 0.005 deg,
every 5 min, archived since Nov 2010, EPSG:4326) covers every year of the data set.

Access: the IEM WMS-T service, cropped to the domain server-side (about 0.6 MB per frame instead of the 3 MB CONUS
PNG; verified 4 Nov 2026: 4-km cell maxima correlate 0.994 with the archive PNG crop). The styled RGB is inverted
back to palette indices with the archive palette (dustahead/data/n0q_palette.npy); dBZ = index * 0.5 - 32. The WMS
background is white, which is also the palette colour of index 183 (59.5 dBZ); white is mapped to "no echo", so a
genuine 59.5 dBZ pixel is lost - negligible at the cell level. Cells take the MAXIMUM reflectivity of their pixels
(a fine line is 1-3 pixels wide; a mean would wash it out). Cached per 5-min time as float16 dBZ.
Channels (sampler.RADAR_KEYS): frames at t0-20, t0-10, t0 min, clip(dBZ, 0, 70) / 60.
"""
from __future__ import annotations

import datetime as dt
import io
from pathlib import Path

import numpy as np
import requests

from ..grid import Grid
from ._cache import keyed_lock

WMS = "https://mesonet.agron.iastate.edu/cgi-bin/wms/nexrad/n0q-t.cgi"
PX = 0.005                                   # mosaic pixel size (degrees)
_PAL = None
_LUT = None


def _lut():
    global _PAL, _LUT
    if _LUT is None:
        _PAL = np.load(Path(__file__).with_name("n0q_palette.npy"))
        keys = _PAL[:, 0].astype(np.int64) * 65536 + _PAL[:, 1].astype(np.int64) * 256 + _PAL[:, 2].astype(np.int64)
        _LUT = {int(k): i for i, k in enumerate(keys)}
    return _LUT


def fetch_rgb(t: dt.datetime, grid: Grid, timeout: int = 120) -> np.ndarray | None:
    """The mosaic over the domain at native pixel size as an RGB array, or None when the frame does not exist."""
    from PIL import Image
    w = int(round((grid.lon_max - grid.lon_min) / PX))
    h = int(round((grid.lat_max - grid.lat_min) / PX))
    params = dict(SERVICE="WMS", VERSION="1.1.1", REQUEST="GetMap", LAYERS="nexrad-n0q-wmst", SRS="EPSG:4326",
                  BBOX=f"{grid.lon_min},{grid.lat_min},{grid.lon_max},{grid.lat_max}", WIDTH=w, HEIGHT=h,
                  FORMAT="image/png", TIME=t.strftime("%Y-%m-%dT%H:%M:00Z"))
    r = requests.get(WMS, params=params, timeout=timeout)
    if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image"):
        return None
    im = Image.open(io.BytesIO(r.content)).convert("RGB")
    return np.asarray(im)


def rgb_to_dbz(rgb: np.ndarray) -> np.ndarray:
    key = rgb[..., 0].astype(np.int64) * 65536 + rgb[..., 1].astype(np.int64) * 256 + rgb[..., 2].astype(np.int64)
    lut = _lut()
    u, inv = np.unique(key, return_inverse=True)
    idx = np.array([lut.get(int(k), 0) for k in u], np.int16)[inv].reshape(key.shape)
    idx[key == 255 * 65536 + 255 * 256 + 255] = 0          # WMS background (white) -> no echo
    return np.clip(idx.astype(np.float32) * 0.5 - 32.0, 0.0, 80.0)


def cell_max(dbz: np.ndarray, grid: Grid) -> np.ndarray:
    """Max reflectivity per grid cell; row 0 of the result is the SOUTH edge (grid convention), the image's is north."""
    ny, nx = grid.shape
    H, W = dbz.shape
    re = np.floor(np.arange(ny) * H / ny).astype(int)
    ce = np.floor(np.arange(nx) * W / nx).astype(int)
    rows = np.maximum.reduceat(dbz, re, axis=0)
    cells = np.maximum.reduceat(rows, ce, axis=1)
    return cells[::-1].astype(np.float16)


def frame_cached(t: dt.datetime, grid: Grid, cache_dir: str | Path) -> np.ndarray | None:
    """Cell-max dBZ for the 5-min frame nearest to t (cache: radar/n0q_YYYYmmddHHMM.npy); None when IEM has no frame."""
    t = t.replace(second=0, microsecond=0)
    t = t - dt.timedelta(minutes=t.minute % 5)
    cache = Path(cache_dir) / "radar"
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"n0q_{t:%Y%m%d%H%M}.npy"
    miss = f.with_suffix(".missing")
    with keyed_lock(f):
        if f.exists():
            return np.load(f)
        if miss.exists():
            return None
        rgb = fetch_rgb(t, grid)
        if rgb is None:
            miss.touch()
            return None
        out = cell_max(rgb_to_dbz(rgb), grid)
        import os
        tmp = f.with_suffix(f".{os.getpid()}.tmp.npy")
        np.save(tmp, out)
        os.replace(tmp, f)
        return out


def radar_stack(t0: dt.datetime, grid: Grid, cache_dir: str | Path, offsets_min=(20, 10, 0)) -> np.ndarray:
    """(len(offsets), ny, nx) float32 dBZ at t0 - offsets; NaN where a frame is missing."""
    out = np.full((len(offsets_min),) + grid.shape, np.nan, np.float32)
    for k, off in enumerate(offsets_min):
        fr = frame_cached(t0 - dt.timedelta(minutes=off), grid, cache_dir)
        if fr is not None:
            out[k] = fr.astype(np.float32)
    return out
