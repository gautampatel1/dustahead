"""GOES-East ABI imagery and the Aerosol Detection Product from the NOAA open-data buckets.

Why GOES-East (GOES-16 until 2025-04-07, GOES-19 after): the GOES-West CONUS sector ends at scan angle
x = +0.070 rad, which cuts off everything east of about 107.5 W - Deming, Las Cruces and El Paso on I-10.
The GOES-East CONUS sector (x in [-0.1013, 0.0386]) covers the whole AZ+NM domain (x in [-0.092, -0.063]).
Verified against the real files on 2026-10-24.

Products (anonymous S3, no account; verified Oct 2026):
  s3://noaa-goes16|19/ABI-L2-MCMIPC/YYYY/DDD/HH/   Cloud & Moisture Imagery, CONUS, 2 km, every 5 min, ~50 MB, 16 bands
  s3://noaa-goes16|19/ABI-L2-ADPC/YYYY/DDD/HH/     Aerosol Detection (Dust / Smoke flags), CONUS, every 10 min, ~0.75 MB
  file name: OR_ABI-L2-MCMIPC-M6_G19_sYYYYDDDHHMMSSt_eYYYYDDDHHMMSSt_cYYYYDDDHHMMSSt.nc  (s = scan start)
  GOES-16 CONUS data stops 2025-04-07 ~19 UTC; GOES-19 is operational GOES-East from 2025-04-07.

Bands we use (channel -> wavelength): 2 -> 0.64 um (visible), 7 -> 3.9 um, 11 -> 8.4 um,
13 -> 10.3 um, 14 -> 11.2 um, 15 -> 12.3 um. 11/13/14/15 make the Dust RGB:
  R = BT(12.3) - BT(10.3), G = BT(11.2) - BT(8.4), B = BT(10.3).

Transfer budget: measured 2026-10-24 on Gautam's PC - a full 40-50 MB MCMIPC file downloads in ~2 s
(150 Mbit/s) and reads locally in 0.8 s, whereas remote partial reads through the proxy took ~50 s per
band. So: download the whole file to a temp dir OUTSIDE the project (deleted right after), regrid, and
cache the result as a 0.6 MB .npz per frame in data/cache/goes/. Raw files are never kept.

Geolocation: the ABI fixed grid is a geostationary projection. We map our lat/lon grid to ABI
scan angles once (closed-form GOES-R geostationary projection, no pyproj) and cache the pixel indices (the CONUS sector does not move).
"""
from __future__ import annotations

import datetime as dt
import os
import re
import tempfile
import uuid
from pathlib import Path

import numpy as np

from ..grid import Grid
from ._cache import keyed_lock

_SCAN_RE = re.compile(r"_s(\d{14})_")
BANDS = (2, 7, 11, 13, 14, 15)
GOES_EAST_SWITCH = dt.datetime(2025, 4, 7, 0, 0, tzinfo=dt.timezone.utc)


def bucket_for(t: dt.datetime, satellite: str = "east") -> str:
    """NOAA bucket holding the operational GOES-East (or West) data at time t."""
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    if satellite == "west":
        return "noaa-goes18" if t >= dt.datetime(2023, 1, 4, tzinfo=dt.timezone.utc) else "noaa-goes17"
    return "noaa-goes19" if t >= GOES_EAST_SWITCH else "noaa-goes16"


def _fs():
    import s3fs
    # path-style addressing: every bucket is reached through s3.amazonaws.com (one host to allow-list)
    return s3fs.S3FileSystem(anon=True, default_block_size=8 * 2 ** 20,
                             config_kwargs={"s3": {"addressing_style": "path"}})


def list_files(bucket: str | None, product: str, t: dt.datetime) -> list[str]:
    """All files of a product in the UTC hour containing t. bucket=None -> operational GOES-East at t."""
    bucket = bucket or bucket_for(t)
    prefix = f"{bucket}/{product}/{t:%Y}/{t.timetuple().tm_yday:03d}/{t:%H}/"
    try:
        return sorted(_fs().ls(prefix))
    except FileNotFoundError:
        return []


def scan_start(path: str) -> dt.datetime:
    s = _SCAN_RE.search(path).group(1)
    return dt.datetime.strptime(s[:13], "%Y%j%H%M%S").replace(tzinfo=dt.timezone.utc)


def nearest_file(bucket: str | None, product: str, t: dt.datetime, max_minutes: float = 6.0) -> str | None:
    """The file whose scan start is closest to t (searching this hour and the previous one)."""
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    bucket = bucket or bucket_for(t)
    cands = list_files(bucket, product, t) + list_files(bucket, product, t - dt.timedelta(hours=1))
    if not cands:
        return None
    best = min(cands, key=lambda f: abs((scan_start(f) - t).total_seconds()))
    return best if abs((scan_start(best) - t).total_seconds()) <= max_minutes * 60 else None


def download(remote: str, raw_dir: str | Path) -> Path:
    dst = Path(raw_dir) / Path(remote).name
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        _fs().get(remote, str(dst))
    return dst


def fetch_tmp(remote: str) -> Path:
    """Download an S3 object to the system temp dir (set DUSTAHEAD_TMP to override). Caller deletes it."""
    d = Path(os.environ.get("DUSTAHEAD_TMP", tempfile.gettempdir())) / "dustahead_raw"
    d.mkdir(parents=True, exist_ok=True)
    dst = d / f"{uuid.uuid4().hex[:8]}_{Path(remote).name}"      # unique per call: threads may fetch the same object
    _fs().get(remote, str(dst))
    return dst


def open_remote(remote: str):
    """Lazily open an S3 NetCDF-4/HDF5 file; only the variables you touch are transferred."""
    import xarray as xr
    f = _fs().open(remote, "rb")
    return xr.open_dataset(f, engine="h5netcdf", mask_and_scale=True)


# ---- geolocation ----------------------------------------------------------
def geos_scan_angles(lon, lat, lon0: float, h: float, a: float, b: float, sweep: str = "x"):
    """Geodetic lon/lat (deg) -> ABI fixed-grid scan angles (x, y) in radians. Closed form from the
    GOES-R Product Definition and Users' Guide (PUG) vol. 3 sec. 4.2.8 - no pyproj needed. Points on the
    far side of the Earth come back NaN."""
    lon, lat = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    H = h + a                                    # satellite distance from the Earth's centre
    e2 = (a * a - b * b) / (a * a)
    phi_c = np.arctan((b * b) / (a * a) * np.tan(lat))                 # geocentric latitude
    r_c = b / np.sqrt(1.0 - e2 * np.cos(phi_c) ** 2)                   # geocentric radius
    dlon = lon - np.radians(lon0)
    sx = H - r_c * np.cos(phi_c) * np.cos(dlon)
    sy = -r_c * np.cos(phi_c) * np.sin(dlon)
    sz = r_c * np.sin(phi_c)
    visible = H * (H - sx) >= sy * sy + (a * a) / (b * b) * sz * sz
    if sweep == "x":                                                   # GOES-R family
        x = np.arcsin(-sy / np.sqrt(sx * sx + sy * sy + sz * sz))
        y = np.arctan(sz / sx)
    else:                                                              # Meteosat convention
        x = np.arctan(-sy / sx)
        y = np.arcsin(sz / np.sqrt(sx * sx + sy * sy + sz * sz))
    x, y = np.where(visible, x, np.nan), np.where(visible, y, np.nan)
    return x, y


class AbiGeolocator:
    """Maps the DustAhead grid onto ABI fixed-grid pixel indices for one sector."""

    def __init__(self, ds, grid: Grid):
        p = ds["goes_imager_projection"].attrs
        h = float(p["perspective_point_height"])
        glon, glat = grid.mesh()
        x_rad, y_rad = geos_scan_angles(glon, glat, float(p["longitude_of_projection_origin"]), h,
                                        float(p["semi_major_axis"]), float(p["semi_minor_axis"]), p.get("sweep_angle_axis", "x"))
        xm = x_rad                                # finite where the point is visible from the satellite
        x, y = ds["x"].values.astype(float), ds["y"].values.astype(float)
        dx, dy = x[1] - x[0], y[1] - y[0]
        self.j = np.rint((x_rad - x[0]) / dx).astype(int)
        self.i = np.rint((y_rad - y[0]) / dy).astype(int)
        self.valid = (self.i >= 0) & (self.i < len(y)) & (self.j >= 0) & (self.j < len(x)) & np.isfinite(xm)
        self.i = np.clip(self.i, 0, len(y) - 1)
        self.j = np.clip(self.j, 0, len(x) - 1)
        # bounding box of needed source pixels: lets us read a slab instead of the whole 1500x2500 image
        self.i0, self.i1 = int(self.i[self.valid].min()), int(self.i[self.valid].max()) + 1
        self.j0, self.j1 = int(self.j[self.valid].min()), int(self.j[self.valid].max()) + 1

    def slab(self, var) -> np.ndarray:
        """Read only the slab of a (y, x) variable that covers our grid, then sample it."""
        a = np.asarray(var[self.i0:self.i1, self.j0:self.j1], np.float32)
        ii = np.clip(self.i - self.i0, 0, a.shape[0] - 1)
        jj = np.clip(self.j - self.j0, 0, a.shape[1] - 1)
        out = a[ii, jj]
        out[~self.valid] = np.nan
        return out

    def __call__(self, arr: np.ndarray, fill=np.nan) -> np.ndarray:
        out = np.asarray(arr, np.float32)[self.i, self.j]
        out[~self.valid] = fill
        return out


_GEO: dict = {}


def geolocator_for(ds, grid: Grid, key: str) -> AbiGeolocator:
    if key not in _GEO:
        _GEO[key] = AbiGeolocator(ds, grid)
    return _GEO[key]


def read_bands(src, grid: Grid, bands=BANDS) -> dict[int, np.ndarray]:
    """Band -> (ny, nx) array on our grid. Reflectance for band 2, brightness temperature (K) otherwise.
    src: local path or an already-open xarray dataset (see open_remote)."""
    import xarray as xr
    ds = src if isinstance(src, xr.Dataset) else xr.open_dataset(src, engine="h5netcdf", mask_and_scale=True)
    try:
        geo = geolocator_for(ds, grid, f"mcmipc_{grid.shape}")
        out = {}
        for b in bands:
            v = geo.slab(ds[f"CMI_C{b:02d}"])
            dqf = geo.slab(ds[f"DQF_C{b:02d}"])
            out[b] = np.where(dqf == 0, v, np.nan).astype(np.float32)
        return out
    finally:
        if ds is not src:
            ds.close()


def frame_cached(remote: str, grid: Grid, cache_dir: str | Path, bands=BANDS) -> dict[int, np.ndarray]:
    """Regridded bands for one MCMIPC file, cached as data/cache/goes/<scan>.npz (0.6 MB)."""
    key = Path(remote).name.split("_s")[1][:13]
    p = Path(cache_dir) / f"{key}.npz"
    with keyed_lock(p):
        if p.exists():
            with np.load(p) as z:
                return {b: z[f"b{b}"].astype(np.float32) for b in bands}
        tmp = fetch_tmp(remote)
        try:
            out = read_bands(tmp, grid, bands)
        finally:
            tmp.unlink(missing_ok=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = p.with_suffix(f".{uuid.uuid4().hex[:6]}.tmp.npz")
        np.savez_compressed(tmp_p, **{f"b{b}": v.astype(np.float16) for b, v in out.items()})
        os.replace(tmp_p, p)
    return {b: v.astype(np.float32) for b, v in out.items()}


ADP_DUST_MASK = 48      # DQF bits 4-5: 0 = high confidence, 16 = medium, 32 = low, 48 = bad / no retrieval (e.g. night)


def read_adp_dust(remote_or_path, grid: Grid, min_confidence: str = "medium", cache_dir: str | Path | None = None) -> np.ndarray:
    """Aerosol Detection Product dust flag on our grid: 1 = dust, 0 = no dust, nan = no usable retrieval.
    Dust detection is mostly daytime; at night nearly every pixel is 'bad/missing' and comes back nan.
    With cache_dir, the regridded result is cached per file (each ADP file serves three lead times)."""
    import xarray as xr
    if cache_dir is not None:
        key = Path(remote_or_path).name.split("_s")[1][:13]
        p = Path(cache_dir) / f"{key}_{min_confidence}.npy"
        with keyed_lock(p):
            if p.exists():
                return np.load(p)
            out = read_adp_dust(remote_or_path, grid, min_confidence, None)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp_p = p.with_suffix(f".{uuid.uuid4().hex[:6]}.tmp.npy")
            np.save(tmp_p, out)
            os.replace(tmp_p, p)
            return out
    tmp = None
    if str(remote_or_path).startswith("noaa-"):
        tmp = fetch_tmp(remote_or_path)
        src = tmp
    else:
        src = remote_or_path
    ds = xr.open_dataset(src, engine="h5netcdf")
    try:
        geo = geolocator_for(ds, grid, f"adpc_{grid.shape}")
        dust = geo.slab(ds["Dust"])
        dqf = geo.slab(ds["DQF"])
        conf = np.nan_to_num(dqf, nan=ADP_DUST_MASK).astype(int) & ADP_DUST_MASK
        ok_level = {"high": (0,), "medium": (0, 16), "low": (0, 16, 32)}[min_confidence]
        usable = np.isin(conf, ok_level) & np.isfinite(dust)
        return np.where(usable, dust, np.nan).astype(np.float32)
    finally:
        ds.close()
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def dust_rgb(bt: dict[int, np.ndarray]) -> np.ndarray:
    """NOAA / SPoRT Dust RGB recipe -> (ny, nx, 3) uint8 image. Inputs in Kelvin."""
    def stretch(x, lo, hi, gamma=1.0):
        return np.clip((np.clip(x, lo, hi) - lo) / (hi - lo), 0, 1) ** (1 / gamma)
    r = stretch(bt[15] - bt[13], -6.7, 2.6, 1.0)
    g = stretch(bt[14] - bt[11], -0.5, 20.0, 2.5)
    b = stretch(bt[13] - 273.15, -11.95, 15.55, 1.0)
    rgb = np.stack([r, g, b], -1)
    return (np.nan_to_num(rgb) * 255).astype(np.uint8)


def frame_times(t0: dt.datetime, n: int, step_min: int) -> list[dt.datetime]:
    """t0-(n-1)*step ... t0."""
    return [t0 - dt.timedelta(minutes=step_min * k) for k in range(n - 1, -1, -1)]
