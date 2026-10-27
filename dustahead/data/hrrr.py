"""High-Resolution Rapid Refresh (HRRR) surface fields via Herbie (s3://noaa-hrrr-bdp-pds, verified Oct 2026:
hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfsfcfFF.grib2, 138 MB per file with a .idx inventory, so Herbie fetches
only the GRIB messages we ask for, ~1-2 MB each).

We use the HRRR ANALYSIS (fxx=0) valid at the top of the hour at or before t0, so every input is
something an operational system would have had at forecast time. Fields:
  u10, v10 (10 m wind), gust (surface gust), cape (surface-based CAPE),
  rh2m (2 m relative humidity), soilm (MSTAV surface moisture availability, 0-100 % -> we store 0-1), t2m, d2m.
  The sfc product carries no SOILW layers (verified in the .idx, 2026-10-24); MSTAV is the surface
  moisture proxy. The physics prior maps it to gravimetric % as 25 * MSTAV_fraction (see sampler).
Rain: the f00 analysis has no precipitation, so hrrr_7day_rain sums the 0-6 h accumulations of the
00/06/12/18 UTC runs (fxx=6) over the previous 7 days (28 fields). Cached per field-hour as .npy.
DCAPE (downdraft CAPE) is not an HRRR output; dcape_proxy() uses the 2 m dewpoint depression. A proxy.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

from ..grid import Grid
from ._cache import keyed_lock
from .regrid import NearestRegridder

# Herbie search strings (regex over the GRIB2 .idx inventory)
SEARCH = {
    "u10": ":UGRD:10 m above ground:",
    "v10": ":VGRD:10 m above ground:",
    "gust": ":GUST:surface:",
    "cape": ":CAPE:surface:",
    "rh2m": ":RH:2 m above ground:",
    "soilm": ":MSTAV:0 m underground:",      # HRRR sfc has no SOILW; MSTAV = surface moisture availability, 0-100 %
    "vis": ":VIS:surface:",
    "t2m": ":TMP:2 m above ground:",
    "d2m": ":DPT:2 m above ground:",
    "apcp6": ":APCP:surface:0-6 hour acc",
}
ANALYSIS_FIELDS = ("u10", "v10", "gust", "cape", "rh2m", "soilm", "t2m", "d2m")
_REGRID: dict = {}
_CACHE: Path | None = None


def set_cache(path: str | Path) -> None:
    global _CACHE
    _CACHE = Path(path) / "hrrr"
    _CACHE.mkdir(parents=True, exist_ok=True)


def _open(run: dt.datetime, field: str, fxx: int):
    import os, tempfile
    from herbie import Herbie
    save_dir = Path(os.environ.get("DUSTAHEAD_TMP", tempfile.gettempdir())) / "dustahead_raw" / "herbie"
    save_dir.mkdir(parents=True, exist_ok=True)
    H = Herbie(run.replace(tzinfo=None), model="hrrr", product="sfc", fxx=fxx, verbose=False, save_dir=save_dir)
    ds = H.xarray(SEARCH[field], remove_grib=True)
    if isinstance(ds, list):
        ds = ds[0]
    return ds


def field_on_grid(run: dt.datetime, field: str, grid: Grid, fxx: int = 0) -> np.ndarray:
    run = run.replace(minute=0, second=0, microsecond=0)
    cache = _CACHE / f"{run:%Y%m%dT%H}_f{fxx:02d}_{field}.npy" if _CACHE else None
    with keyed_lock(cache or f"{run}_{fxx}_{field}"):
        if cache and cache.exists():
            return np.load(cache)
        ds = _open(run, field, fxx)
        var = [v for v in ds.data_vars][0]
        lon = ds["longitude"].values
        lon = np.where(lon > 180, lon - 360, lon)
        key = ("hrrr", grid.shape)
        with keyed_lock("hrrr_regridder"):
            if key not in _REGRID:
                _REGRID[key] = NearestRegridder(lon, ds["latitude"].values, grid, max_km=6.0)
        out = _REGRID[key](np.squeeze(ds[var].values))
        if cache:
            import os
            tmp = cache.with_suffix(f".{os.getpid()}_{id(out)}.tmp.npy")
            np.save(tmp, out)
            os.replace(tmp, cache)
    return out


def hrrr_stack(t0: dt.datetime, grid: Grid) -> dict[str, np.ndarray]:
    """All HRRR analysis inputs for one forecast time (run hour <= t0)."""
    out = {k: field_on_grid(t0, k, grid) for k in ANALYSIS_FIELDS}
    out["soilm"] = out["soilm"] / 100.0          # MSTAV % -> fraction
    out["dcape"] = dcape_proxy(out["t2m"], out["d2m"])
    out["wind10"] = np.hypot(out["u10"], out["v10"])
    return out


def hrrr_7day_rain(t0: dt.datetime, grid: Grid, days: int = 7) -> np.ndarray:
    """Sum of 0-6 h HRRR precipitation (mm) from the 00/06/12/18 UTC runs over the previous `days`."""
    total = np.zeros(grid.shape, np.float32)
    last = t0.replace(minute=0, second=0, microsecond=0)
    last = last - dt.timedelta(hours=last.hour % 6)          # most recent synoptic run <= t0
    for k in range(4 * days):
        run = last - dt.timedelta(hours=6 * (k + 1))          # runs whose 6 h accumulation ended before t0
        try:
            total += np.nan_to_num(field_on_grid(run, "apcp6", grid, fxx=6))
        except Exception as e:  # noqa: BLE001  (missing run: skip, but say so)
            print(f"[hrrr] rain: skipped run {run:%Y-%m-%d %H}Z ({type(e).__name__})")
    return total


def dcape_proxy(t2m_k: np.ndarray, d2m_k: np.ndarray) -> np.ndarray:
    """Downdraft-CAPE proxy (J/kg) from the 2 m dewpoint depression: drier low levels -> stronger
    evaporatively cooled outflow (after Gilmore & Wicker 1998). Proxy only; say so in the paper."""
    dd = np.clip(t2m_k - d2m_k, 0, 40)
    return (60.0 * dd).astype(np.float32)
