"""Preflight: prove every data source, package and the GPU work on THIS machine before any bulk download.

    python scripts/00_check_sources.py            # ~2 minutes, prints OK / FAIL per item with a fix hint
    python scripts/00_check_sources.py GOES HRRR  # only checks whose name contains GOES or HRRR
    python scripts/00_check_sources.py --gpu      # skip checks whose name contains "gpu"

Run this first on the training PC. Every FAIL must be fixed (or consciously accepted) before 03_build_samples.
"""
from __future__ import annotations

import importlib
import shutil
import sys
import time
import datetime as dt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

T_KNOWN = dt.datetime(2025, 8, 26, 2, 0, tzinfo=dt.timezone.utc)   # Phoenix haboob, 25 Aug 2025 19:00 MST
RESULTS: list[tuple[str, str, str]] = []
ONLY = [a for a in sys.argv[1:] if not a.startswith("-")]          # substrings of check names to run
SKIP = [a[2:] for a in sys.argv[1:] if a.startswith("--")]        # --gpu  skips checks containing "gpu"


def check(name: str, hint: str = ""):
    def deco(fn):
        low = name.lower()
        if (ONLY and not any(o.lower() in low for o in ONLY)) or any(k.lower() in low for k in SKIP):
            return fn
        t = time.time()
        try:
            msg = fn() or ""
            RESULTS.append((name, "OK", f"{msg} ({time.time() - t:.1f}s)"))
        except Exception as e:  # noqa: BLE001
            RESULTS.append((name, "FAIL", f"{type(e).__name__}: {str(e)[:160]}  -> {hint}"))
        print(f"  {RESULTS[-1][1]:4s} {name:36s} {RESULTS[-1][2]}", flush=True)
        return fn
    return deco


# ---------------- environment ----------------
@check("python packages", "pip install -r requirements.txt")
def _pkgs():
    missing = []
    for m in ("numpy", "pandas", "scipy", "yaml", "xarray", "h5netcdf", "s3fs", "pyproj", "requests", "herbie", "torch", "rasterio", "py3dep", "streamlit"):
        try:
            importlib.import_module(m)
        except Exception:  # noqa: BLE001
            missing.append(m)
    if missing:
        raise RuntimeError("missing: " + ", ".join(missing))
    return "all present"


@check("GPU (CUDA)", "install torch for CUDA 12.8: pip install torch --index-url https://download.pytorch.org/whl/cu128")
def _gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    p = torch.cuda.get_device_properties(0)
    return f"{p.name}, {p.total_memory / 2**30:.1f} GB, torch {torch.__version__}"


@check("disk space", "need >= 100 GB free on the drive holding data/")
def _disk():
    free = shutil.disk_usage(Path(__file__).resolve().parents[1]).free / 2**30
    if free < 100:
        raise RuntimeError(f"only {free:.0f} GB free")
    return f"{free:.0f} GB free"


# ---------------- NOAA on AWS ----------------
@check("GOES-East MCMIPC listing", "AWS S3 blocked? try from another network; buckets noaa-goes16/19 are public")
def _goes_list():
    from dustahead.data.goes import list_files, nearest_file
    files = list_files(None, "ABI-L2-MCMIPC", T_KNOWN)
    f = nearest_file(None, "ABI-L2-MCMIPC", T_KNOWN)
    if not files or f is None:
        raise RuntimeError("no files")
    return f"{len(files)} files in hour; nearest {Path(f).name[:45]}"


@check("GOES-East frame download + regrid", "pip install h5netcdf s3fs; check pyproj")
def _goes_read():
    import numpy as np
    from dustahead import load_config
    from dustahead.data.goes import fetch_tmp, nearest_file, read_bands
    from dustahead.grid import Grid
    g = Grid.from_config(load_config())
    f = nearest_file(None, "ABI-L2-MCMIPC", T_KNOWN)
    t = time.time()
    tmp = fetch_tmp(f)
    t_dl = time.time() - t
    try:
        b = read_bands(tmp, g)
    finally:
        tmp.unlink(missing_ok=True)
    bt13 = b[13]
    ok = np.isfinite(bt13).mean()
    if ok < 0.9:
        raise RuntimeError(f"only {ok:.0%} valid pixels after regrid")
    return f"BT10.3 {np.nanmin(bt13):.0f}-{np.nanmax(bt13):.0f} K, {ok:.0%} valid; download {t_dl:.1f}s + regrid {time.time() - t - t_dl:.1f}s"


@check("GOES-East ADPC (dust flag)", "product ABI-L2-ADPC missing for this hour?")
def _adp():
    import numpy as np
    from dustahead import load_config
    from dustahead.data.goes import nearest_file, read_adp_dust
    from dustahead.grid import Grid
    f = nearest_file(None, "ABI-L2-ADPC", T_KNOWN, max_minutes=11)
    d = read_adp_dust(f, Grid.from_config(load_config()))
    return f"dust pixels {int(np.nansum(d))}, retrieved {np.isfinite(d).mean():.0%}"


@check("GLM lightning window", "GLM-L2-LCFA on the GOES-East bucket")
def _glm():
    from dustahead import load_config
    from dustahead.data.glm import files_in_window, flash_density
    from dustahead.grid import Grid
    files = files_in_window(None, "GLM-L2-LCFA", T_KNOWN, 10)
    if len(files) < 25:
        raise RuntimeError(f"only {len(files)} files in 10 min (expect ~30)")
    d = flash_density(files, Grid.from_config(load_config()))
    return f"{len(files)} files, {int(d.sum())} flashes in domain"


@check("HRRR via Herbie", "pip install herbie-data eccodes cfgrib (conda-forge eccodes on Windows)")
def _hrrr():
    import numpy as np
    from dustahead import load_config
    from dustahead.data import hrrr
    from dustahead.grid import Grid
    g = Grid.from_config(load_config())
    u = hrrr.field_on_grid(T_KNOWN, "u10", g)
    s = hrrr.field_on_grid(T_KNOWN, "soilm", g)
    r = hrrr.field_on_grid(T_KNOWN - dt.timedelta(hours=2), "apcp6", g, fxx=6)
    return f"u10 {np.nanmin(u):.1f}..{np.nanmax(u):.1f} m/s, soilm {np.nanmean(s):.3f}, apcp6 max {np.nanmax(r):.1f} mm"


# ---------------- Iowa Environmental Mesonet ----------------
@check("IEM NWS warnings (DS.W)", "https://mesonet.agron.iastate.edu/json/vtec_events_bystate.py")
def _warn():
    from dustahead.events.warnings import fetch_polygon, fetch_warnings
    df = fetch_warnings("AZ", 2025, "DS", "W")
    rings = fetch_polygon("PSR", 2025, 10)
    return f"{len(df)} AZ dust warnings in 2025; polygon PSR#10 has {len(rings or [])} ring(s)"


@check("IEM ASOS 5-min reports", "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help")
def _asos():
    import pandas as pd
    from dustahead.events.stations import fetch_asos
    df = fetch_asos(["PHX", "CGZ", "TUS"], pd.Timestamp("2025-08-26T00:00Z"), pd.Timestamp("2025-08-26T02:00Z"))
    ds = df[df.wxcodes.str.contains("DS")]
    if ds.empty:
        raise RuntimeError("expected +DS rows at PHX 00:35-01:10Z")
    return f"{len(df)} rows, {len(ds)} with DS, min vis {df.vsby.min():.2f} mi, max gust {df.gust.max():.0f} kt"


@check("IEM station metadata", "https://mesonet.agron.iastate.edu/geojson/network/AZ_ASOS.geojson")
def _stations():
    from dustahead import load_config
    from dustahead.events.stations import fetch_station_table
    from dustahead.grid import Grid
    st = fetch_station_table(Grid.from_config(load_config()))
    if st.network.isna().all():
        raise RuntimeError("fell back to the built-in list")
    return f"{len(st)} stations in domain, {int(st.i10_core.sum())} on I-10 core list"


# ---------------- NCEI ----------------
@check("NCEI Storm Events", "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/")
def _se():
    from dustahead.events.storm_events import fetch_dust_storm_events, list_detail_files
    files = list_detail_files()
    df = fetch_dust_storm_events(2025, cache_dir="data/raw/storm_events")
    return f"years {min(files)}-{max(files)}; 2025 dust storm entries AZ/NM/TX/CA: {len(df)}"


# ---------------- static layers ----------------
@check("USGS 3DEP elevation", "pip install py3dep; service elevation.nationalmap.gov")
def _dem():
    import os
    os.environ.setdefault("HYRIVER_CACHE_DISABLE", "true")
    import py3dep
    dem = py3dep.get_dem((-112.2, 33.3, -112.0, 33.5), resolution=90)
    return f"Phoenix tile {dem.shape}, {float(dem.min()):.0f}-{float(dem.max()):.0f} m"


@check("SoilGrids WCS", "https://maps.isric.org/mapserv?map=/map/clay.map&SERVICE=WCS&REQUEST=GetCapabilities")
def _soil():
    import requests
    r = requests.get("https://maps.isric.org/mapserv", params={"map": "/map/clay.map", "SERVICE": "WCS", "VERSION": "2.0.1", "REQUEST": "GetCapabilities"}, timeout=60)
    r.raise_for_status()
    if "clay_0-5cm_mean" not in r.text:
        raise RuntimeError("coverage clay_0-5cm_mean not listed")
    return "clay_0-5cm_mean available"


@check("USDA Cropland Data Layer", "https://nassgeodata.gmu.edu/CropScape/ (GetCDLFile REST)")
def _cdl():
    import re
    import requests
    r = requests.get("https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile",
                     params=dict(year=2024, bbox="-1800000,1150000,-1790000,1160000"), timeout=120)
    r.raise_for_status()
    m = re.search(r"<returnURL>(.*?)</returnURL>", r.text)
    if not m:
        raise RuntimeError(r.text[:120])
    return "returnURL received"


@check("NLCD land cover (manual)", "download Annual NLCD Land Cover CONUS from https://www.mrlc.gov/data and pass --nlcd to 02_build_static.py")
def _nlcd():
    p = list(Path("data/static/tif").glob("*nlcd*.tif")) + list(Path("data/static/tif").glob("*NLCD*.tif")) if Path("data/static/tif").exists() else []
    if not p:
        raise RuntimeError("no NLCD GeoTIFF in data/static/tif/ (optional for Phase 1: CDL gives the bare-ground layer)")
    return p[0].name


# ---------------- transfer speed ----------------
@check("download speed (S3)", "a slow link means the full build takes days; run it overnight with --workers 6")
def _speed():
    from dustahead.data.goes import _fs, nearest_file
    f = nearest_file(None, "ABI-L2-MCMIPC", T_KNOWN)
    t = time.time()
    with _fs().open(f, "rb") as fh:
        n = len(fh.read(24 * 2**20))
    mbps = n * 8 / 2**20 / (time.time() - t)
    return f"{mbps:.0f} Mbit/s"


if __name__ == "__main__":
    fails = [r for r in RESULTS if r[1] == "FAIL"]
    print(f"\n{len(RESULTS) - len(fails)} OK, {len(fails)} FAIL\n")
    sys.exit(1 if fails else 0)
