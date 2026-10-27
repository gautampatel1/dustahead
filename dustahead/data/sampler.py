"""Builds one training sample: the 62-channel input stack at forecast time t0 and its targets.

Channel layout (index : content : normalisation)
   0-35  satellite   6 frames (t0-50..t0 min) x 6 bands [0.64, 3.9, 8.4, 10.3, 11.2, 12.3 um]
                     band 2 reflectance as-is; brightness temps (K-273.15)/30
  36-41  lightning   GLM flash density per frame, log1p(flashes/cell/10 min)
  42-46  IR change   BT(10.3um)[k] - BT(10.3um)[k-1], k=1..5, /10   (cooling = updraft/outflow)
  47-54  HRRR        u10/15, v10/15, gust/25, cape/2000, dcape/1500, rh/100, soilm (MSTAV fraction), rain7d/50
  55-60  static      elevation/3000, slope/30, bare, landcover, sand/100, clay/100
  61     physics     erodibility prior E in [0,1]

Targets (per lead 15/30/60 min):
  y[l]  (ny, nx) in {0,1}:   1 inside 12 km of a station reporting dust at t0+lead, 0 inside 12 km of
                              a station reporting no dust; ADP dust pixels add weak 1s (weight 0.3)
  w[l]  (ny, nx) >= 0:       0 where unknown (masked), 1 near stations, 0.3 for ADP-only pixels
  station table:             (i, j, y_15, y_30, y_60, dust_now) for the station BCE term
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from ..grid import Grid
from . import goes, glm, hrrr, physics
from .static import NAMES as STATIC_NAMES

BANDS = (2, 7, 11, 13, 14, 15)
N_SAT, N_GLM, N_DIFF, N_HRRR, N_STATIC, N_PHYS = 36, 6, 5, 8, 6, 1
N_CH = N_SAT + N_GLM + N_DIFF + N_HRRR + N_STATIC + N_PHYS   # 62
HRRR_KEYS = ("u10", "v10", "gust", "cape", "dcape", "rh2m", "soilm", "rain7d")
HRRR_SCALE = (15.0, 15.0, 25.0, 2000.0, 1500.0, 100.0, 1.0, 50.0)
STATIC_SCALE = (3000.0, 30.0, 1.0, 1.0, 100.0, 100.0)

CH = {  # channel index ranges, for the gate and for plotting
    "satellite": slice(0, 36), "lightning": slice(36, 42), "ir_change": slice(42, 47),
    "hrrr": slice(47, 55), "static": slice(55, 61), "physics": slice(61, 62),
    "station": slice(62, 63),   # painted at load time by the dataset: +1 dust reported at t0, -1 no dust, 0 no report (12 km discs)
}
N_CH_IN = 63


def paint_station_now(st_ij: np.ndarray, st_now: np.ndarray, shape: tuple[int, int], radius_cells: int = 3) -> np.ndarray:
    """One channel of what the stations say RIGHT NOW (available in real time): the model's 'persistence' input."""
    out = np.zeros(shape, np.float32)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    for (i, j), now in zip(st_ij, st_now):
        if now < 0:
            continue
        m = np.hypot(yy - i, xx - j) <= radius_cells
        out[m] = 1.0 if now == 1 else np.where(out[m] == 1.0, 1.0, -1.0)
    return out


def genesis_per_station(st_ij: np.ndarray, st_now: np.ndarray, radius_cells: int = 12) -> np.ndarray:
    """Per station: True when no station within `radius_cells` (~50 km, itself included) reports dust at t0.
    A positive label at such a station is a GENESIS case - onset that persistence cannot forecast."""
    ij = np.asarray(st_ij, float)
    now = np.asarray(st_now, int)
    out = np.ones(len(ij), bool)
    dusty = ij[now == 1]
    for k, (i, j) in enumerate(ij):
        if len(dusty) and (np.hypot(dusty[:, 0] - i, dusty[:, 1] - j) <= radius_cells).any():
            out[k] = False
    return out


def normalise_band(b: int, v: np.ndarray) -> np.ndarray:
    return v if b == 2 else (v - 273.15) / 30.0


def assemble(sat: list[dict[int, np.ndarray]], flashes: list[np.ndarray], hr: dict[str, np.ndarray],
             static: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Stack already-regridded pieces into the (62, ny, nx) float32 input."""
    ny, nx = static.shape[1:]
    x = np.zeros((N_CH, ny, nx), np.float32)
    k = 0
    for fr in sat:
        for b in BANDS:
            x[k] = np.nan_to_num(normalise_band(b, fr[b]))
            k += 1
    for f in flashes:
        x[k] = np.log1p(f)
        k += 1
    for t in range(1, len(sat)):
        x[k] = np.nan_to_num(sat[t][13] - sat[t - 1][13]) / 10.0
        k += 1
    for key, s in zip(HRRR_KEYS, HRRR_SCALE):
        x[k] = np.nan_to_num(hr[key]) / s
        k += 1
    for s_idx, s in enumerate(STATIC_SCALE):
        x[k] = static[s_idx] / s
        k += 1
    x[k] = prior
    return x


def paint_targets(grid: Grid, st_rows: pd.DataFrame, leads, radius_km: float, adp: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    y = np.zeros((len(leads),) + grid.shape, np.float32)
    w = np.zeros_like(y)
    for li, lead in enumerate(leads):
        if adp and lead in adp and adp[lead] is not None:
            d = adp[lead]
            known = np.isfinite(d)
            y[li][known] = (d[known] > 0.5)
            w[li][known] = 0.3
        for r in st_rows.itertuples():
            v = getattr(r, f"y_{lead}")
            if v < 0:
                continue
            m = grid.disk_mask(r.lon, r.lat, radius_km)
            y[li][m] = v
            w[li][m] = 1.0
    return y, w


def build_sample(t0: dt.datetime, grid: Grid, cfg: dict, static: np.ndarray, labels: pd.DataFrame,
                 stations: pd.DataFrame, use_adp: bool = True) -> dict | None:
    """Fetch everything for one t0 and return the sample dict, or None if satellite data is missing."""
    from ..events.stations import station_targets
    T, S, P = cfg["time"], cfg["satellite"], cfg["paths"]
    hrrr.set_cache(P["cache"])
    times = goes.frame_times(t0, T["history_frames"], T["step_min"])
    sat, flashes = [], []
    for t in times:
        f = goes.nearest_file(S["bucket"], S["product"], t)
        if f is None:
            return None
        sat.append(goes.frame_cached(f, grid, Path(P["cache"]) / "goes", BANDS))
        flashes.append(glm.flash_density_at(t, grid, cfg))
    hr = hrrr.hrrr_stack(t0, grid)
    hr["rain7d"] = hrrr.hrrr_7day_rain(t0, grid, days=7)
    clay, bare = static[STATIC_NAMES.index("clay")], static[STATIC_NAMES.index("bare")]
    prior = physics.erodibility_prior(hr["wind10"], hr["soilm"] * 25.0, clay, bare)
    x = assemble(sat, flashes, hr, static, prior)

    st = station_targets(labels, stations, pd.Timestamp(t0), T["leads_min"])
    st = st.merge(stations[["sid", "lon", "lat", "i", "j"]], left_on="station", right_on="sid")
    adp = None
    if use_adp:
        adp = {}
        for lead in T["leads_min"]:
            f = goes.nearest_file(S["bucket"], S["dust_product"], t0 + dt.timedelta(minutes=lead))
            adp[lead] = goes.read_adp_dust(f, grid, cache_dir=Path(P["cache"]) / "adp") if f else None
    y, w = paint_targets(grid, st, T["leads_min"], cfg["label"]["station_radius_km"], adp)
    dust_now = st.dust_now.values
    genesis = not bool((dust_now == 1).any())
    return dict(x=x.astype(np.float16), y=y.astype(np.uint8), w=w.astype(np.float16),
                st_ij=st[["i", "j"]].values.astype(np.int16),
                st_y=st[[f"y_{l}" for l in T["leads_min"]]].values.astype(np.int8),
                st_now=dust_now.astype(np.int8), t0=np.datetime64(t0.replace(tzinfo=None)),
                genesis=np.array(genesis), dust_rgb=goes.dust_rgb(sat[-1]))


def save_sample(path: str | Path, sample: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **sample)


def load_sample(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def sample_times(event, cfg: dict) -> list[dt.datetime]:
    """Forecast times every step_min from (first warning - hours_before) to (last expiry + hours_after),
    capped at samples_per_event. Genesis cases live in the hours before the first warning."""
    T = cfg["time"]
    step = T["step_min"]
    start = pd.Timestamp(event.t_start).floor(f"{step}min") - pd.Timedelta(hours=T["hours_before"])
    end = pd.Timestamp(event.t_end).ceil(f"{step}min") + pd.Timedelta(hours=T["hours_after"])
    cap = T["samples_per_event"] // 2 if getattr(event, "kind", "dust") == "nodust" else T["samples_per_event"]
    n = min(cap, int((end - start) / pd.Timedelta(minutes=step)) + 1)
    return [(start + pd.Timedelta(minutes=step * k)).to_pydatetime() for k in range(n)]
