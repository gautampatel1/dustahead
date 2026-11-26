"""Builds one training sample: the 62-channel input stack at forecast time t0 and its targets.

Channel layout (index : content : normalisation)
   0-35  satellite   6 frames (t0-50..t0 min) x 6 bands [0.64, 3.9, 8.4, 10.3, 11.2, 12.3 um]
                     band 2 reflectance as-is; brightness temps (K-273.15)/30
  36-41  lightning   GLM flash density per frame, log1p(flashes/cell/10 min)
  42-46  IR change   BT(10.3um)[k] - BT(10.3um)[k-1], k=1..5, /10   (cooling = updraft/outflow)
  47-54  HRRR        u10/15, v10/15, gust/25, cape/2000, dcape/1500, rh/100, soilm (MSTAV fraction), rain7d/50
  55-60  static      elevation/3000, slope/30, bare, landcover, sand/100, clay/100
  61     physics     erodibility prior E in [0,1]
  62     station     painted at load time by the dataset (+1 dust reported now, -1 clear, 0 no report)
  69-71  radar       OPTIONAL (key `xr`, channels.radar): NEXRAD composite reflectivity at t0-20, t0-10, t0,
                     clip(dBZ,0,70)/60 - the outflow boundary ("fine line") that precedes a haboob (data/radar.py).
  63-68  hrrr_fcst   OPTIONAL (stored as a separate key `xf`, enabled by channels.hrrr_fcst): HRRR short-range
                     FORECAST valid about t0+1 h from the run initialised 2 h before t0 (so it existed at t0):
                     u10/15, v10/15, gust/25, cape/2000, refc/60 (composite reflectivity, dBZ), and gust/25 valid
                     about t0+2 h. The forecast outflow is the physical driver of dust onset; the analysis only says
                     what the wind is doing now.

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

FCST_KEYS = ("u10_f1", "v10_f1", "gust_f1", "cape_f1", "refc_f1", "gust_f2")     # HRRR forecast channels (key `xf`)
FCST_SCALE = (15.0, 15.0, 25.0, 2000.0, 60.0, 25.0)
N_FCST = len(FCST_KEYS)

CH = {  # channel index ranges, for the gate and for plotting
    "satellite": slice(0, 36), "lightning": slice(36, 42), "ir_change": slice(42, 47),
    "hrrr": slice(47, 55), "static": slice(55, 61), "physics": slice(61, 62),
    "station": slice(62, 63),   # painted at load time by the dataset: +1 dust reported at t0, -1 no dust, 0 no report (12 km discs)
    "hrrr_fcst": slice(63, 63 + N_FCST),   # only present when the model asks for it (channels.hrrr_fcst > 0)
}
RADAR_KEYS = ("dbz_m20", "dbz_m10", "dbz_0")      # NEXRAD composite reflectivity at t0-20, t0-10, t0 (key `xr`)
N_RADAR = len(RADAR_KEYS)
CH["radar"] = slice(63 + N_FCST, 63 + N_FCST + N_RADAR)   # static slot: after the forecast block even if that is absent
WIND_KEYS = ("peak_kt", "d_peak_30min", "d_temp_30min", "has_report")   # station wind painted at load time (key `st_wnow`)
N_WIND = len(WIND_KEYS)
WIND_SCALE = (40.0, 20.0, 10.0)     # knots, knots per 30 min, deg F per 30 min
CH["wind"] = slice(63 + N_FCST + N_RADAR, 63 + N_FCST + N_RADAR + N_WIND)   # static slot after the radar block
N_CH_IN = 63                 # base input: stored x (62) + station channel
N_CH_IN_FCST = 63 + N_FCST   # with the HRRR forecast block appended
N_CH_IN_FULL = 63 + N_FCST + N_RADAR
N_CH_IN_WIND = N_CH_IN_FULL + N_WIND
N_DNOW = 2                          # station blowing-dust state at t0: severity level, has-report mask (key `st_dvis_now`)
CH["dust_now"] = slice(N_CH_IN_WIND, N_CH_IN_WIND + N_DNOW)     # static slot after the wind block
N_CH_IN_DNOW = N_CH_IN_WIND + N_DNOW
N_HOT = 2                           # dust hotspot climatology: satellite dust frequency, warning-polygon frequency
CH["hotspot"] = slice(N_CH_IN_DNOW, N_CH_IN_DNOW + N_HOT)       # static slot after the dust_now block
N_CH_IN_HOT = N_CH_IN_DNOW + N_HOT


def assemble_forecast(fc: dict[str, np.ndarray]) -> np.ndarray:
    """(N_FCST, ny, nx) float32 block from hrrr.forecast_stack() output, normalised like the analysis channels."""
    return np.stack([np.nan_to_num(fc[k]) / s for k, s in zip(FCST_KEYS, FCST_SCALE)]).astype(np.float32)


def assemble_radar(stack: np.ndarray) -> np.ndarray:
    """(N_RADAR, ny, nx) float32 from radar.radar_stack(): clip(dBZ, 0, 70) / 60, missing frames -> 0."""
    return (np.clip(np.nan_to_num(stack, nan=0.0), 0.0, 70.0) / 60.0).astype(np.float32)


def input_stack(sample: dict, st_now: np.ndarray | None = None, with_forecast: bool = False,
                with_radar: bool = False, radar_drop_newest: int = 0, radar_key: str = "xr",
                with_wind: bool = False, st_wnow: np.ndarray | None = None, with_dust_now: bool = False,
                hotspot: np.ndarray | None = None) -> np.ndarray:
    """The model input for one stored sample: x (62) + painted station channel (+ the forecast block `xf`
    + the radar block `xr`). Blocks sit at fixed channel slots (CH), so when the radar block is wanted the forecast
    slot is filled with zeros if the model does not use it. `st_now` overrides the stored reports (station dropout,
    live reports). A sample missing a wanted block gets zeros there, and a warning is printed once."""
    x = sample["x"].astype(np.float32)
    now = sample["st_now"] if st_now is None else st_now
    parts = [x, paint_station_now(sample["st_ij"], now, x.shape[1:])[None]]
    with_wind_slot = with_wind or with_dust_now
    if with_forecast or with_radar or with_wind_slot:
        parts.append(_block(sample, "xf", N_FCST, x.shape[1:], warn=with_forecast))
    if with_wind_slot and not with_radar:       # keep the wind block at its fixed slot: radar slot filled with zeros
        parts.append(np.zeros((N_RADAR,) + x.shape[1:], np.float32))
    if with_radar:
        xr = _block(sample, radar_key, N_RADAR, x.shape[1:], warn=True)
        if radar_drop_newest:          # latency test: the newest frame(s) are not yet in the mosaic at t0
            xr = xr.copy()
            xr[N_RADAR - radar_drop_newest:] = 0.0
        parts.append(xr)
    if with_wind:
        wnow = sample.get("st_wnow") if st_wnow is None else st_wnow
        if wnow is None:
            if "st_wnow" not in _WARNED:
                print("[sampler] sample without `st_wnow`: zeros used (run scripts/20_add_wind.py)", flush=True)
                _WARNED.add("st_wnow")
            parts.append(np.zeros((N_WIND,) + x.shape[1:], np.float32))
        else:
            parts.append(paint_station_wind(sample["st_ij"], wnow, x.shape[1:]))
    elif with_dust_now:
        parts.append(np.zeros((N_WIND,) + x.shape[1:], np.float32))
    if with_dust_now:
        if "st_dvis_now" not in sample:
            if "st_dvis_now" not in _WARNED:
                print("[sampler] sample without `st_dvis_now`: zeros used (run scripts/22_add_dust_tiers.py)", flush=True)
                _WARNED.add("st_dvis_now")
            parts.append(np.zeros((N_DNOW,) + x.shape[1:], np.float32))
        else:
            parts.append(paint_station_dust_now(sample["st_ij"], sample["st_dvis_now"], x.shape[1:]))
    out = np.concatenate(parts, 0)
    if hotspot is not None:                # fixed slot after every other block: unused slots before it are zeros
        pad = N_CH_IN_DNOW - out.shape[0]
        out = np.concatenate([out, np.zeros((pad,) + x.shape[1:], np.float32), hotspot.astype(np.float32)], 0)
    return out


def load_hotspot(cfg: dict) -> np.ndarray | None:
    """channels.hotspot: the dust hotspot climatology (N_HOT, ny, nx) written by scripts/26_build_hotspot.py, or None.
    Dust lifts again and again from the same sources (dry lake beds, fallow fields); the map tells the gate where
    strong wind has produced dust before. Built from 2018-2024 only, so no validation or test event is in it."""
    if not cfg.get("channels", {}).get("hotspot", 0):
        return None
    from pathlib import Path
    f = Path(cfg["paths"]["static"]) / "hotspot_2018_2024.npy"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: run scripts/26_build_hotspot.py")
    return np.load(f).astype(np.float32)


def wants_dust_now(cfg: dict) -> bool:
    """channels.dust_now: paint what the stations report about dust RIGHT NOW at any severity (a dust weather code
    with its visibility), not only the strict <= 1 mi label of the station channel. Blowing dust at 3-6 mi upstream
    is the earliest ground observation of a storm that later reaches 1 mi."""
    return bool(cfg.get("channels", {}).get("dust_now", 0))


def paint_station_dust_now(st_ij: np.ndarray, st_dvis_now: np.ndarray, shape: tuple[int, int], radius_cells: int = 3) -> np.ndarray:
    """(N_DNOW, ny, nx): channel 0 = severity (6 - visibility) / 6 of a dust-coded report (0 = no dust code, 1 = zero
    visibility), channel 1 = 1 where a station reported at all. Worst report wins where discs overlap."""
    out = np.zeros((N_DNOW,) + tuple(shape), np.float32)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    dv = np.asarray(st_dvis_now, np.float32)
    lvl = np.clip((6.0 - dv) / 6.0, 0.0, 1.0)
    for k in np.argsort(np.nan_to_num(lvl, nan=-1.0)):
        if not np.isfinite(dv[k]):
            continue
        i, j = st_ij[k]
        m = np.hypot(yy - i, xx - j) <= radius_cells
        out[0][m] = lvl[k]
        out[1][m] = 1.0
    return out


_WARNED = set()


def _block(sample: dict, key: str, n: int, shape, warn: bool) -> np.ndarray:
    if key in sample:
        return sample[key].astype(np.float32)
    if warn and key not in _WARNED:
        print(f"[sampler] sample without block `{key}`: zeros used (run the matching scripts/1x_add_*.py)", flush=True)
        _WARNED.add(key)
    return np.zeros((n,) + tuple(shape), np.float32)


def wants_forecast(cfg: dict) -> bool:
    return bool(cfg.get("channels", {}).get("hrrr_fcst", 0))


def wants_wind(cfg: dict) -> bool:
    """channels.wind: paint what the stations measure RIGHT NOW (peak wind, its 30-min change, the 30-min temperature
    change) as sky-branch inputs. A thunderstorm outflow reaches a station as a gust, a wind shift and a temperature
    drop minutes before the dust wall does."""
    return bool(cfg.get("channels", {}).get("wind", 0))


def wants_radar(cfg: dict) -> bool:
    return bool(cfg.get("channels", {}).get("radar", 0))


def radar_key(cfg: dict) -> str:
    """channels.radar_key: which stored radar block to use. `xr` = frames at t0-20/-10/0 (19_add_radar.py default);
    `xr_lag5` = t0-25/-15/-5, i.e. what a live system has with the IEM mosaic's ~5-min publication delay."""
    return str(cfg.get("channels", {}).get("radar_key", "xr"))


def radar_drop_newest(cfg: dict) -> int:
    """channels.radar_latency_frames: how many of the newest radar frames (t0, then t0-10) to blank, so that a
    model only ever sees frames the IEM mosaic has published by t0 (it lags real time by a few minutes)."""
    return int(cfg.get("channels", {}).get("radar_latency_frames", 0))


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


def paint_station_wind(st_ij: np.ndarray, st_wnow: np.ndarray, shape: tuple[int, int], radius_cells: int = 3) -> np.ndarray:
    """(N_WIND, ny, nx): station wind at t0 painted on 12 km discs. st_wnow: (S, 3) = peak wind (max of sustained
    and gust, knots), its change over the last ~30 min (knots), the temperature change over the last ~30 min (deg F);
    NaN = not reported. Channel 4 is 1 where a station reported wind at all, so "calm" and "no station" differ.
    Where discs overlap the station with the stronger wind wins."""
    out = np.zeros((N_WIND,) + tuple(shape), np.float32)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    w = np.asarray(st_wnow, np.float32)
    order = np.argsort(np.nan_to_num(w[:, 0], nan=-1.0))          # strongest painted last
    for k in order:
        if not np.isfinite(w[k, 0]):
            continue
        i, j = st_ij[k]
        m = np.hypot(yy - i, xx - j) <= radius_cells
        for c in range(3):
            out[c][m] = np.nan_to_num(w[k, c]) / WIND_SCALE[c]
        out[3][m] = 1.0
    return np.clip(out, -3.0, 3.0)


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


def label_variant(cfg: dict) -> str | None:
    """label.variant: None (default) = the strict rule (dust report confirmed by a warning or a Storm Events entry);
    "station_confirmed" = the strict labels PLUS reports with a real dust code (not haze) and visibility <= 1 mi that
    no warning covered. A parallel label version for one experiment (v18, 20 Nov): the strict rule stays the main one."""
    v = (cfg or {}).get("label", {}).get("variant")
    return str(v) if v else None


def load_confirmed(cfg: dict) -> dict | None:
    """The station-confirmed additions written by scripts/34_add_confirmed_labels.py, or None when label.variant is
    not station_confirmed. Returns {"<split>/<file>": int8 (S, L + 1)}: columns 0..L-1 are the confirmed label at each
    lead, the last column the confirmed report at t0 (1 dust, 0 no dust, -1 no report). Only samples that gain at
    least one positive are stored, so a missing key means "same as the strict labels"."""
    if label_variant(cfg) != "station_confirmed":
        return None
    f = Path(cfg["paths"]["samples"]) / "confirmed_labels.npz"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: run scripts/34_add_confirmed_labels.py before using label.variant station_confirmed")
    with np.load(f, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def apply_confirmed(s: dict, c: np.ndarray | None, radius_cells: int = 3) -> dict:
    """Return the sample with the station-confirmed labels `c` (one entry of load_confirmed) merged in: station rows
    that are 0 under the strict rule but 1 in `c` become 1, their 12 km discs in the pixel target are repainted, and
    the current-report column `st_now` (an INPUT and the genesis definition) is raised the same way. Rows without a
    report (-1) stay -1; nothing is ever lowered. c None = unchanged."""
    if c is None:
        return s
    add = (c[:, :-1] == 1) & (s["st_y"] == 0)
    add_now = (c[:, -1] == 1) & (s["st_now"] == 0)
    if not add.any() and not add_now.any():
        return s
    s = dict(s)
    st_y, st_now = s["st_y"].copy(), s["st_now"].copy()
    st_y[add] = 1
    st_now[add_now] = 1
    if add.any():
        y, w = s["y"].copy(), s["w"].copy()
        yy, xx = np.mgrid[0:y.shape[1], 0:y.shape[2]]
        for k, li in zip(*np.nonzero(add)):
            m = np.hypot(yy - int(s["st_ij"][k, 0]), xx - int(s["st_ij"][k, 1])) <= radius_cells
            y[li][m] = 1
            w[li][m] = np.maximum(w[li][m], 1.0)
        s["y"], s["w"] = y, w
    s["st_y"], s["st_now"] = st_y, st_now
    s["genesis"] = np.array(not bool((st_now == 1).any()))
    return s


def onset_times(event, cfg: dict, step_min: int = 5) -> list[dt.datetime]:
    """EXTRA forecast times at `step_min` cadence through the onset window only (first-warning - hours_before ...
    first warning + 1 h), skipping the times sample_times() already covers. GOES CONUS scans every 5 min, so the
    in-between frames are real new observations; the labels are re-derived from the 5-min station reports.
    Why: genesis positives (the primary metric) live in this window and are the scarcest thing in the data;
    doubling the cadence here roughly doubles them (neighbouring samples are correlated, so less than 2x in effect)."""
    T = cfg["time"]
    base = set(sample_times(event, cfg))
    start = pd.Timestamp(event.t_start).floor(f"{step_min}min") - pd.Timedelta(hours=T["hours_before"])
    end = pd.Timestamp(event.t_start).ceil(f"{step_min}min") + pd.Timedelta(hours=1)
    n = int((end - start) / pd.Timedelta(minutes=step_min)) + 1
    out = [(start + pd.Timedelta(minutes=step_min * k)).to_pydatetime() for k in range(n)]
    return [t for t in out if t not in base]


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
