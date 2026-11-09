"""ASOS / AWOS surface stations: the ground truth for DustAhead.

Station label rule (per report, per station):
    visibility <= 1 mile
    AND a dust code present in the present-weather string (DS, DU, BLDU, SS, SA, ... ; HZ counts only with a warning)
    AND the report time lies inside a Dust Storm Warning window (±pad) whose polygon contains the station
        or whose text names the station's county (in_warning_strict records the polygon-only version),
        OR inside an NCEI Storm Events "Dust Storm" entry for that county.
    visibility <= 1/4 mile is flagged 'severe'.
    Rain codes (RA, TSRA, ...) are RECORDED (rain_too) but do not exclude: the 25 Aug 2025 Phoenix
    haboob reads "-TSRA +DS" at 0.00 mi. Rain-only low visibility is excluded because it has no dust code.

Why the warning cross-check: automated ASOS sensors measure visibility but cannot identify dust;
the DS/DU codes come from human augmentation or present-weather logic that can confuse dust with
haze or smoke. Requiring an NWS warning or Storm Events entry removes most of those errors.

Data: IEM ASOS download service, https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py
  report_type: 1 = 5-minute ASOS (HFMETAR), 2 = routine hourly, 3 = specials. We take all three
  (verified Oct 2026: PHX 2025-08-26 00:35Z returns "-RA +DS" 0.00 mi from type 1 and "-TSRA +DS" from type 3).
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..grid import Grid

IEM = "https://mesonet.agron.iastate.edu"
NETWORKS = ("AZ_ASOS", "NM_ASOS", "TX_ASOS", "CA_ASOS")  # TX for El Paso, CA for Blythe (both on I-10)
ASOS_FIELDS = ("vsby", "wxcodes", "sknt", "gust", "drct", "tmpf", "relh", "p01i")

# Stations along or near Interstate 10 (used when the network download is unavailable)
I10_CORE = {
    "BLH": (-114.72, 33.62), "YUM": (-114.61, 32.66), "GBN": (-112.72, 32.95), "BXK": (-112.69, 33.42),
    "GYR": (-112.38, 33.42), "LUF": (-112.38, 33.54), "PHX": (-112.01, 33.43), "CHD": (-111.81, 33.27),
    "IWA": (-111.66, 33.31), "A39": (-112.03, 33.00), "CGZ": (-111.77, 32.95), "P08": (-111.43, 32.94),
    "MZJ": (-111.33, 32.51), "AVQ": (-111.22, 32.41), "RYN": (-111.17, 32.14), "TUS": (-110.94, 32.13),
    "DMA": (-110.88, 32.17), "P33": (-109.90, 32.25), "SAD": (-109.64, 32.85), "FHU": (-110.34, 31.59),
    "DUG": (-109.60, 31.47), "OLS": (-110.85, 31.42), "LSB": (-108.69, 32.33), "DMN": (-107.72, 32.26),
    "SVC": (-108.16, 32.64), "TCS": (-107.27, 33.24), "LRU": (-106.92, 32.29), "ELP": (-106.38, 31.81),
    "ALM": (-105.99, 32.84), "ABQ": (-106.61, 35.04), "ROW": (-104.53, 33.31), "CNM": (-104.26, 32.34),
}


def fetch_station_table(grid: Grid, networks=NETWORKS) -> pd.DataFrame:
    """All ASOS stations inside the grid, with lon/lat, county UGC and WFO."""
    rows = []
    for net in networks:
        try:
            js = requests.get(f"{IEM}/geojson/network/{net}.geojson", timeout=60).json()
        except Exception:  # noqa: BLE001
            continue
        for f in js.get("features", []):
            lon, lat = f["geometry"]["coordinates"]
            p = f["properties"]
            if grid.inside(lon, lat) and p.get("online", True):
                rows.append(dict(sid=p["sid"], name=p.get("sname"), lon=lon, lat=lat, network=net,
                                 wfo=p.get("wfo"), ugc_county=p.get("ugc_county"), county=p.get("county"),
                                 elevation=p.get("elevation")))
    if not rows:  # offline fallback
        rows = [dict(sid=k, name=k, lon=v[0], lat=v[1], network=None, wfo=None, ugc_county=None, county=None,
                     elevation=None) for k, v in I10_CORE.items()]
    df = pd.DataFrame(rows).drop_duplicates("sid").reset_index(drop=True)
    df["i"], df["j"] = grid.to_ij(df.lon.values, df.lat.values)
    df["i10_core"] = df.sid.isin(I10_CORE)
    return df


def fetch_asos(stations: list[str], start: pd.Timestamp, end: pd.Timestamp, report_types=(1, 2, 3)) -> pd.DataFrame:
    """ASOS reports (5-minute + routine + specials) for stations over a UTC range. One row per (station, time)."""
    params = {"station": ",".join(stations), "data": ",".join(ASOS_FIELDS), "tz": "Etc/UTC", "format": "onlycomma",
              "latlon": "no", "missing": "empty", "trace": "0.0001",
              "report_type": ",".join(str(r) for r in report_types),
              "sts": start.strftime("%Y-%m-%dT%H:%MZ"), "ets": end.strftime("%Y-%m-%dT%H:%MZ")}
    r = requests.get(f"{IEM}/cgi-bin/request/asos.py", params=params, timeout=300)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if df.empty or "valid" not in df.columns:
        raise RuntimeError(f"IEM ASOS returned no data for {stations[:3]}... {start}-{end}: {r.text[:200]}")
    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    for c in ASOS_FIELDS:
        if c != "wxcodes":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["wxcodes"] = df["wxcodes"].fillna("").astype(str)
    # 5-minute and special reports can share a minute: keep the row with the richest weather string
    df["_n"] = df.wxcodes.str.len()
    df = df.sort_values(["station", "valid", "_n"]).drop_duplicates(["station", "valid"], keep="last").drop(columns="_n")
    return df.reset_index(drop=True)


_CODE_RE = re.compile(r"[A-Z]{2,6}")


def has_code(wx: str, codes) -> bool:
    toks = set(_CODE_RE.findall(wx.upper()))
    expanded = set(toks)
    for t in toks:                      # split compound tokens: VCBLDU -> VC, BL, DU ; TSRA -> TS, RA
        for k in range(0, len(t) - 1, 2):
            expanded.add(t[k:k + 2])
    return any(c in expanded or c in toks for c in codes)


def _to_utc(col: pd.Series) -> pd.Series:
    """Timestamps as tz-aware UTC whether the source was naive (assumed UTC) or aware."""
    col = pd.to_datetime(col)
    return col.dt.tz_localize("UTC") if col.dt.tz is None else col.dt.tz_convert("UTC")


def warning_flags(obs: pd.DataFrame, warnings: pd.DataFrame, stations: pd.DataFrame, cfg: dict,
                  polygons: dict | None = None, storm_events: pd.DataFrame | None = None) -> pd.DataFrame:
    """The warning cross-check shared by every label source: adds in_warning_strict (polygon hit), in_warning
    (polygon hit OR the station's county named in a concurrent Dust Storm Warning) and in_storm_event (NCEI Storm
    Events dust entry for the county) to a table with columns station, valid."""
    from .warnings import point_in_rings

    L = cfg["label"]
    pad = pd.Timedelta(minutes=L["warning_pad_min"])
    obs = obs.copy()
    obs["valid"] = _to_utc(obs.valid)
    warnings = warnings.copy()
    if len(warnings):
        warnings["issue"], warnings["expire"] = _to_utc(warnings.issue), _to_utc(warnings.expire)
    if storm_events is not None and len(storm_events):
        storm_events = storm_events.copy()
        storm_events["begin"], storm_events["end"] = _to_utc(storm_events.begin), _to_utc(storm_events["end"])
    ds = warnings[warnings.phenomena == "DS"] if not warnings.empty else warnings
    st = stations.set_index("sid")
    in_w = np.zeros(len(obs), bool)          # polygon hit (strict)
    in_wc = np.zeros(len(obs), bool)         # county named in a concurrent warning (relaxed)
    in_se = np.zeros(len(obs), bool)
    for k, (sid, t) in enumerate(zip(obs.station, obs.valid)):
        if sid not in st.index:
            continue
        lon, lat, county = st.loc[sid, "lon"], st.loc[sid, "lat"], st.loc[sid, "county"]
        cand = ds[(ds.issue - pad <= t) & (ds.expire + pad >= t)] if len(ds) else ds
        for _, w in cand.iterrows():
            key = (w.wfo, w.year, w.eventid)
            if polygons and polygons.get(key) is not None and point_in_rings(lon, lat, polygons[key]):
                in_w[k] = True
            if not county or str(county).split()[0] in str(w.locations):
                in_wc[k] = True
            if in_w[k]:
                break
        if storm_events is not None and len(storm_events) and county:
            se = storm_events[(storm_events.begin - pad <= t) & (storm_events.end + pad >= t)]
            in_se[k] = bool(se.cz_name.str.upper().str.contains(str(county).split()[0].upper()).any())
    obs["in_warning_strict"] = in_w
    obs["in_warning"] = in_w | in_wc          # polygon hit, or the station's county is under a concurrent warning
    obs["in_storm_event"] = in_se
    return obs


def label_reports(obs: pd.DataFrame, warnings: pd.DataFrame, stations: pd.DataFrame, cfg: dict,
                  polygons: dict | None = None, storm_events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Apply the station label rule to every ASOS report.

    polygons: optional {(wfo, year, eventid): rings} from warnings.fetch_polygon; without one the check falls
    back to the warning's county list, then to time alone.
    storm_events: optional NCEI Storm Events rows (see events/storm_events.py) as a second cross-check.
    Adds: dust_code, rain_too, vis_ok, in_warning, in_storm_event, label, severe.
    """
    L = cfg["label"]
    obs = obs.copy()
    obs["dust_code"] = obs.wxcodes.map(lambda s: has_code(s, L["dust_codes"]))
    obs["haze_only"] = obs.wxcodes.map(lambda s: has_code(s, ["HZ"])) & ~obs.wxcodes.map(lambda s: has_code(s, [c for c in L["dust_codes"] if c != "HZ"]))
    obs["rain_too"] = obs.wxcodes.map(lambda s: has_code(s, L["rain_codes"]))
    obs["vis_ok"] = obs.vsby <= L["vis_max_mi"]
    obs["severe_vis"] = obs.vsby <= L["severe_vis_mi"]

    obs = warning_flags(obs, warnings, stations, cfg, polygons, storm_events)
    confirmed = obs.in_warning | obs.in_storm_event
    obs["label"] = (obs.vis_ok & obs.dust_code & confirmed).astype(int)
    obs["severe"] = (obs.label.astype(bool) & obs.severe_vis).astype(int)
    return obs


def station_targets(labels: pd.DataFrame, stations: pd.DataFrame, t0: pd.Timestamp, leads_min, tol_min: int = 5) -> pd.DataFrame:
    """For one forecast time t0, the label at each station at t0 + lead (nearest report within ±tol)."""
    rows = []
    sids = set(stations.sid)
    for sid, grp in labels.groupby("station"):
        if sid not in sids:
            continue
        row = dict(station=sid)
        tol = float(grp.tol_min.iloc[0]) if "tol_min" in grp else tol_min     # hourly sources (PM10) use 30
        for lead in leads_min:
            tt = t0 + pd.Timedelta(minutes=lead)
            near = grp.iloc[(grp.valid - tt).abs().argsort()[:1]]
            ok = (not near.empty) and abs((near.valid.iloc[0] - tt).total_seconds()) <= tol * 60
            row[f"y_{lead}"] = int(near.label.iloc[0]) if ok else -1   # -1 = no report (masked in loss)
        near0 = grp.iloc[(grp.valid - t0).abs().argsort()[:1]]
        ok0 = (not near0.empty) and abs((near0.valid.iloc[0] - t0).total_seconds()) <= tol * 60
        row["dust_now"] = int(near0.label.iloc[0]) if ok0 else -1
        rows.append(row)
    return pd.DataFrame(rows)


def station_wind_targets(obs: pd.DataFrame, sids, t0, leads_min, cfg: dict, tol_min: int = 5) -> np.ndarray:
    """(S, L) int8 OUTFLOW targets for the sky branch: 1 when the report nearest t0 + lead (within ±tol_min) has
    sustained wind or gust >= label.outflow_kt, 0 when it is below, -1 when there is no wind reading (or the table has
    no wind columns, e.g. the extended Synoptic stations). ASOS sknt is the 2-min mean wind in knots, gust the peak.
    Why a wind target: a haboob needs an outflow AND erodible ground; dust reports say both happened, wind reports
    say the outflow did - on every gusty afternoon, dust or not - so the sky factor sigmoid(z) can be taught
    P(outflow arrives) directly. These are targets, so reports after t0 are fine."""
    out = np.full((len(sids), len(leads_min)), -1, np.int8)
    if obs.empty or "sknt" not in obs:
        return out
    t0 = pd.Timestamp(t0)
    t0 = (t0.tz_convert("UTC") if t0.tzinfo else t0.tz_localize("UTC")).tz_localize(None)
    sknt = pd.to_numeric(obs["sknt"], errors="coerce").to_numpy(float)
    gust = pd.to_numeric(obs["gust"], errors="coerce").to_numpy(float) if "gust" in obs else np.full(len(obs), np.nan)
    wind = np.fmax(sknt, gust)                    # NaN only when both are missing
    df = pd.DataFrame(dict(station=obs["station"].astype(str).to_numpy(), wind=wind,
                           valid=pd.to_datetime(obs["valid"], utc=True).dt.tz_localize(None).to_numpy()))
    df = df[np.isfinite(df.wind)]
    by = {k: g for k, g in df.groupby("station")}
    thr, tol = float(cfg["label"].get("outflow_kt", 30)), pd.Timedelta(minutes=tol_min)
    for k, sid in enumerate(sids):
        g = by.get(str(sid))
        if g is None:
            continue
        for l, lead in enumerate(leads_min):
            gap = (g.valid - (t0 + pd.Timedelta(minutes=lead))).abs()
            n = gap.idxmin()
            if gap[n] <= tol:
                out[k, l] = int(g.wind[n] >= thr)
    return out


def save_labels(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path) if str(path).endswith(".parquet") else df.to_csv(path, index=False)
