"""Synoptic Data (formerly MesoWest) API: EXTRA ground truth beyond the ASOS/AWOS network.

What it adds inside the DustAhead domain (AZ + NM):
  * ADOT / NMDOT road-weather stations (RWIS, network "ADOT"/"NMDOT"): forward-scatter visibility sensors ON THE
    HIGHWAY, including Interstate 10 - the road the product warns about, where ASOS has no stations.
  * PM10 monitors (ADEQ, Maricopa County AQ, Pinal County, EPA AirNow via Synoptic): a dust storm is a PM10 event
    (haboobs read 500-5,000 ug/m3 hourly) - a label source that does not depend on a visibility sensor at all.
  * Hundreds of other mesonet stations with wind and humidity (useful later as inputs; not labels).

Label rules for these stations (an EXTENSION of the ASOS rule in events/stations.py - the warning cross-check is
kept, the dust code is replaced by what the sensor can say):
  RWIS:  visibility <= 1 mile AND inside a Dust Storm Warning window/polygon (or a Storm Events entry)
         AND no precipitation reported by the station's own precip sensor in the same interval
         (RWIS has no present-weather code, so rain-only low visibility must be excluded explicitly).
  PM10:  hourly PM10 >= cfg.label.pm10_ugm3 (default 250 ug/m3) AND inside a warning window/polygon.
         Severe when PM10 >= 4 x that threshold (one-hour haboob peaks are typically 1,000-5,000 ug/m3).
Both are recorded with `source` = "rwis" | "pm10" so every result can be reported with and without them.

Access: token in <repo>/.secrets/synoptic_token.txt (never committed) or env SYNOPTIC_TOKEN. Free academic tier.
API: https://api.synopticdata.com/v2/stations/{metadata,timeseries}; times UTC ("obtimezone=UTC"), units metric.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API = "https://api.synopticdata.com/v2/stations"
VARS = ("visibility", "PM_10_concentration", "precip_accum_one_hour", "wind_speed", "wind_gust", "relative_humidity")


def token(root: str | Path | None = None) -> str:
    t = os.environ.get("SYNOPTIC_TOKEN")
    if t:
        return t.strip()
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    f = root / ".secrets" / "synoptic_token.txt"
    if f.exists():
        return f.read_text().strip()
    raise RuntimeError("no Synoptic token: set SYNOPTIC_TOKEN or write .secrets/synoptic_token.txt")


def station_inventory(bbox: tuple[float, float, float, float], tok: str | None = None) -> pd.DataFrame:
    """All active stations in (lon_min, lat_min, lon_max, lat_max) that report ANY of VARS, with period of record,
    network and which of the variables they carry. One row per station."""
    tok = tok or token()
    r = requests.get(f"{API}/metadata", params=dict(token=tok, bbox=",".join(map(str, bbox)), vars=",".join(VARS),
                                                    varsoperator="or", sensorvars=1, complete=1, status="active"), timeout=120)
    r.raise_for_status()
    js = r.json()
    if js.get("SUMMARY", {}).get("RESPONSE_CODE") != 1:
        raise RuntimeError(js.get("SUMMARY"))
    rows = []
    for s in js.get("STATION", []):
        sv = s.get("SENSOR_VARIABLES", {}) or {}
        por = s.get("PERIOD_OF_RECORD", {}) or {}
        rows.append(dict(stid=s["STID"], name=s.get("NAME"), network=s.get("SHORTNAME") or s.get("MNET_ID"),
                         lon=float(s["LONGITUDE"]), lat=float(s["LATITUDE"]), elevation=s.get("ELEVATION"),
                         county=s.get("COUNTY"), state=s.get("STATE"),
                         por_start=por.get("start"), por_end=por.get("end"),
                         has_vis="visibility" in sv, has_pm10="PM_10_concentration" in sv,
                         has_precip="precip_accum_one_hour" in sv, has_wind="wind_speed" in sv))
    return pd.DataFrame(rows)


def timeseries(stids: list[str], start: pd.Timestamp, end: pd.Timestamp, variables=("visibility", "PM_10_concentration", "precip_accum_one_hour"),
               tok: str | None = None) -> pd.DataFrame:
    """Observations for stations over a UTC window, long format: stid, valid (UTC), variable, value (metric units:
    visibility in statute miles? -> Synoptic returns km unless units=... we request units=english for miles)."""
    tok = tok or token()
    out = []
    for k in range(0, len(stids), 100):           # the API caps stid lists; 100 per request is safe
        chunk = stids[k:k + 100]
        r = requests.get(f"{API}/timeseries", params=dict(token=tok, stid=",".join(chunk), start=start.strftime("%Y%m%d%H%M"),
                                                            end=end.strftime("%Y%m%d%H%M"), vars=",".join(variables), obtimezone="UTC",
                                                            units="english", hfmetars=1), timeout=180)
        r.raise_for_status()
        js = r.json()
        for s in js.get("STATION", []):
            obs = s.get("OBSERVATIONS", {}) or {}
            times = obs.get("date_time")
            if not times:
                continue
            for key, vals in obs.items():
                if key == "date_time":
                    continue
                var = key.rsplit("_set_", 1)[0]
                for t, v in zip(times, vals):
                    if v is not None:
                        out.append((s["STID"], t, var, float(v)))
    df = pd.DataFrame(out, columns=["stid", "valid", "variable", "value"])
    if len(df):
        df["valid"] = pd.to_datetime(df.valid, utc=True)
    return df


# ---- extended station table and label rule ----------------------------------------------------------------------
def extended_station_table(inventory: pd.DataFrame, asos: pd.DataFrame, grid, i10_km: float | None = None) -> pd.DataFrame:
    """Non-ASOS stations that can label dust, in the stations.csv layout plus `source`:
      rwis      road-weather visibility (AZDOT / NMDOT / CALTRANS)
      vis_other any other visibility sensor (WSMR, YPG, CWOP, ...)
      pm10      PM10 monitors (AirNow, NMAQ)
    A station already in the ASOS table (STID = 'K' + sid, or the same sid) is skipped - ASOS keeps precedence."""
    asos_ids = set(asos.sid) | {"K" + s for s in asos.sid}
    rows = []
    for r in inventory.itertuples():
        if r.stid in asos_ids or not grid.inside(r.lon, r.lat):
            continue
        if i10_km is not None and getattr(r, "km_to_i10", 0) > i10_km:
            continue
        if r.has_pm10:
            src = "pm10"
        elif r.has_vis:
            src = "rwis" if str(r.network).upper() in ("AZDOT", "NMDOT", "CALTRANS", "TXDOT") else "vis_other"
        else:
            continue
        rows.append(dict(sid=r.stid, name=r.name, lon=r.lon, lat=r.lat, network=r.network, wfo=None,
                         ugc_county=None, county=getattr(r, "county", None), elevation=r.elevation, source=src))
    df = pd.DataFrame(rows).drop_duplicates("sid").reset_index(drop=True)
    if len(df):
        df["i"], df["j"] = grid.to_ij(df.lon.values, df.lat.values)
    df["i10_core"] = False
    return df


def label_extended(ts: pd.DataFrame, stations_ext: pd.DataFrame, warnings: pd.DataFrame, cfg: dict,
                   polygons: dict | None = None, storm_events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Apply the extended label rule to a Synoptic time series (long format from timeseries()).
    Returns the same columns station_targets() needs (station, valid, label, severe, tol_min) plus source and the
    warning flags, so the rows can be concatenated with the ASOS labels of the same event."""
    from .stations import warning_flags
    L = cfg["label"]
    if ts.empty:
        return pd.DataFrame(columns=["station", "valid", "label", "severe", "source", "tol_min"])
    wide = ts.pivot_table(index=["stid", "valid"], columns="variable", values="value", aggfunc="first").reset_index()
    wide = wide.rename(columns={"stid": "station", "PM_10_concentration": "pm10", "precip_accum_one_hour": "precip1h"})
    for c in ("visibility", "pm10", "precip1h"):
        if c not in wide:
            wide[c] = float("nan")
    src = stations_ext.set_index("sid").source
    wide = wide[wide.station.isin(src.index)]
    wide["source"] = wide.station.map(src)
    wide = warning_flags(wide, warnings, stations_ext, cfg, polygons, storm_events)
    confirmed = wide.in_warning | wide.in_storm_event
    pm_thr, pm_sev = float(L.get("pm10_ugm3", 250)), float(L.get("pm10_severe_ugm3", 1000))
    vis_ok = wide.visibility <= L["vis_max_mi"]
    no_rain = ~(wide.precip1h > 0)                       # RWIS has no present-weather code: exclude measured rain
    is_pm = wide.source == "pm10"
    lab_vis = vis_ok & no_rain & confirmed & ~is_pm
    lab_pm = (wide.pm10 >= pm_thr) & confirmed & is_pm
    wide["label"] = (lab_vis | lab_pm).astype(int)
    wide["severe"] = ((lab_vis & (wide.visibility <= L["severe_vis_mi"])) | (lab_pm & (wide.pm10 >= pm_sev))).astype(int)
    wide["tol_min"] = np.where(is_pm, 30, 5)             # hourly PM10 value stands for its hour
    # a visibility station with no visibility value at that time, or a PM10 station with no PM10, is "no report"
    has_obs = np.where(is_pm, wide.pm10.notna(), wide.visibility.notna())
    return wide[has_obs].reset_index(drop=True)
