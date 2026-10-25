"""NWS warnings from the Iowa Environmental Mesonet (IEM) VTEC archive.

Used for two things:
1. The EVENT LIST: each day with a Dust Storm Warning (DS.W) in AZ/NM is a dust event.
   Days with a Severe Thunderstorm Warning (SV.W) but no DS.W within ±1 day are "null days"
   (storm, no dust) - the hard negatives the model must learn to reject.
2. LABEL CROSS-CHECK: a station dust report only counts if it falls inside a warning window
   and polygon (ASOS sensors cannot name dust by themselves; see events/stations.py).

Endpoints (verified Oct 2026):
  https://mesonet.agron.iastate.edu/json/vtec_events_bystate.py?state=AZ&year=2025&phenomena=DS&significance=W
  https://mesonet.agron.iastate.edu/geojson/vtec_event.py?wfo=PSR&year=2025&phenomena=DS&significance=W&etn=10&sbw=1
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

IEM = "https://mesonet.agron.iastate.edu"
STATES = ("AZ", "NM")


def _get(url: str, params: dict | None = None, retries: int = 3) -> dict:
    for k in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            if k == retries - 1:
                raise
            time.sleep(2 * (k + 1))
    raise RuntimeError("unreachable")


def fetch_warnings(state: str, year: int, phenomena: str = "DS", significance: str = "W") -> pd.DataFrame:
    """All warnings of one type for one state and year. One row per VTEC event."""
    js = _get(f"{IEM}/json/vtec_events_bystate.py",
              dict(state=state, year=year, phenomena=phenomena, significance=significance))
    df = pd.DataFrame(js.get("events", []))
    if df.empty:
        return pd.DataFrame(columns=["state", "wfo", "eventid", "phenomena", "significance", "issue", "expire", "locations"])
    df["issue"] = pd.to_datetime(df["issue"], utc=True)
    df["expire"] = pd.to_datetime(df["expire"], utc=True)
    df["state"] = state
    df["year"] = year
    return df[["state", "year", "wfo", "eventid", "phenomena", "significance", "issue", "expire", "locations", "area"]]


def fetch_polygon(wfo: str, year: int, eventid: int, phenomena: str = "DS", significance: str = "W") -> list | None:
    """Storm-based warning polygon(s) as a list of (lon, lat) rings, or None if unavailable."""
    js = _get(f"{IEM}/geojson/vtec_event.py",
              dict(wfo=wfo, year=year, phenomena=phenomena, significance=significance, etn=eventid, sbw=1))
    rings = []
    for feat in js.get("features", []):
        g = feat.get("geometry") or {}
        if g.get("type") == "Polygon":
            rings.append(np.asarray(g["coordinates"][0], float))
        elif g.get("type") == "MultiPolygon":
            rings.extend(np.asarray(p[0], float) for p in g["coordinates"])
    return rings or None


def point_in_rings(lon: float, lat: float, rings: list) -> bool:
    """Ray-casting point-in-polygon over a list of rings."""
    for ring in rings:
        x, y = ring[:, 0], ring[:, 1]
        inside = False
        j = len(ring) - 1
        for i in range(len(ring)):
            if ((y[i] > lat) != (y[j] > lat)) and (lon < (x[j] - x[i]) * (lat - y[i]) / (y[j] - y[i] + 1e-12) + x[i]):
                inside = not inside
            j = i
        if inside:
            return True
    return False


def fetch_all_warnings(years, states=STATES, phenomena=("DS", "SV")) -> pd.DataFrame:
    frames = []
    for st in states:
        for yr in years:
            for ph in phenomena:
                df = fetch_warnings(st, yr, ph, "W")
                if not df.empty:
                    frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_event_list(warnings: pd.DataFrame, gap_hours: float = 3.0) -> pd.DataFrame:
    """Cluster DS.W warnings into events and pick null days from SV.W.

    An EVENT is a run of dust warnings where consecutive issue times are < gap_hours apart.
    Returns one row per event: t_start, t_end, n_warnings, states, kind ('dust' | 'nodust')  -- not the word "null": pandas reads it back as NaN.
    """
    ds = warnings[warnings.phenomena == "DS"].sort_values("issue")
    events = []
    cur = None
    for _, w in ds.iterrows():
        if cur is None or (w.issue - cur["t_end"]).total_seconds() > gap_hours * 3600:
            if cur is not None:
                events.append(cur)
            cur = dict(t_start=w.issue, t_end=w.expire, n_warnings=1, states={w.state}, wfos={w.wfo})
        else:
            cur["t_end"] = max(cur["t_end"], w.expire)
            cur["n_warnings"] += 1
            cur["states"].add(w.state)
            cur["wfos"].add(w.wfo)
    if cur is not None:
        events.append(cur)
    ev = pd.DataFrame(events)
    if not ev.empty:
        ev["kind"] = "dust"
        ev["states"] = ev.states.map(lambda s: "+".join(sorted(s)))
        ev["wfos"] = ev.wfos.map(lambda s: "+".join(sorted(s)))

    # null days: severe-thunderstorm days with no dust warning within ±1 day
    sv = warnings[warnings.phenomena == "SV"]
    dust_days = set(ds.issue.dt.floor("D")) if not ds.empty else set()
    nulls = []
    for day, grp in sv.groupby(sv.issue.dt.floor("D")):
        near = any(abs((day - d).days) <= 1 for d in dust_days)
        if not near:
            nulls.append(dict(t_start=grp.issue.min(), t_end=grp.expire.max(), n_warnings=len(grp),
                              states="+".join(sorted(grp.state.unique())), wfos="+".join(sorted(grp.wfo.unique())),
                              kind="nodust"))
    out = pd.concat([ev, pd.DataFrame(nulls)], ignore_index=True)
    out["event_id"] = [f"{k}_{r.t_start:%Y%m%d_%H%M}" for k, r in zip(out.kind, out.itertuples())]
    out["year"] = out.t_start.dt.year
    return out.sort_values("t_start").reset_index(drop=True)


def save_event_list(events: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(path, index=False)
