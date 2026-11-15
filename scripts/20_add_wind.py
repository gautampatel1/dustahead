"""Step 20: add STATION WIND to every stored sample (inputs at t0 and targets at t0 + lead), in place.

    python -u scripts/20_add_wind.py [--workers 6] [--limit N] [--force]

Why: a dust storm is a thunderstorm outflow (or a synoptic wind) crossing erodible ground. The airport stations
(ASOS, Automated Surface Observing System) report wind every 5-20 minutes whether or not dust lifts, so "strong wind
arrives" is observed about 36 times more often than "dust arrives" (13,596 wind onsets against 379 dust onsets in
the 544 event days). The wind head (loss.w_wind) learns the outflow from those; the gate decides about the dust.

Per sample, for the ASOS stations of its station table (other sources carry no wind here; NaN):
  st_wnow (S, 3) float32   peak wind at t0 = max(sustained, gust) in knots from the newest report in (t0 - 30 min, t0],
                           its change against the newest report in (t0 - 75 min, t0 - 20 min], and the temperature
                           change (deg F) over the same pair. Only reports valid at or before t0 (hard rule 3).
  st_wy   (S, L) float32   highest peak wind reported within +-10 min of t0 + lead, knots. NaN = no report.
Source: data/labels/events/<event>.csv (IEM ASOS reports already downloaded by step 1). Resumable.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_G: dict = {}


def _init(cfg_path, force):
    from dustahead import load_config
    cfg = load_config(cfg_path)
    L = Path(cfg["paths"]["labels"])
    st = pd.read_csv(L / ("stations_ext.csv" if (L / "stations_ext.csv").exists() else "stations.csv"))
    if "source" in st:
        st = st[st.source == "asos"]
    cell = {}
    for r in st.itertuples():
        cell.setdefault((int(r.i), int(r.j)), []).append(r.sid)
    _G.update(cfg=cfg, labels=L, cell=cell, force=force, ev={})


def _event_series(event: str) -> dict:
    """station id -> (times int64 ns, peak wind kt, temperature F), sorted by time; cached per worker."""
    if event not in _G["ev"]:
        _G["ev"].clear()                       # samples arrive grouped by event: keep one event in memory
        f = _G["labels"] / "events" / f"{event}.csv"
        d = pd.read_csv(f, usecols=["station", "valid", "sknt", "gust", "tmpf"])
        d["t"] = pd.to_datetime(d.valid, utc=True).astype("int64")
        d["pk"] = d[["sknt", "gust"]].max(axis=1)
        d = d[np.isfinite(d.pk)].sort_values("t")
        _G["ev"][event] = {s: (g.t.values, g.pk.values.astype(np.float32), g.tmpf.values.astype(np.float32))
                           for s, g in d.groupby("station")}
    return _G["ev"][event]


def station_wind(series: dict, sids, t0_ns: int, leads_min) -> tuple[np.ndarray, np.ndarray]:
    """One station (possibly several ids in the same grid cell: the windiest wins) -> (now[3], target[L])."""
    MIN = 60 * 10 ** 9
    now = np.full(3, np.nan, np.float32)
    tgt = np.full(len(leads_min), np.nan, np.float32)
    for sid in sids:
        if sid not in series:
            continue
        t, pk, tf = series[sid]
        k = np.searchsorted(t, t0_ns, side="right") - 1                 # newest report at or before t0
        if k >= 0 and t0_ns - t[k] < 30 * MIN and not (pk[k] <= now[0]):
            now[:] = (pk[k], np.nan, np.nan)
            k0 = np.searchsorted(t, t0_ns - 20 * MIN, side="right") - 1  # newest report at least 20 min older
            if k0 >= 0 and k0 != k and t0_ns - t[k0] < 75 * MIN:
                now[1] = pk[k] - pk[k0]
                now[2] = tf[k] - tf[k0]
        for li, lead in enumerate(leads_min):
            tt = t0_ns + lead * MIN
            a, b = np.searchsorted(t, tt - 10 * MIN, side="left"), np.searchsorted(t, tt + 10 * MIN, side="right")
            if b > a:
                tgt[li] = np.nanmax([tgt[li], pk[a:b].max()])
    return now, tgt


def add_one(path):
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as z:
            if "st_wy" in z.files and not _G["force"]:
                return "skip"
            s = {k: z[k] for k in z.files}
        event = path.stem.rsplit("_", 1)[0]
        series = _event_series(event)
        t0_ns = int(pd.Timestamp(s["t0"][()]).tz_localize("UTC").value)
        leads = _G["cfg"]["time"]["leads_min"]
        S = len(s["st_ij"])
        src = s["st_src"] if "st_src" in s else np.zeros(S, int)
        wnow = np.full((S, 3), np.nan, np.float32)
        wy = np.full((S, len(leads)), np.nan, np.float32)
        for k in range(S):
            if src[k] != 0:
                continue
            sids = _G["cell"].get((int(s["st_ij"][k, 0]), int(s["st_ij"][k, 1])), ())
            wnow[k], wy[k] = station_wind(series, sids, t0_ns, leads)
        s["st_wnow"], s["st_wy"] = wnow, wy
        tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, **s)
        os.replace(tmp, path)
        return f"ok {int(np.isfinite(wnow[:, 0]).sum())} now, {int(np.isfinite(wy).any(1).sum())} targets"
    except Exception as e:  # noqa: BLE001
        return f"{path.name}: {type(e).__name__}: {str(e)[:160]}"


def main():
    from dustahead import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    files = sorted(Path(cfg["paths"]["samples"]).glob("*/*.npz"), key=lambda p: p.stem)   # grouped by event
    files = [f for f in files if ".tmp" not in f.name][: a.limit]
    print(f"{len(files)} samples", flush=True)
    stats: dict = {}
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.config, a.force)) as ex:
        for n, r in enumerate(ex.map(add_one, files, chunksize=64), 1):
            key = r.split(" ")[0] if r.startswith(("ok", "skip")) else "error"
            stats[key] = stats.get(key, 0) + 1
            if key == "error" and stats[key] <= 20:
                print("  ", r, flush=True)
            if n % 2000 == 0:
                print(f"  {n}/{len(files)} {stats}", flush=True)
    print("done", stats, flush=True)


if __name__ == "__main__":
    main()
