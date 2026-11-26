"""Step 22: add the BLOWING-DUST observation to every stored sample (targets at t0 + lead, state at t0), in place.

    python -u scripts/22_add_dust_tiers.py [--workers 6] [--limit N] [--force]

Why: the strict label (visibility <= 1 mi AND a dust code AND a warning / Storm Events cross-check; hard rule 1) is
the benchmark and stays untouched, but it has only 379 onsets in the 544 event days. The same ASOS reports (Automated
Surface Observing System) carry a dust weather code (DS, DU, BLDU ...) 35,531 times; with visibility <= 3 mi that is
15,643 reports and 2,490 onsets. Of the strong-wind reports that a strict-label model must call "no dust", 17 % carry
a dust code. This step stores the observation so a second head can learn the lower-severity event and a second
product tier ("blowing dust, reduced visibility") can be verified. Nothing is hand-drawn: it is the station's own code.

Per sample, for the ASOS stations of its station table (other sources: NaN):
  st_dvis     (S, L) float32   nearest report within +-5 min of t0 + lead (as the strict labels): its visibility in
                               miles if it carries a dust code, 99 if it does not, NaN if there is no report.
  st_dvis_now (S,)   float32   the same for the newest report in (t0 - 20 min, t0].
Rain in the same report never excludes (rule 1: the Phoenix 2025 haboob reads `-TSRA +DS`). Resumable.
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
MIN = 60 * 10 ** 9
CLEAR = 99.0


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
    """station id -> (times int64 ns, dust visibility: miles if dust-coded else 99), sorted by time."""
    if event not in _G["ev"]:
        _G["ev"].clear()
        f = _G["labels"] / "events" / f"{event}.csv"
        d = pd.read_csv(f, usecols=["station", "valid", "vsby", "dust_code"], dtype={"station": str})
        d["t"] = pd.to_datetime(d.valid, utc=True).dt.as_unit("ns").astype("int64")
        dc = d.dust_code.astype(str).str.lower().isin(("true", "1", "1.0"))
        d["dv"] = np.where(dc, d.vsby.fillna(CLEAR), CLEAR).astype(np.float32)
        d = d.sort_values("t")
        _G["ev"][event] = {s: (g.t.values, g.dv.values) for s, g in d.groupby("station")}
    return _G["ev"][event]


def station_dust(series: dict, sids, t0_ns: int, leads_min, tol_min: int = 5) -> tuple[float, np.ndarray]:
    """One station (several ids in one grid cell: the lowest dust visibility wins) -> (now, target[L])."""
    now = np.nan
    tgt = np.full(len(leads_min), np.nan, np.float32)
    for sid in sids:
        if sid not in series:
            continue
        t, dv = series[sid]
        k = np.searchsorted(t, t0_ns, side="right") - 1                 # newest report at or before t0
        if k >= 0 and t0_ns - t[k] < 20 * MIN:
            now = float(np.nanmin([now, dv[k]]))
        for li, lead in enumerate(leads_min):
            tt = t0_ns + lead * MIN
            j = np.searchsorted(t, tt)
            cand = [c for c in (j - 1, j) if 0 <= c < len(t) and abs(int(t[c]) - tt) <= tol_min * MIN]
            if cand:
                c = min(cand, key=lambda q: abs(int(t[q]) - tt))
                tgt[li] = np.nanmin([tgt[li], dv[c]])
    return now, tgt


def add_one(path):
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as z:
            if "st_dvis" in z.files and not _G["force"]:
                return "skip"
            s = {k: z[k] for k in z.files}
        series = _event_series(path.stem.rsplit("_", 1)[0])
        t0_ns = int(pd.Timestamp(s["t0"][()]).tz_localize("UTC").as_unit("ns").value)
        leads = _G["cfg"]["time"]["leads_min"]
        S = len(s["st_ij"])
        src = s["st_src"] if "st_src" in s else np.zeros(S, int)
        now = np.full(S, np.nan, np.float32)
        dv = np.full((S, len(leads)), np.nan, np.float32)
        for k in range(S):
            if src[k] != 0:
                continue
            sids = _G["cell"].get((int(s["st_ij"][k, 0]), int(s["st_ij"][k, 1])), ())
            now[k], dv[k] = station_dust(series, sids, t0_ns, leads)
        s["st_dvis"], s["st_dvis_now"] = dv, now
        tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, **s)
        os.replace(tmp, path)
        return f"ok {int(np.isfinite(dv).any(1).sum())} targets, {int((dv <= 3).any(1).sum())} blowing dust"
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
    files = sorted(Path(cfg["paths"]["samples"]).glob("*/*.npz"), key=lambda p: p.stem)
    files = [f for f in files if ".tmp" not in f.name][: a.limit]
    print(f"{len(files)} samples", flush=True)
    stats, n_tier, n_tgt = {}, 0, 0
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.config, a.force)) as ex:
        for n, r in enumerate(ex.map(add_one, files, chunksize=64), 1):
            key = r.split(" ")[0] if r.startswith(("ok", "skip")) else "error"
            stats[key] = stats.get(key, 0) + 1
            if key == "ok":
                n_tgt += int(r.split()[1]); n_tier += int(r.split()[3])
            if key == "error" and stats[key] <= 20:
                print("  ", r, flush=True)
            if n % 2000 == 0:
                print(f"  {n}/{len(files)} {stats}", flush=True)
    print("done", stats, f"station rows with a target {n_tgt}, with blowing dust (<= 3 mi) {n_tier}", flush=True)
    sys.exit(0 if stats.get("error", 0) == 0 and (n_tier > 0 or stats.get("ok", 0) == 0) else 1)


if __name__ == "__main__":
    main()
