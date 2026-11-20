"""Step 34: the "station-confirmed" label version, as a side file (no sample is rewritten).

    python -u scripts/34_add_confirmed_labels.py [--splits train val]

Strict rule (hard rule 1): visibility <= 1 mi AND a dust code AND a Dust Storm Warning or a Storm Events entry that
covers the report. On 19 Nov we found 267 reports on event days with a real dust code (not haze) and visibility
<= 1 mi that no warning covered, and on 20 Nov 28 more days with none at all. This version counts those as dust too:
    confirmed = strict label  OR  (visibility <= label.vis_max_mi AND a dust code other than HZ (haze))
For every sample it re-derives the station rows from the event's 5-minute ASOS (Automated Surface Observing System)
reports with the same nearest-report rule as events/stations.station_targets (within 5 min of t0 + lead; of t0 for
the current report), and stores ONLY the samples that gain a positive:
    data/samples/confirmed_labels.npz    "<split>/<file>" -> int8 (S, L + 1): labels per lead, then the report at t0
    data/samples/index_confirmed.csv     split, file, cpos15, cpos30, cpos60 (positives per lead after the merge)
The strict labels inside the samples are untouched; the dataset merges this file in only when label.variant is
station_confirmed (sampler.apply_confirmed). The closed test year is not processed.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dustahead import load_config  # noqa: E402
from dustahead.events.stations import has_code  # noqa: E402

MIN = 60 * 10 ** 9


def event_series(f: Path, cfg: dict) -> dict:
    """station id -> (report times int64 ns sorted, confirmed-dust flag per report)."""
    L = cfg["label"]
    real = [c for c in L["dust_codes"] if c != "HZ"]
    d = pd.read_csv(f, usecols=["station", "valid", "vsby", "wxcodes", "label"], dtype={"station": str, "wxcodes": str})
    d["t"] = pd.to_datetime(d.valid, utc=True).dt.as_unit("ns").astype("int64")
    wx = d.wxcodes.fillna("")
    d["c"] = (d.label == 1) | ((d.vsby <= L["vis_max_mi"]) & wx.map(lambda s: bool(s) and has_code(s, real)))
    d = d.sort_values("t")
    return {s: (g.t.values, g.c.values.astype(bool)) for s, g in d.groupby("station")}


def nearest(t: np.ndarray, c: np.ndarray, tt: int, tol_min: int = 5) -> int:
    """Confirmed flag of the report nearest to tt (1/0), or -1 when none lies within tol_min."""
    j = np.searchsorted(t, tt)
    cand = [q for q in (j - 1, j) if 0 <= q < len(t) and abs(int(t[q]) - tt) <= tol_min * MIN]
    if not cand:
        return -1
    return int(c[min(cand, key=lambda q: abs(int(t[q]) - tt))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["train", "val"])
    a = ap.parse_args()
    cfg = load_config()
    L, root = Path(cfg["paths"]["labels"]), Path(cfg["paths"]["samples"])
    leads = cfg["time"]["leads_min"]
    st = pd.read_csv(L / ("stations_ext.csv" if (L / "stations_ext.csv").exists() else "stations.csv"), dtype={"sid": str})
    if "source" in st:
        st = st[st.source == "asos"]
    cell = {}
    for r in st.itertuples():
        cell.setdefault((int(r.i), int(r.j)), []).append(r.sid)

    out, rows, n_files, n_add, n_add_now, n_strict, missing = {}, [], 0, np.zeros(len(leads), int), 0, np.zeros(len(leads), int), set()
    cur_ev, series = None, {}
    for split in a.splits:
        for f in sorted((root / split).glob("*.npz"), key=lambda p: p.stem):
            if ".tmp" in f.name:
                continue
            event = f.stem.rsplit("_", 1)[0]
            if event != cur_ev:
                ef = L / "events" / f"{event}.csv"
                series = event_series(ef, cfg) if ef.exists() else None
                if series is None:
                    missing.add(event)
                cur_ev = event
            n_files += 1
            if series is None:
                continue
            with np.load(f, allow_pickle=False) as z:
                ij, sy, now, t0 = z["st_ij"], z["st_y"].astype(np.int8), z["st_now"].astype(np.int8), z["t0"][()]
                src = z["st_src"] if "st_src" in z.files else np.zeros(len(ij), int)
            n_strict += (sy == 1).sum(0)
            t0_ns = int(pd.Timestamp(t0).tz_localize("UTC").as_unit("ns").value)
            c = np.concatenate([sy, now[:, None]], 1)
            changed = False
            for k in range(len(ij)):
                if src[k] != 0:
                    continue
                for sid in cell.get((int(ij[k, 0]), int(ij[k, 1])), ()):
                    if sid not in series:
                        continue
                    t, cf = series[sid]
                    if not cf.any():
                        continue
                    for li, lead in enumerate(leads):
                        if sy[k, li] == 0 and nearest(t, cf, t0_ns + lead * MIN) == 1:
                            c[k, li] = 1; n_add[li] += 1; changed = True
                    if now[k] == 0 and nearest(t, cf, t0_ns) == 1:
                        c[k, -1] = 1; n_add_now += 1; changed = True
            if changed:
                out[f"{split}/{f.name}"] = c
                rows.append(dict(split=split, file=f.name, **{f"cpos{l}": int((c[:, li] == 1).sum()) for li, l in enumerate(leads)}))
            if n_files % 4000 == 0:
                print(f"  {n_files} samples, {len(out)} gain a positive", flush=True)
    np.savez_compressed(root / "confirmed_labels.npz", **out)
    pd.DataFrame(rows, columns=["split", "file"] + [f"cpos{l}" for l in leads]).to_csv(root / "index_confirmed.csv", index=False)
    print(f"{n_files} samples in {a.splits}; {len(out)} gain a positive; events without a report file: {len(missing)}")
    for li, l in enumerate(leads):
        print(f"  +{l} min: strict positives {int(n_strict[li])}, added {int(n_add[li])} (+{100 * n_add[li] / max(n_strict[li], 1):.0f} %)")
    print(f"  current-report column: {n_add_now} rows raised to dust")
    stn = [k for k in out if k.split("/")[1].startswith("stndust_")]
    print(f"  of the samples that gain a positive, {len(stn)} are on station-confirmed days (no warning at all)")
    sys.exit(0 if len(out) else 1)


if __name__ == "__main__":
    main()
