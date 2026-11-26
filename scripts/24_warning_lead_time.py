"""Step 24: how early does the model alert a station, against the official Dust Storm Warning?

    python scripts/24_warning_lead_time.py --runs runs/gated_unet_v8 runs/gated_unet_v8_fold1 \
        --configs default configs/fold1.yaml --thr 0.8 --out results/lead_time_v8_2fold

The operational comparison. Station persistence cannot warn before dust arrives; the NWS (National Weather Service)
Dust Storm Warning (DS.W) can. For every DUST ONSET at an ASOS station (Automated Surface Observing System) in the
validation events (first strict-label report after 60 min without one) this script finds
  * the model's first alert for that station in the 90 min before the onset: the earliest forecast time t0 < onset
    at which P(dust) >= thr at any lead for the station's cell. Model lead = onset - t0.
  * the first Dust Storm Warning whose polygon contains the station (or, without a polygon, whose text names the
    station's county) and whose padded window covers the onset. NWS lead = onset - issue time (negative = the warning
    came after the dust).
With two runs trained on opposite halves of the 2025 events (2-fold), every 2025 onset is scored out of sample.
The other side of the ledger is reported too: how many model alerts and how many warned stations saw no dust.
Rows of eval_val_preds.npz are rebuilt from the sample files in the order evaluate.collect() read them.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MONSOON = (6, 7, 8, 9)


def rows_for(run: str, cfg: dict):
    """Per prediction row: forecast time, station cell, event id; plus the stored predictions and labels."""
    from dustahead.train.dataset import split_files
    files, _ = split_files(cfg["paths"]["samples"], "val", cfg)
    z = np.load(Path(run) / "eval_val_preds.npz", allow_pickle=True)
    t0, ij, ev = [], [], []
    for f in files:
        with np.load(f, allow_pickle=False) as s:
            n = len(s["st_ij"])
            t0.append(np.repeat(np.datetime64(s["t0"][()], "s"), n))
            ij.append(s["st_ij"].astype(int))
            ev.append(np.repeat(f.stem.rsplit("_", 1)[0], n))
    t0, ij, ev = np.concatenate(t0), np.concatenate(ij), np.concatenate(ev)
    assert len(t0) == len(z["y"]), f"{run}: {len(t0)} rebuilt rows against {len(z['y'])} stored (wrong --configs entry?)"
    assert (ev == z["event"]).all(), f"{run}: event order differs from the stored predictions"
    src = z["src"] if "src" in z.files else np.zeros(len(t0), int)
    return pd.DataFrame(dict(t0=t0, i=ij[:, 0], j=ij[:, 1], event=ev, src=src)), z["p_model"], z["y"]


def main():
    from dustahead import load_config
    from dustahead.events.stations import _to_utc
    from dustahead.events.warnings import point_in_rings
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--configs", nargs="+", required=True, help="one per run; 'default' for configs/default.yaml")
    ap.add_argument("--thr", type=float, nargs="+", default=[0.8, 0.5])
    ap.add_argument("--window-min", type=int, default=90)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfgs = [load_config(None if c == "default" else c) for c in a.configs]
    cfg = cfgs[0]
    leads = cfg["time"]["leads_min"]
    pad = pd.Timedelta(minutes=cfg["label"]["warning_pad_min"])
    L = Path(cfg["paths"]["labels"])

    parts = [rows_for(r, c) for r, c in zip(a.runs, cfgs)]
    R = pd.concat([p[0] for p in parts], ignore_index=True)
    P, Y = np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts])
    keep = (R.src == 0).values
    R, P, Y = R[keep].reset_index(drop=True), P[keep], Y[keep]
    R["t0"] = pd.to_datetime(R.t0).dt.tz_localize("UTC")
    print(f"{len(R)} ASOS rows, {R.event.nunique()} events", flush=True)

    st = pd.read_csv(L / ("stations_ext.csv" if (L / "stations_ext.csv").exists() else "stations.csv"))
    st = st[st.source == "asos"] if "source" in st else st
    cell = {(int(r.i), int(r.j)): r for r in st.drop_duplicates(["i", "j"]).itertuples()}
    W = pd.read_csv(L / "warnings.csv")
    W = W[W.phenomena == "DS"].copy()
    W["issue"], W["expire"] = _to_utc(W.issue), _to_utc(W.expire)
    polys = {}
    for w in W.itertuples():
        pf = L / "polygons" / f"{w.wfo}_{w.year}_{w.eventid}.npz"
        if pf.exists():
            with np.load(pf) as z:
                polys[(w.wfo, w.year, w.eventid)] = [z[k] for k in z.files] or None

    def covers(w, s) -> bool:
        rings = polys.get((w.wfo, w.year, w.eventid))
        if rings is not None:
            return point_in_rings(s.lon, s.lat, rings)
        return bool(s.county) and str(s.county).split()[0] in str(w.locations)

    # ---- dust onsets per station, from the labels the benchmark uses
    onsets = []
    for (ev, i, j), g in R.groupby(["event", "i", "j"], sort=False):
        idx = g.index.values
        times = sorted({g.t0.loc[k] + pd.Timedelta(minutes=leads[l]) for k in idx for l in range(len(leads)) if Y[k, l] == 1})
        last = None
        for t in times:
            if last is None or (t - last) > pd.Timedelta(minutes=60):
                onsets.append(dict(event=ev, i=int(i), j=int(j), onset=t))
            last = t
    O = pd.DataFrame(onsets)
    print(f"{len(O)} dust onsets at ASOS stations", flush=True)

    by_station = {k: g for k, g in R.groupby(["event", "i", "j"], sort=False)}
    rec = []
    for o in O.itertuples():
        s = cell.get((o.i, o.j))
        g = by_station[(o.event, o.i, o.j)]
        win = g[(g.t0 < o.onset) & (g.t0 >= o.onset - pd.Timedelta(minutes=a.window_min))]
        r = dict(event=o.event, station=getattr(s, "sid", f"{o.i},{o.j}"), onset=o.onset, monsoon=o.onset.month in MONSOON,
                 n_forecasts=len(win))
        for thr in a.thr:
            hit = win[(P[win.index.values] >= thr).any(1)]
            r[f"model_lead_{thr}"] = (o.onset - hit.t0.min()).total_seconds() / 60 if len(hit) else np.nan
        nws = np.nan
        if s is not None:
            cand = W[(W.issue - pad <= o.onset) & (W.expire + pad >= o.onset)]
            iss = [w.issue for w in cand.itertuples() if covers(w, s)]
            if iss:
                nws = (o.onset - min(iss)).total_seconds() / 60
        r["nws_lead"] = nws
        rec.append(r)
    D = pd.DataFrame(rec)
    D = D[D.n_forecasts > 0].reset_index(drop=True)        # onsets with at least one forecast in the window

    def summary(d: pd.DataFrame, thr: float) -> dict:
        m, n = d[f"model_lead_{thr}"], d.nws_lead
        m_ok, n_ok = m.notna(), n.notna() & (n > 0)
        mm, nn = m.fillna(0), n.where(n_ok, 0)          # minutes of warning before the dust; 0 = none
        return dict(onsets=int(len(d)),
                    model_alert_before=float(m_ok.mean()), model_median_lead_when_alerted=float(m[m_ok].median()) if m_ok.any() else None,
                    nws_warning_exists=float(n.notna().mean()), nws_warning_before=float(n_ok.mean()),
                    nws_median_lead_when_before=float(n[n_ok].median()) if n_ok.any() else None,
                    nws_median_lead_all_warned=float(n[n.notna()].median()) if n.notna().any() else None,
                    model_earlier=float((mm > nn).mean()), nws_earlier=float((nn > mm).mean()), neither=float(((mm == 0) & (nn == 0)).mean()),
                    either=float((m_ok | n_ok).mean()))

    out = dict(runs=a.runs, window_min=a.window_min, thresholds=a.thr, results={})
    for thr in a.thr:
        out["results"][str(thr)] = dict(all=summary(D, thr), monsoon=summary(D[D.monsoon], thr), other=summary(D[~D.monsoon], thr))
        # the cost: alerts at station-times with no dust at that station at any lead (labels known at all leads)
        known = (Y >= 0).all(1)
        alert = (P >= thr).any(1) & known
        dust = (Y == 1).any(1)
        out["results"][str(thr)]["model_alert_rows"] = dict(alerts=int(alert.sum()), with_dust_within_60=int((alert & dust).sum()),
                                                             false_alert_ratio=float(1 - (alert & dust).sum() / max(alert.sum(), 1)))
        print(thr, json.dumps(out["results"][str(thr)], default=float)[:1500], flush=True)

    # NWS side of the cost: of (warning, ASOS station inside it) pairs in these events, how many saw strict dust
    lab = []
    for ev in R.event.unique():
        f = L / "events" / f"{ev}.csv"
        if f.exists():
            x = pd.read_csv(f, usecols=["station", "valid", "label"], dtype={"station": str})
            x = x[x.label == 1]
            if len(x):
                x["valid"] = _to_utc(x.valid); lab.append(x)
    lab = pd.concat(lab) if lab else pd.DataFrame(columns=["station", "valid"])
    t_lo, t_hi = R.t0.min() - pd.Timedelta(hours=3), R.t0.max() + pd.Timedelta(hours=3)
    pairs = hit = 0
    for w in W[(W.issue >= t_lo) & (W.issue <= t_hi)].itertuples():
        for s in st.itertuples():
            if covers(w, s):
                pairs += 1
                x = lab[(lab.station == s.sid) & (lab.valid >= w.issue - pad) & (lab.valid <= w.expire + pad)]
                hit += int(len(x) > 0)
    out["nws_warned_station_pairs"] = dict(pairs=pairs, with_strict_dust=hit, no_dust_ratio=float(1 - hit / max(pairs, 1)))
    print("NWS warned-station pairs", out["nws_warned_station_pairs"], flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    D.to_csv(f"{a.out}.csv", index=False)
    json.dump(out, open(f"{a.out}.json", "w"), indent=1, default=float)
    print("wrote", f"{a.out}.json / .csv")


if __name__ == "__main__":
    main()
