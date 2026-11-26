"""Step 28: are dust ONSETS preceded by something a station upstream already reports? (no training, no torch)

    python3 scripts/28_upstream_precursors.py [--labels data/labels] [--lead 60] [--out results/upstream_precursors.json]

Theory 3: many "onsets" are not new storms but an existing dust wall or outflow arriving from upstream at highway
speed, so tracking what the stations already see could beat re-predicting it. Before building a tracker this
measures the ceiling from the ASOS reports (Automated Surface Observing System) of every event day:

  onset at station s, forecast time t0 : strict dust label at s around t0 + lead, and no strict dust within 50 km
                                         of s at t0 (the benchmark's genesis rule)
  precursor at t0                      : another station within R km reporting, in the 20 min up to t0,
                                         (a) strict dust, (b) any dust weather code, (c) peak wind >= 30 kt

For each precursor and radius: the share of onsets that had it (the most a tracker using it could catch), how
common it is when no onset follows, and the precision of "alert when the precursor is present" (onsets per alert).
Time is binned to 10 minutes; a station-bin counts when the station reported in it.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

BIN = 10          # minutes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels")
    ap.add_argument("--lead", type=int, nargs="+", default=[30, 60])
    ap.add_argument("--radii", type=float, nargs="+", default=[50, 100, 150])
    ap.add_argument("--years", type=int, nargs="*", default=None)
    ap.add_argument("--out", default="results/upstream_precursors.json")
    a = ap.parse_args()
    L = Path(a.labels)
    st = pd.read_csv(L / "stations.csv", dtype={"sid": str}).drop_duplicates("sid").reset_index(drop=True)
    sidx = {s: k for k, s in enumerate(st.sid)}
    lat, lon = np.deg2rad(st.lat.values), np.deg2rad(st.lon.values)
    dlat, dlon = lat[:, None] - lat[None, :], lon[:, None] - lon[None, :]
    D = 2 * 6371 * np.arcsin(np.sqrt(np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2))
    S = len(st)
    near50 = D <= 50                                   # includes the station itself
    rings = {r: (D <= r) & ~np.eye(S, dtype=bool) for r in a.radii}
    PRE = ("strict_dust", "dust_code", "wind30")
    cnt = {ld: {(p, r): np.zeros(4, np.int64) for p in PRE for r in a.radii} for ld in a.lead}   # [onset&pre, onset, pre, n]
    n_events = 0
    for f in sorted(glob.glob(str(L / "events" / "*.csv"))):
        if a.years and int(Path(f).stem.split("_")[1][:4]) not in a.years:
            continue
        x = pd.read_csv(f, usecols=["station", "valid", "sknt", "gust", "dust_code", "label"], dtype={"station": str})
        x = x[x.station.isin(sidx)]
        if x.empty:
            continue
        n_events += 1
        t = pd.to_datetime(x.valid, utc=True)
        b = ((t - t.min()).dt.total_seconds() // (BIN * 60)).astype(int).values
        s = x.station.map(sidx).values
        nb = b.max() + 1
        rep = np.zeros((S, nb), bool); strict = rep.copy(); dc = rep.copy(); w30 = rep.copy()
        rep[s, b] = True
        strict[s[x.label.values == 1], b[x.label.values == 1]] = True
        dcm = x.dust_code.astype(str).str.lower().isin(("true", "1", "1.0")).values
        dc[s[dcm], b[dcm]] = True
        wm = (x[["sknt", "gust"]].max(axis=1) >= 30).values
        w30[s[wm], b[wm]] = True

        def now(m):                                    # the 20 min up to t0: this bin or the one before
            o = m.copy(); o[:, 1:] |= m[:, :-1]; return o
        strict_now, dc_now, w_now, rep_now = now(strict), now(dc), now(w30), now(rep)
        genesis = ~((near50.astype(np.int32) @ strict_now.astype(np.int32)) > 0)
        pre = {"strict_dust": strict_now, "dust_code": dc_now, "wind30": w_now}
        for ld in a.lead:
            k = ld // BIN
            if nb <= k + 1:
                continue
            tgt = np.zeros((S, nb), bool); known = np.zeros((S, nb), bool)
            for d in (k - 1, k, k + 1):                # strict dust / a report around t0 + lead (+-10 min)
                if d < nb:
                    tgt[:, : nb - d] |= strict[:, d:]
                    known[:, : nb - d] |= rep[:, d:]
            m = genesis & known & rep_now
            for p in PRE:
                for r in a.radii:
                    has = (rings[r].astype(np.int32) @ pre[p].astype(np.int32)) > 0
                    c = cnt[ld][(p, r)]
                    c += np.array([(m & tgt & has).sum(), (m & tgt).sum(), (m & has).sum(), m.sum()])
    out = dict(events=n_events, stations=S, years=a.years, rows=[])
    for ld in a.lead:
        for (p, r), c in cnt[ld].items():
            both, on, pr, n = (int(v) for v in c)
            row = dict(lead=ld, precursor=p, radius_km=r, onsets=on, station_times=n,
                       onsets_with_precursor=both / max(on, 1), precursor_rate_without_onset=(pr - both) / max(n - on, 1),
                       precision=both / max(pr, 1), base_rate=on / max(n, 1))
            row["lift"] = row["precision"] / max(row["base_rate"], 1e-12)
            out["rows"].append(row)
            print(f"+{ld:<2d} {p:11s} within {r:>3.0f} km | onsets {on:5d} | had precursor {row['onsets_with_precursor']:.0%} | "
                  f"precursor without onset {row['precursor_rate_without_onset']:.1%} | precision {row['precision']:.2%} (base {row['base_rate']:.3%}, lift {row['lift']:.1f}x)", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out, "from", n_events, "event days")


if __name__ == "__main__":
    main()
