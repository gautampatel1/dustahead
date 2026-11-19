"""Step 32: how much station-observed dust lies OUTSIDE our event days? (the event list comes from warnings only)

    python -u scripts/32_scan_all_days_dust.py [--start 2018-01-01] [--end 2026-11-01]

The event list (step 1) holds days with an NWS (National Weather Service) Dust Storm Warning plus matched no-dust
days. Dust that reached an ASOS station (Automated Surface Observing System) on a day with no warning anywhere in
the domain is not in the samples at all, neither as a positive to learn from nor as a case to be scored on. Those
are exactly the storms where a model could add most to the official warning. This script downloads every report of
every station for the whole period from the Iowa Environmental Mesonet, month by month, keeps the reports with a
real dust weather code (not haze) and visibility <= 6 mi, and counts those that fall outside the event windows.
Output: data/labels/all_days_dust_reports.csv and a printed summary (reports, station-days, onsets, by year and
by severity, inside and outside the event list). Nothing is added to the labels: this is a count to decide on.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DUST = ("DS", "SS", "DU", "BLDU", "VCBLDU", "DRDU", "SA", "BLSA")      # label.dust_codes without HZ (haze)


def main():
    from dustahead import load_config
    from dustahead.events.stations import fetch_asos, has_code
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-11-01")
    a = ap.parse_args()
    cfg = load_config(a.config)
    L = Path(cfg["paths"]["labels"])
    st = pd.read_csv(L / "stations.csv", dtype={"sid": str})
    out = L / "all_days_dust_reports.csv"
    parts_dir = L / "all_days_parts"
    parts_dir.mkdir(exist_ok=True)
    months = pd.date_range(a.start, a.end, freq="MS", tz="UTC")
    for m0 in months:
        f = parts_dir / f"{m0:%Y%m}.csv"
        if f.exists():
            continue
        m1 = m0 + pd.offsets.MonthBegin(1)
        rows = []
        for k in range(0, len(st), 30):                      # 30 stations per request
            for attempt in range(4):
                try:
                    d = fetch_asos(list(st.sid[k:k + 30]), m0, m1, tuple(cfg["label"]["asos_report_types"]))
                    break
                except Exception as e:  # noqa: BLE001
                    if "no data" in str(e):
                        d = None
                        break
                    print(f"  retry {m0:%Y-%m} stations {k}: {str(e)[:100]}", flush=True)
                    time.sleep(20 * (attempt + 1)); d = None
            if d is None or d.empty:
                continue
            d = d[(d.vsby <= 6) & (d.wxcodes != "")]
            d = d[d.wxcodes.map(lambda w: has_code(w, DUST))]
            rows.append(d[["station", "valid", "vsby", "wxcodes", "sknt", "gust"]])
        pd.concat(rows).to_csv(f, index=False) if rows else pd.DataFrame(columns=["station", "valid", "vsby", "wxcodes", "sknt", "gust"]).to_csv(f, index=False)
        print(f"{m0:%Y-%m}: {sum(len(r) for r in rows)} dust reports", flush=True)
    D = pd.concat([pd.read_csv(f, dtype={"station": str}) for f in sorted(parts_dir.glob("*.csv"))], ignore_index=True)
    D["valid"] = pd.to_datetime(D.valid, utc=True)
    ev = pd.read_csv(L / "events.csv")
    ev["t_start"], ev["t_end"] = pd.to_datetime(ev.t_start, utc=True), pd.to_datetime(ev.t_end, utc=True)
    pad = pd.Timedelta(hours=6)                               # samples cover a few hours around each event
    inside = pd.Series(False, index=D.index)
    for e in ev.itertuples():
        inside |= (D.valid >= e.t_start - pad) & (D.valid <= e.t_end + pad)
    D["in_event_list"] = inside
    D["year"] = D.valid.dt.year
    D.to_csv(out, index=False)

    def onsets(x: pd.DataFrame) -> int:
        n = 0
        for _, g in x.sort_values("valid").groupby("station"):
            n += 1 + int((g.valid.diff().dt.total_seconds() > 3600).sum())
        return n

    print(f"\n{len(D)} dust-coded reports with visibility <= 6 mi, {a.start} to {a.end}, {st.sid.nunique()} stations")
    for name, m in (("visibility <= 1 mi (severe)", D.vsby <= 1), ("visibility <= 3 mi", D.vsby <= 3), ("visibility <= 6 mi", D.vsby <= 6)):
        x = D[m]
        i, o = x[x.in_event_list], x[~x.in_event_list]
        print(f"{name:28s} | inside event list: {len(i):6d} reports, {onsets(i):5d} onsets, {i.assign(d=i.valid.dt.date).groupby(['station', 'd']).ngroups:5d} station-days"
              f" | OUTSIDE: {len(o):6d} reports, {onsets(o):5d} onsets, {o.assign(d=o.valid.dt.date).groupby(['station', 'd']).ngroups:5d} station-days, {o.valid.dt.date.nunique():4d} days")
    sev = D[(D.vsby <= 1) & ~D.in_event_list]
    print("severe, outside the event list, by year:", sev.year.value_counts().sort_index().to_dict())
    print("severe, outside the event list, top stations:", sev.station.value_counts().head(12).to_dict())
    print("wrote", out)


if __name__ == "__main__":
    main()
