"""Step 33: event days from STATION reports alone, for severe dust that no warning covered.

    python scripts/33_station_events.py            # needs data/labels/all_days_dust_reports.csv from step 32

The event list (step 1) comes from NWS (National Weather Service) Dust Storm Warnings, so severe dust on a day
without a warning has no samples at all. Step 32 found 133 such reports with visibility <= 1 mi in 2018-2026. This
step groups the ones in the TRAINING years (splits.train; never the validation or the closed test year) into events
(reports less than 3 h apart, any station) and writes data/labels/events_station.csv in the events.csv layout, with
event ids `stndust_<start>`. Build them with
    python -u scripts/03_build_samples.py --events-file data/labels/events_station.csv
Under the strict label rule these days carry no positive (no warning confirms the report), so the dataset hides them
unless label.variant is station_confirmed (scripts/34_add_confirmed_labels.py). ASOS = Automated Surface Observing System.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dustahead import load_config  # noqa: E402


def main():
    cfg = load_config()
    L = Path(cfg["paths"]["labels"])
    d = pd.read_csv(L / "all_days_dust_reports.csv", dtype={"station": str})
    d["valid"] = pd.to_datetime(d.valid, utc=True)
    out_flag = ~d.in_event_list.astype(str).str.lower().isin(("true", "1"))
    sev = d[out_flag & (d.vsby <= cfg["label"]["vis_max_mi"]) & d.year.isin(cfg["splits"]["train"])].sort_values("valid")
    events, cur = [], None
    for r in sev.itertuples():
        if cur is None or (r.valid - cur["last"]) > pd.Timedelta(hours=3):
            if cur:
                events.append(cur)
            cur = dict(t_start=r.valid, last=r.valid, n=0, st=set())
        cur["last"] = r.valid
        cur["n"] += 1
        cur["st"].add(r.station)
    if cur:
        events.append(cur)
    pad = pd.Timedelta(minutes=cfg["label"]["warning_pad_min"])       # a warning would outlast the last report
    ev = pd.DataFrame([dict(t_start=e["t_start"], t_end=e["last"] + pad, n_warnings=0, states="", wfos="", kind="dust",
                            event_id=f"stndust_{e['t_start']:%Y%m%d_%H%M}", year=e["t_start"].year,
                            n_reports=e["n"], stations="+".join(sorted(e["st"]))) for e in events])
    ev.to_csv(L / "events_station.csv", index=False)
    print(f"{len(sev)} severe reports outside the event list in {sorted(cfg['splits']['train'])} -> {len(ev)} events")
    print(ev.groupby("year").agg(events=("event_id", "size"), reports=("n_reports", "sum")).to_string())
    print(ev[["event_id", "t_end", "n_reports", "stations"]].to_string())


if __name__ == "__main__":
    main()
