"""Step 3: labels + samples for every event in data/labels/events.csv.

    python scripts/03_build_samples.py [--limit 2] [--only dust|null] [--workers 4] [--no-adp]

Per event: fetch 5-min ASOS reports, DS.W polygons and Storm Events, build station labels, then for
each forecast time t0 (every 10 min from 3 h before the first warning to 2 h after the last expiry)
read 6 GOES frames + GLM + HRRR remotely, compute the physics prior, paint targets and write
data/samples/<split>/<event_id>_<t0>.npz. Resumable: existing files and cached frames are skipped.
Transfer ~2 GB per event; stored ~250 MB per event (samples + cache).
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dustahead import load_config
from dustahead.data.sampler import build_sample, sample_times, save_sample
from dustahead.data.static import load_static
from dustahead.events.stations import fetch_asos, fetch_station_table, label_reports
from dustahead.events.storm_events import fetch_dust_storm_events
from dustahead.events.warnings import fetch_polygon
from dustahead.grid import Grid

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=None, help="max events (for a first test)")
ap.add_argument("--only", choices=["dust", "null"], default=None)
ap.add_argument("--workers", type=int, default=4, help="parallel forecast times per event (I/O bound)")
ap.add_argument("--no-adp", action="store_true")
ap.add_argument("--event", default=None, help="build only this event_id")
ap.add_argument("--max-samples", type=int, default=None, help="per event, for quick tests")
ap.add_argument("--null-ratio", type=float, default=1.0, help="null days per dust event (per year), sampled with seed 0")
a = ap.parse_args()

cfg = load_config()
grid = Grid.from_config(cfg)
L = Path(cfg["paths"]["labels"])
events = pd.read_csv(L / "events.csv", parse_dates=["t_start", "t_end"])
warnings = pd.read_csv(L / "warnings.csv", parse_dates=["issue", "expire"])
if a.event:
    events = events[events.event_id == a.event]
if a.only:
    events = events[events.kind == a.only]
# keep null days in proportion to dust events, per year (there are ~2x more null days than dust events)
keep = []
for yr, grp in events.groupby("year"):
    n_dust = int((grp.kind == "dust").sum())
    nulls = grp[grp.kind == "null"].sample(min(int(a.null_ratio * n_dust), int((grp.kind == "null").sum())), random_state=0) if n_dust else grp.iloc[0:0]
    keep.append(pd.concat([grp[grp.kind == "dust"], nulls]))
events = pd.concat(keep).sort_values("t_start") if keep else events
if a.limit:
    events = events.head(a.limit)
print(f"building {int((events.kind == 'dust').sum())} dust events + {int((events.kind == 'null').sum())} null days")

stations = fetch_station_table(grid)
stations.to_csv(L / "stations.csv", index=False)
static = load_static(cfg["paths"]["static"], grid)
split_of = {y: s for s, ys in cfg["splits"].items() for y in ys}
storm_events = {}

for ev in events.itertuples():
    split = split_of.get(ev.year)
    if split is None:
        continue
    t_lo = ev.t_start - pd.Timedelta(hours=cfg["time"]["hours_before"] + 1)
    t_hi = ev.t_end + pd.Timedelta(hours=cfg["time"]["hours_after"] + 1)
    obs = fetch_asos(list(stations.sid), t_lo, t_hi, tuple(cfg["label"]["asos_report_types"]))
    ds_w = warnings[(warnings.phenomena == "DS") & (warnings.expire >= t_lo) & (warnings.issue <= t_hi)]
    polys = {(w.wfo, w.year, w.eventid): fetch_polygon(w.wfo, w.year, w.eventid) for w in ds_w.itertuples()}
    if ev.year not in storm_events:
        try:
            storm_events[ev.year] = fetch_dust_storm_events(ev.year, cache_dir=cfg["paths"]["raw"] + "/storm_events")
        except Exception as e:  # noqa: BLE001
            print(f"[storm events {ev.year}] unavailable: {e}")
            storm_events[ev.year] = None
    labels = label_reports(obs, ds_w, stations, cfg, polys, storm_events[ev.year])
    labels.to_csv(L / f"labels_{ev.event_id}.csv", index=False)
    n_pos = int(labels.label.sum())
    print(f"[{ev.event_id}] {len(labels)} reports from {labels.station.nunique()} stations, {n_pos} dust-positive, "
          f"{int(labels.severe.sum())} severe")

    def one(t0):
        out = Path(cfg["paths"]["samples"]) / split / f"{ev.event_id}_{t0:%Y%m%dT%H%M}.npz"
        if out.exists():
            return "skip"
        try:
            s = build_sample(t0, grid, cfg, static, labels, stations, use_adp=not a.no_adp)
        except Exception as e:  # noqa: BLE001
            return f"{t0:%H:%M} failed: {type(e).__name__}: {e}"
        if s is None:
            return f"{t0:%H:%M} no satellite data"
        save_sample(out, s)
        return "ok"

    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(one, sample_times(ev, cfg)[: a.max_samples]))
    print(f"  {results.count('ok')} written, {results.count('skip')} existed, "
          f"{sum(r not in ('ok', 'skip') for r in results)} failed")
    for r in results:
        if r not in ("ok", "skip"):
            print("   ", r)
