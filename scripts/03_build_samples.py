"""Step 3: labels + samples for every event in data/labels/events.csv.

    python scripts/03_build_samples.py [--limit 5] [--no-adp] [--only dust|null]

For each event: 72 forecast times at 10-min steps across a 12 h window. For each t0 it downloads
6 GOES frames, GLM, HRRR, computes the physics prior, paints targets from station labels and
writes data/samples/<split>/<event_id>_<t0>.npz (~6 MB each). Resumable: existing files are skipped.
Expect ~1-2 min per sample on a home connection the first time (GOES files are cached in data/raw).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dustahead import load_config
from dustahead.data.sampler import build_sample, sample_times, save_sample
from dustahead.data.static import load_static
from dustahead.events.stations import fetch_asos, fetch_station_table, label_reports
from dustahead.events.warnings import fetch_polygon
from dustahead.grid import Grid

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=None, help="max events (for a first test)")
ap.add_argument("--no-adp", action="store_true")
ap.add_argument("--only", choices=["dust", "null"], default=None)
a = ap.parse_args()

cfg = load_config()
grid = Grid.from_config(cfg)
L = Path(cfg["paths"]["labels"])
events = pd.read_csv(L / "events.csv", parse_dates=["t_start", "t_end"])
warnings = pd.read_csv(L / "warnings.csv", parse_dates=["issue", "expire"])
if a.only:
    events = events[events.kind == a.only]
if a.limit:
    events = events.head(a.limit)

stations = fetch_station_table(grid)
stations.to_csv(L / "stations.csv", index=False)
static = load_static(cfg["paths"]["static"], grid)
split_of = {y: s for s, ys in cfg["splits"].items() for y in ys}

for ev in events.itertuples():
    split = split_of.get(ev.year)
    if split is None:
        continue
    t_lo, t_hi = ev.t_start - pd.Timedelta(hours=7), ev.t_start + pd.Timedelta(hours=7)
    obs = fetch_asos(list(stations.sid), t_lo, t_hi)
    ds_w = warnings[(warnings.phenomena == "DS") & (warnings.expire >= t_lo) & (warnings.issue <= t_hi)]
    polys = {(w.wfo, w.year, w.eventid): fetch_polygon(w.wfo, w.year, w.eventid) for w in ds_w.itertuples()}
    labels = label_reports(obs, ds_w, stations, cfg, polys)
    labels.to_csv(L / f"labels_{ev.event_id}.csv", index=False)
    n_pos = int(labels.label.sum())
    print(f"[{ev.event_id}] {len(labels)} reports, {n_pos} dust-positive")
    for t0 in sample_times(ev, cfg):
        out = Path(cfg["paths"]["samples"]) / split / f"{ev.event_id}_{t0:%Y%m%dT%H%M}.npz"
        if out.exists():
            continue
        try:
            s = build_sample(t0, grid, cfg, static, labels, stations, use_adp=not a.no_adp)
        except Exception as e:  # noqa: BLE001
            print(f"  {t0:%H:%M} failed: {e}")
            continue
        if s is None:
            print(f"  {t0:%H:%M} no satellite data")
            continue
        save_sample(out, s)
    print(f"  done {ev.event_id}")
