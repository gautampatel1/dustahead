"""Step 1: pull NWS Dust Storm / Severe Thunderstorm warnings and build the event list.

    python scripts/01_build_event_list.py --years 2023 2024 2025 2026

Writes data/labels/warnings.csv (every warning) and data/labels/events.csv (dust events + null days),
and prints the go/no-go count per split.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dustahead import load_config
from dustahead.events.warnings import build_event_list, fetch_all_warnings, save_event_list

ap = argparse.ArgumentParser()
ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025, 2026])
ap.add_argument("--config", default=None)
a = ap.parse_args()
cfg = load_config(a.config)
out = Path(cfg["paths"]["labels"])

w = fetch_all_warnings(a.years)
save_event_list(w, out / "warnings.csv")
ev = build_event_list(w)
save_event_list(ev, out / "events.csv")

print(f"{len(w)} warnings -> {len(ev)} events")
for split, yrs in cfg["splits"].items():
    sub = ev[ev.year.isin(yrs)]
    print(f"{split:5s} {yrs}: {int((sub.kind == 'dust').sum()):3d} dust events, {int((sub.kind == 'null').sum()):3d} null days")
