"""Step 14a: what extra ground truth does Synoptic Data offer inside the domain? Writes data/labels/synoptic_stations.csv
and prints counts by network and sensor, plus how many of them sit within 25 km of Interstate 10. Then pulls one
known haboob (Phoenix, 2025-08-25 20:00-2025-08-26 04:00 UTC) to show what the RWIS visibility and PM10 sensors saw.

    python scripts/14_synoptic_inventory.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dustahead import load_config
from dustahead.app.predict import I10
from dustahead.events.synoptic import station_inventory, timeseries
from dustahead.grid import Grid


def main():
    cfg = load_config()
    D = cfg["domain"]
    grid = Grid.from_config(cfg)
    inv = station_inventory((D["lon_min"], D["lat_min"], D["lon_max"], D["lat_max"]))
    print(f"{len(inv)} active stations with visibility / PM10 / precip / wind sensors in the domain", flush=True)
    # distance to I-10 (km) from the waypoint polyline
    lon10, lat10 = np.array(I10).T
    d = np.full(len(inv), np.inf)
    for k in range(len(lon10) - 1):
        for q in np.linspace(0, 1, 20):
            lo, la = lon10[k] + q * (lon10[k + 1] - lon10[k]), lat10[k] + q * (lat10[k + 1] - lat10[k])
            dk = np.hypot((inv.lon.values - lo) * 111 * np.cos(np.radians(la)), (inv.lat.values - la) * 111)
            d = np.minimum(d, dk)
    inv["km_to_i10"] = np.round(d, 1)
    inv["i"], inv["j"] = grid.to_ij(inv.lon.values, inv.lat.values)
    out = Path(cfg["paths"]["labels"]) / "synoptic_stations.csv"
    inv.to_csv(out, index=False)
    for col in ("has_vis", "has_pm10"):
        sub = inv[inv[col]]
        print(f"\n{col}: {len(sub)} stations; within 25 km of I-10: {int((sub.km_to_i10 <= 25).sum())}")
        print(sub.groupby("network").size().sort_values(ascending=False).head(12).to_string())
        print("period of record starts:", sub.por_start.str[:4].value_counts().sort_index().to_dict())
    print("\nwrote", out, flush=True)

    # one known haboob: what did the extra sensors see?
    ev_start, ev_end = pd.Timestamp("2025-08-25T20:00Z"), pd.Timestamp("2025-08-26T04:00Z")
    near = inv[(inv.has_vis | inv.has_pm10) & (inv.km_to_i10 <= 60)]
    ts = timeseries(list(near.stid), ev_start, ev_end)
    print(f"\nPhoenix haboob 2025-08-25: {len(ts)} observations from {ts.stid.nunique() if len(ts) else 0} stations")
    if len(ts):
        vis = ts[ts.variable == "visibility"]
        pm = ts[ts.variable == "PM_10_concentration"]
        print(f"visibility obs {len(vis)} from {vis.stid.nunique()} stations; <= 1 mi: {int((vis.value <= 1.0).sum())} obs at {vis[vis.value <= 1.0].stid.nunique()} stations")
        print(f"PM10 obs {len(pm)} from {pm.stid.nunique()} stations; >= 250 ug/m3: {int((pm.value >= 250).sum())} obs at {pm[pm.value >= 250].stid.nunique()} stations; max {pm.value.max() if len(pm) else 'n/a'}")
        if len(vis):
            print(vis[vis.value <= 1.0].sort_values("valid").head(15).to_string())
        if len(pm):
            print(pm[pm.value >= 250].sort_values("valid").head(15).to_string())
        ts.to_csv(Path(cfg["paths"]["labels"]) / "synoptic_phoenix_20250825.csv", index=False)


if __name__ == "__main__":
    main()
