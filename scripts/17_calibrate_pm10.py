"""Step 17: calibrate the PM10 label threshold against the ASOS rule, using co-located stations.

    python scripts/17_calibrate_pm10.py [--radius-cells 3] [--window-min 30]

For every PM10 observation (data/labels/synoptic/<event>.csv) that has an ASOS station within `radius_cells`
(3 cells ~ 12 km, the label disc) and an ASOS report within ±window of its time, pair the two. Then sweep the PM10
threshold and score "PM10 >= thr" against the ASOS label (vis <= 1 mi AND dust code AND warning) with hits / false
alarms / misses -> CSI. The threshold with the best CSI is the one at which "PM10 dust" means the same thing as the
ASOS dust label the model was built on. Also reports the RWIS visibility rule the same way (threshold sweep on miles).
Writes results/pm10_calibration.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dustahead import load_config


def pair(ext_obs: pd.DataFrame, asos_lab: pd.DataFrame, pairs: dict, window: pd.Timedelta) -> pd.DataFrame:
    """ext_obs: station, valid, value. For each row, the max ASOS label among co-located stations within ±window
    (vectorised: one merge_asof per direction over the exploded station pairs)."""
    link = pd.DataFrame([(e, a) for e, al in pairs.items() for a in al], columns=["station", "asos"])
    x = ext_obs.merge(link, on="station")
    if x.empty:
        return pd.DataFrame(columns=["station", "valid", "value", "asos_label"])
    x = x.sort_values("valid").reset_index(drop=True)
    x["row"] = np.arange(len(x))
    lab = asos_lab[["station", "valid", "label"]].rename(columns={"station": "asos"}).sort_values("valid")
    best = pd.Series(-1, index=x.row)
    for direction in ("backward", "forward"):
        m = pd.merge_asof(x[["row", "valid", "asos"]], lab, on="valid", by="asos", direction=direction, tolerance=window)
        got = m.label.notna().values
        best.loc[m.row[got]] = np.maximum(best.loc[m.row[got]].values, m.label[got].astype(int).values)
    x["asos_label"] = best.values
    # several ASOS neighbours -> keep the max label per extended observation
    out = x.groupby(["station", "valid", "value"], as_index=False).asos_label.max()
    return out[out.asos_label >= 0].reset_index(drop=True)


def sweep(df: pd.DataFrame, thresholds, ge: bool = True) -> list[dict]:
    rows = []
    for t in thresholds:
        pred = (df.value >= t) if ge else (df.value <= t)
        h = int((pred & (df.asos_label == 1)).sum()); fa = int((pred & (df.asos_label == 0)).sum()); mi = int((~pred & (df.asos_label == 1)).sum())
        csi = h / (h + fa + mi) if h + fa + mi else float("nan")
        rows.append(dict(threshold=float(t), hits=h, false_alarms=fa, misses=mi, csi=csi,
                         pod=h / (h + mi) if h + mi else float("nan"), far=fa / (h + fa) if h + fa else float("nan")))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius-cells", type=int, default=3)
    ap.add_argument("--window-min", type=int, default=30)
    a = ap.parse_args()
    cfg = load_config()
    L = Path(cfg["paths"]["labels"])
    st = pd.read_csv(L / "stations_ext.csv")
    asos = st[st.source == "asos"]
    pairs = {}
    for r in st[st.source != "asos"].itertuples():
        d = np.hypot(asos.i.values - r.i, asos.j.values - r.j)
        near = asos.sid.values[d <= a.radius_cells]
        if len(near):
            pairs[r.sid] = list(near)
    print(f"{len(pairs)} extended stations have an ASOS station within {a.radius_cells} cells: "
          f"{st.set_index('sid').loc[list(pairs)].source.value_counts().to_dict()}", flush=True)
    window = pd.Timedelta(minutes=a.window_min)
    pm, vis = [], []
    src = st.set_index("sid").source
    for raw in sorted((L / "synoptic").glob("*.csv")):
        ev = raw.stem
        lab_f = L / "events" / f"{ev}.csv"
        if not lab_f.exists():
            continue
        ts = pd.read_csv(raw, parse_dates=["valid"])
        if ts.empty:
            continue
        ts["valid"] = pd.to_datetime(ts.valid, utc=True)
        lab = pd.read_csv(lab_f, parse_dates=["valid"])
        lab["valid"] = pd.to_datetime(lab.valid, utc=True)
        ts = ts[ts.station.isin(pairs)] if "station" in ts else ts[ts.stid.isin(pairs)].rename(columns={"stid": "station"})
        p = ts[(ts.variable == "PM_10_concentration")]
        v = ts[(ts.variable == "visibility") & ts.station.map(src).isin(["rwis", "vis_other"])]
        if len(p):
            pm.append(pair(p, lab, pairs, window))
        if len(v):
            vis.append(pair(v, lab, pairs, pd.Timedelta(minutes=5)))
    pm = pd.concat(pm, ignore_index=True) if pm else pd.DataFrame(columns=["value", "asos_label"])
    vis = pd.concat(vis, ignore_index=True) if vis else pd.DataFrame(columns=["value", "asos_label"])
    res = dict(radius_cells=a.radius_cells, window_min=a.window_min,
               pm10=dict(n_pairs=int(len(pm)), n_asos_dust=int((pm.asos_label == 1).sum()),
                         sweep=sweep(pm, [100, 150, 200, 250, 300, 400, 500, 600, 750, 1000, 1250, 1500, 2000, 3000])),
               rwis_visibility=dict(n_pairs=int(len(vis)), n_asos_dust=int((vis.asos_label == 1).sum()),
                                    sweep=sweep(vis, [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0], ge=False)))
    print(f"\nPM10 pairs {len(pm)} (ASOS dust {int((pm.asos_label == 1).sum())}); median PM10 when ASOS says dust: "
          f"{pm[pm.asos_label == 1].value.median() if (pm.asos_label == 1).any() else float('nan'):.0f}, when clear: {pm[pm.asos_label == 0].value.median() if len(pm) else float('nan'):.0f}")
    for r in res["pm10"]["sweep"]:
        print(f"  PM10 >= {r['threshold']:6.0f}: CSI {r['csi']:.3f}  POD {r['pod']:.2f}  FAR {r['far']:.2f}  (hits {r['hits']}, FA {r['false_alarms']}, miss {r['misses']})")
    print(f"\nRWIS/other visibility pairs {len(vis)} (ASOS dust {int((vis.asos_label == 1).sum())})")
    for r in res["rwis_visibility"]["sweep"]:
        print(f"  vis <= {r['threshold']:4.2f} mi: CSI {r['csi']:.3f}  POD {r['pod']:.2f}  FAR {r['far']:.2f}  (hits {r['hits']}, FA {r['false_alarms']}, miss {r['misses']})")
    Path("results").mkdir(exist_ok=True)
    Path("results/pm10_calibration.json").write_text(json.dumps(res, indent=1))
    print("wrote results/pm10_calibration.json")


if __name__ == "__main__":
    main()
