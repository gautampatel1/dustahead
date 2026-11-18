"""Step 26: dust HOTSPOT climatology from the years before validation -> data/static/hotspot_2018_2024.npy

    python -u scripts/26_build_hotspot.py [--stride 3] [--workers 6] [--last-year 2024]

Why: the false alarms are the problem, and the 16 Nov analysis showed the model cannot tell which strong wind will
lift dust. Dust sources are persistent places: dry lake beds (Lordsburg Playa, Willcox Playa), fallow fields, cleared
land. Two maps say where dust has been seen before:
  channel 0  satellite dust frequency: share of training samples in which the ADP (Aerosol Detection Product of the
             GOES satellite) flagged dust in the cell (the weak pixel labels, weight 0.3, already stored in the
             samples), relative to its 99th percentile, clipped to 0-1.
  channel 1  warning frequency: number of NWS (National Weather Service) Dust Storm Warning polygons that covered
             the cell, relative to its 99th percentile, clipped to 0-1.
Only events up to --last-year are used: the validation year (2025) and the test year (2026) contribute nothing, so
the map is a climatology the forecast could have had (hard rules 2 and 3). Both maps are smoothed over ~12 km.
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def adp_one(path):
    try:
        with np.load(path, allow_pickle=False) as z:
            y, w = z["y"][0], z["w"][0]
        return ((y == 1) & (w > 0) & (w < 1)).astype(np.uint8)
    except Exception:  # noqa: BLE001
        return None


def smooth(a: np.ndarray, r: int = 3) -> np.ndarray:
    """Box mean over (2r+1)^2 cells with edge padding (no scipy needed)."""
    p = np.pad(a.astype(np.float64), r, mode="edge")
    c = p.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    k = 2 * r + 1
    return ((c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / k ** 2).astype(np.float32)


def main():
    from dustahead import load_config
    from dustahead.grid import Grid
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--last-year", type=int, default=2024)
    a = ap.parse_args()
    cfg = load_config(a.config)
    grid = Grid.from_config(cfg)
    root, L = Path(cfg["paths"]["samples"]), Path(cfg["paths"]["labels"])

    idx = pd.read_csv(root / "index.csv")
    year = idx.event.str.extract(r"_(\d{4})\d{4}_")[0].astype(int)
    sel = idx[(idx.split == "train") & (year <= a.last_year)].iloc[:: a.stride]
    files = [root / r.split / r.file for r in sel.itertuples()]
    print(f"{len(files)} training samples from {year[sel.index].min()}-{year[sel.index].max()}", flush=True)
    tot, n = None, 0
    with ProcessPoolExecutor(a.workers) as ex:
        for k, m in enumerate(ex.map(adp_one, files, chunksize=32), 1):
            if m is not None:
                tot = m.astype(np.int32) if tot is None else tot + m
                n += 1
            if k % 1000 == 0:
                print(f"  {k}/{len(files)}", flush=True)
    adp = tot / max(n, 1)

    W = pd.read_csv(L / "warnings.csv")
    W = W[(W.phenomena == "DS") & (W.year <= a.last_year)]
    lon, lat = np.meshgrid(grid.lons(), grid.lats())
    pts = np.column_stack([lon.ravel(), lat.ravel()])
    from matplotlib.path import Path as MplPath
    warn, n_poly = np.zeros(lon.shape, np.float32), 0
    for w in W.itertuples():
        pf = L / "polygons" / f"{w.wfo}_{w.year}_{w.eventid}.npz"
        if not pf.exists():
            continue
        with np.load(pf) as z:
            rings = [z[k] for k in z.files]
        if not rings:
            continue
        hit = np.zeros(len(pts), bool)
        for ring in rings:
            hit |= MplPath(ring[:, :2]).contains_points(pts)
        warn += hit.reshape(lon.shape)
        n_poly += 1

    out = []
    for name, m in (("satellite dust frequency", adp), ("warning frequency", warn)):
        s = smooth(m)
        q = float(np.percentile(s[s > 0], 99)) if (s > 0).any() else 1.0
        out.append(np.clip(s / q, 0, 1))
        print(f"{name}: non-zero cells {float((m > 0).mean()):.1%}, 99th percentile {q:.4g}", flush=True)
    out = np.stack(out).astype(np.float32)
    assert out.shape[1:] == (grid.ny, grid.nx), out.shape
    f = Path(cfg["paths"]["static"]) / f"hotspot_2018_{a.last_year}.npy"
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, out)
    st = pd.read_csv(L / "stations.csv")
    print(f"wrote {f} from {n} samples and {n_poly} warning polygons; "
          f"mean value at the ASOS stations: {out[0][st.i, st.j].mean():.2f} / {out[1][st.i, st.j].mean():.2f}", flush=True)
    sys.exit(0 if out[0].max() > 0 and out[1].max() > 0 else 1)


if __name__ == "__main__":
    main()
