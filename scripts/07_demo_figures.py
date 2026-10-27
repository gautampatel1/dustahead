"""Step 7: run the product ensemble over one stored event and save figures + alerts for the slides/demo.

    python scripts/07_demo_figures.py --event dust_20250825_2236 [--split val] [--every 3] [--out results/demo]

For every k-th sample of the event: a 2x3 panel (Dust RGB with station reports, P(dust) at +15/+30/+60 with the
I-10 line, ground gate, painted target at +30) and the road alerts; plus alerts.csv for the whole event.
Uses product.checkpoints from the config (the max ensemble) unless --ckpt is given.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dustahead import load_config
from dustahead.app.predict import DustAhead, I10
from dustahead.data.sampler import load_sample, paint_station_now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--ckpt", action="append", default=None)
    ap.add_argument("--out", default="results/demo")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    ckpts = a.ckpt or cfg["product"]["checkpoints"]
    E = DustAhead(ckpts, cfg)
    files = sorted(Path(cfg["paths"]["samples"], a.split).glob(f"{a.event}_*.npz"))
    if not files:
        sys.exit(f"no samples for {a.event} in {a.split}")
    out = Path(a.out) / a.event
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext = [E.grid.lon_min, E.grid.lon_max, E.grid.lat_min, E.grid.lat_max]
    lon10, lat10 = zip(*I10)
    rows = []
    for k, f in enumerate(files):
        s = load_sample(f)
        ij, now = s["st_ij"], s["st_now"]
        r = E.predict_array(s["x"].astype(np.float32), paint_station_now(ij, now, s["x"].shape[1:])[None], ij, now)
        t0 = str(s["t0"])[:16]
        for al in r["alerts"]:
            rows.append(dict(t0=t0, **al))
        n_pos = [int((s["st_y"][:, l] == 1).sum()) for l in range(3)]
        print(f"{t0}  dust-now stations {int((now == 1).sum()):2d}  positives +15/+30/+60 {n_pos}  "
              f"maxP {[round(float(r['probs'][l].max()), 2) for l in range(3)]}  alerts {len(r['alerts'])}")
        if k % a.every:
            continue
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        lons, lats = E.grid.lons()[ij[:, 1]], E.grid.lats()[ij[:, 0]]

        def road(ax):
            ax.plot(lon10, lat10, color="white", lw=1.5, alpha=0.9)
            ax.scatter(lons[now == -1], lats[now == -1], s=6, c="lightgrey", alpha=0.6, zorder=4)
            ax.scatter(lons[now == 1], lats[now == 1], s=25, c="red", edgecolors="white", lw=0.6, zorder=5)
            ax.set_axis_off()

        axs[0, 0].imshow(s["dust_rgb"], origin="lower", extent=ext); road(axs[0, 0]); axs[0, 0].set_title(f"Dust RGB at t0 = {t0}Z (red = station reports dust)")
        for l, lead in enumerate(E.leads):
            ax = axs[0, l + 1] if l < 2 else axs[1, 0]
            im = ax.imshow(r["probs"][l], origin="lower", extent=ext, vmin=0, vmax=1, cmap="magma"); road(ax)
            ax.set_title(f"P(dust) +{lead} min (max ensemble)"); plt.colorbar(im, ax=ax, fraction=0.035)
        axs[1, 1].imshow(r["gate"], origin="lower", extent=ext, vmin=0, vmax=1, cmap="YlOrBr"); road(axs[1, 1]); axs[1, 1].set_title("Ground gate σ(h): can the surface emit?")
        axs[1, 2].imshow(s["y"][1], origin="lower", extent=ext, cmap="Greys"); road(axs[1, 2]); axs[1, 2].set_title("Target at +30 min (stations + ADP pixels)")
        al = r["alerts"]
        fig.suptitle(f"DustAhead · {a.event} · {len(al)} I-10 segment alert(s)" + (f" · earliest +{min(x['lead_min'] for x in al)} min" if al else ""), fontsize=14)
        fig.tight_layout()
        fig.savefig(out / f"{k:03d}_{t0.replace(':', '')}.png", dpi=110)
        plt.close(fig)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / "alerts.csv", index=False)
    print(f"wrote {len(list(out.glob('*.png')))} figures and {len(rows)} alert rows to {out}")


if __name__ == "__main__":
    main()
