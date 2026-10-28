"""Step 9: slide media for the results talk — real figures, no simulation.

    python scripts/09_slide_media.py --out results/slide_media

Writes
  replay/<event>_<k>_<t0>.png   2-panel strips (Dust RGB with station reports + I-10 | ensemble P(dust) at +30)
  gate_<event>_<t0>.png         gate σ(h) of gated_unet v4 (closed) next to v8 (open) on the same sample
  curves.json                   POD / FAR / CSI vs threshold at +15/+30/+60 for the max ensemble, all and genesis
                                stations, from runs/*/eval_val_preds.npz (real validation numbers)
  unet_arch.json                channel counts of the U-Net for the architecture figure
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dustahead import load_config
from dustahead.app.predict import DustAhead, I10
from dustahead.data.sampler import load_sample, paint_station_now

REPLAY = {  # event -> (first t0, last t0) shown in the replay, every 20 min
    "dust_20250825_2236": ("20250825T2100", "20250826T0040"),
    "dust_20250307_1821": ("20250307T1900", "20250307T2300"),
}
GATE_SAMPLES = ["dust_20250825_2236_20250825T2300", "dust_20250307_1821_20250307T2120"]


def strip(E, s, r, out, lead_idx=1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext = [E.grid.lon_min, E.grid.lon_max, E.grid.lat_min, E.grid.lat_max]
    ij, now = s["st_ij"], s["st_now"]
    lons, lats = E.grid.lons()[ij[:, 1]], E.grid.lats()[ij[:, 0]]
    lon10, lat10 = zip(*I10)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.0), facecolor="#1d1611")

    def road(ax):
        ax.plot(lon10, lat10, color="white", lw=1.6, alpha=0.9)
        ax.scatter(lons[now == -1], lats[now == -1], s=8, c="#cfc3b2", alpha=0.7, zorder=4)
        ax.scatter(lons[now == 1], lats[now == 1], s=42, c="#ff3b3b", edgecolors="white", lw=0.8, zorder=5)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_axis_off()

    axs[0].imshow(s["dust_rgb"], origin="lower", extent=ext); road(axs[0])
    axs[1].imshow(r["probs"][lead_idx], origin="lower", extent=ext, vmin=0, vmax=1, cmap="magma"); road(axs[1])
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005, wspace=0.02)
    fig.savefig(out, dpi=100, facecolor="#1d1611")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/slide_media")
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    cfg = load_config()
    out = Path(a.out); (out / "replay").mkdir(parents=True, exist_ok=True)
    E = DustAhead(cfg["product"]["checkpoints"], cfg)
    root = Path(cfg["paths"]["samples"], a.split)

    # 1. replay strips
    for ev, (t_a, t_b) in REPLAY.items():
        files = sorted(root.glob(f"{ev}_*.npz"))
        files = [f for f in files if t_a <= f.stem.rsplit("_", 1)[1] <= t_b][::2]
        rows = []
        for k, f in enumerate(files):
            s = load_sample(f)
            r = E.predict_array(s["x"].astype(np.float32), paint_station_now(s["st_ij"], s["st_now"], s["x"].shape[1:])[None],
                                s["st_ij"], s["st_now"])
            t0 = f.stem.rsplit("_", 1)[1]
            strip(E, s, r, out / "replay" / f"{ev}_{k:02d}_{t0}.png")
            rows.append(dict(k=k, t0=t0, dust_now=int((s["st_now"] == 1).sum()),
                             pos=[int((s["st_y"][:, l] == 1).sum()) for l in range(3)],
                             pmax=[round(float(r["probs"][l].max()), 3) for l in range(3)],
                             alerts=len(r["alerts"])))
            print(ev, t0, rows[-1])
        (out / "replay" / f"{ev}.json").write_text(json.dumps(rows, indent=1))

    # 2. gate before / after on the same sample
    import torch
    from dustahead.train.train import load_checkpoint
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext = [E.grid.lon_min, E.grid.lon_max, E.grid.lat_min, E.grid.lat_max]
    lon10, lat10 = zip(*I10)
    m4, _ = load_checkpoint("runs/gated_unet_v4/best.pt", E.device)
    m8, _ = load_checkpoint("runs/gated_unet_v8/best.pt", E.device)
    for name in GATE_SAMPLES:
        s = load_sample(root / f"{name}.npz")
        x = s["x"].astype(np.float32)
        x = np.concatenate([x, paint_station_now(s["st_ij"], s["st_now"], x.shape[1:])[None]], 0)
        xt = torch.from_numpy(x)[None].to(E.device)
        gates = []
        with torch.no_grad():
            for m in (m4, m8):
                _, _, h = m(xt, return_parts=True)
                gates.append(torch.sigmoid(h)[0, 0].float().cpu().numpy())
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.0), facecolor="#1d1611")
        for ax, g in zip(axs, gates):
            ax.imshow(g, origin="lower", extent=ext, vmin=0, vmax=1, cmap="YlOrBr")
            ax.plot(lon10, lat10, color="#1d1611", lw=1.6, alpha=0.9); ax.set_axis_off()
        fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005, wspace=0.02)
        fig.savefig(out / f"gate_{name}.png", dpi=100, facecolor="#1d1611"); plt.close(fig)
        print("gate", name, [round(float(g.mean()), 3) for g in gates])

    # 3. threshold curves (real validation predictions)
    P = {Path(c).parent.name: np.load(Path(c).parent / "eval_val_preds.npz", allow_pickle=True) for c in cfg["product"]["checkpoints"]}
    first = next(iter(P.values()))
    y, gen, ps = first["y"], first["genesis"], first["p_station_persistence"]
    pe = np.maximum.reduce([v["p_model_raw"] for v in P.values()])
    ths = np.round(np.arange(0.05, 0.96, 0.05), 2)

    def counts(p, yy, t):
        m = yy >= 0; f = p[m] >= t; pos = yy[m] == 1
        h, fa, mi = int((f & pos).sum()), int((f & ~pos).sum()), int((~f & pos).sum())
        return dict(hits=h, fa=fa, miss=mi, pod=h / max(h + mi, 1), far=fa / max(h + fa, 1), csi=h / max(h + fa + mi, 1))

    curves = dict(thresholds=ths.tolist(), leads=cfg["time"]["leads_min"], ensemble={}, persistence={})
    for sub, mask in (("all", np.ones(len(y), bool)), ("genesis", gen)):
        curves["ensemble"][sub] = [[counts(pe[mask, l], y[mask, l], t) for t in ths] for l in range(3)]
        curves["persistence"][sub] = [counts(ps[mask, l], y[mask, l], 0.5) for l in range(3)]
    (out / "curves.json").write_text(json.dumps(curves))
    (out / "unet_arch.json").write_text(json.dumps(dict(cin=63, base=cfg["model"]["base_channels"], depth=cfg["model"]["depth"], leads=3)))
    print("wrote", out)


if __name__ == "__main__":
    main()
