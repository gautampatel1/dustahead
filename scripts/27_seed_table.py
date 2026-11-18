"""Step 27: compare two recipes over several random seeds each (mean and spread), from their eval_val.json files.

    python scripts/27_seed_table.py --a runs/gated_unet_v17_s42 runs/gated_unet_v17_s7 runs/gated_unet_v17_s11 \
        --b runs/gated_unet_v11f runs/gated_unet_v11f_s7 runs/gated_unet_v11f_s11 --names hotspot baseline --out results/seeds_v17.json

17 Nov: the onset score of ONE recipe moves by 0.03-0.10 between seeds, as much as any difference between recipes.
A recipe is therefore judged on the mean of at least three seeds, with the spread shown, and on a seed-level test:
Welch's t on the per-seed scores (small n: read it as a rough guide, not a certificate).
ASOS stations (Automated Surface Observing System); CSI = Critical Success Index.
"""
import argparse
import json
from pathlib import Path

import numpy as np

KEYS = [("all, best threshold", "asos_{L}", "csi_best"), ("onset at 0.5", "asos_genesis_{L}", "csi"),
        ("onset POD at 0.5", "asos_genesis_{L}", "pod"), ("onset FAR at 0.5", "asos_genesis_{L}", "far")]


def welch(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or len(y) < 2:
        return None
    se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    return float((x.mean() - y.mean()) / se) if se > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    load = lambda runs: [json.load(open(Path(r) / f"eval_{a.split}.json"))["models"]["model"] for r in runs]  # noqa: E731
    A, B = load(a.a), load(a.b)
    out = dict(a=a.a, b=a.b, names=a.names, rows={})
    for title, key, field in KEYS:
        for L in (15, 30, 60):
            va = [m[key.format(L=L)][field] for m in A]
            vb = [m[key.format(L=L)][field] for m in B]
            r = dict(a=va, b=vb, mean_a=float(np.nanmean(va)), mean_b=float(np.nanmean(vb)), t=welch(va, vb))
            out["rows"][f"{title} +{L}"] = r
            print(f"{title:22s} +{L:<2d} | {a.names[0]} {r['mean_a']:.3f} [{np.nanmin(va):.3f}, {np.nanmax(va):.3f}] | "
                  f"{a.names[1]} {r['mean_b']:.3f} [{np.nanmin(vb):.3f}, {np.nanmax(vb):.3f}] | t {r['t'] if r['t'] is None else round(r['t'], 2)}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
