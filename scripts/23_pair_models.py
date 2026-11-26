"""Step 23: paired comparison of two recipes on the same validation rows, optionally pooled over folds.

    python scripts/23_pair_models.py --a runs/gated_unet_v14b runs/gated_unet_v14b_fold1 \
                                     --b runs/gated_unet_v11f runs/gated_unet_v11f_fold1 --out results/pair_radar.json

For each lead: ASOS-only (Automated Surface Observing System) station CSI (Critical Success Index) at the best
threshold and onset (genesis) CSI at 0.5 for A and B, the difference with an event-bootstrap 95 % interval, and the
paired p-value P(A <= B) from resampling events. Rows of the two recipes must be identical per fold.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dustahead.eval.metrics import station_scores  # noqa: E402


def load(runs):
    zs = [np.load(Path(r) / "eval_val_preds.npz", allow_pickle=True) for r in runs]
    cat = lambda k: np.concatenate([z[k] for z in zs])  # noqa: E731
    src = cat("src") if all("src" in z.files for z in zs) else np.zeros(len(cat("y")), int)
    return cat("p_model"), cat("y"), cat("genesis"), cat("event"), src


def csi(p, y, thr):
    return station_scores(p, y, thr)["csi"]


def best_thr(p, y):
    ts = np.arange(0.05, 0.96, 0.05)
    return float(ts[int(np.nanargmax([csi(p, y, t) for t in ts]))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    pa, y, g, ev, src = load(a.a)
    pb, yb, gb, evb, _ = load(a.b)
    assert pa.shape == pb.shape and (y == yb).all() and (ev == evb).all(), "the two recipes were scored on different rows"
    asos = src == 0
    events = np.unique(ev)
    idx = {e: np.flatnonzero(ev == e) for e in events}
    rng = np.random.default_rng(0)
    draws = [np.concatenate([idx[e] for e in rng.choice(events, len(events))]) for _ in range(a.B)]
    out = dict(a=a.a, b=a.b, n_rows=int(len(y)), n_events=int(len(events)), leads={})
    for l, lead in enumerate((15, 30, 60)):
        res = {}
        for tag, m, fixed in (("all_best", asos & (y[:, l] >= 0), None), ("onset_05", asos & g & (y[:, l] >= 0), 0.5)):
            ta = fixed if fixed is not None else best_thr(pa[m, l], y[m, l])
            tb = fixed if fixed is not None else best_thr(pb[m, l], y[m, l])
            d = []
            for dr in draws:
                k = dr[m[dr]]
                d.append(csi(pa[k, l], y[k, l], ta) - csi(pb[k, l], y[k, l], tb))
            d = np.array(d, float); d = d[np.isfinite(d)]
            res[tag] = dict(a=csi(pa[m, l], y[m, l], ta), b=csi(pb[m, l], y[m, l], tb), n_pos=int((y[m, l] == 1).sum()),
                            diff_lo=float(np.percentile(d, 2.5)), diff_hi=float(np.percentile(d, 97.5)), p_a_le_b=float((d <= 0).mean()))
        out["leads"][str(lead)] = res
        print(lead, {k: {q: round(v, 3) if isinstance(v, float) else v for q, v in r.items()} for k, r in res.items()}, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
