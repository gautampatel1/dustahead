"""Pooled 2-fold cross-validation onset (genesis) metrics at the evaluation threshold (0.5), with event-bootstrap
confidence intervals and the paired p-value against station persistence. Reads the eval_val_preds.npz files that
05_evaluate.py wrote for a fold-0 run and a fold-1 run of the same recipe.

    python scripts/12_pool_genesis.py --runs runs/gated_unet_v8 runs/gated_unet_v8_fold1 --out results/val_pool_v8_2fold_genesis05.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dustahead.eval.metrics import station_scores
from dustahead.eval.bootstrap import event_bootstrap, paired_bootstrap_pvalue

ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--B", type=int, default=1000); ap.add_argument("--thr", type=float, default=0.5)
a = ap.parse_args()
zs = [np.load(Path(r) / "eval_val_preds.npz", allow_pickle=True) for r in a.runs]
y = np.concatenate([z["y"] for z in zs]); g = np.concatenate([z["genesis"] for z in zs]); ev = np.concatenate([z["event"] for z in zs])
pp = np.concatenate([z["p_station_persistence"] for z in zs])
out = dict(runs=a.runs, threshold=a.thr, n_stations=int(len(y)), n_events=int(len(np.unique(ev))),
           pos=(y == 1).sum(0).tolist(), gen_pos=(y[g] == 1).sum(0).tolist(), results={})
for key in ("p_model", "p_model_raw"):
    p = np.concatenate([z[key] for z in zs]); out["results"][key] = {}
    for l, lead in enumerate((15, 30, 60)):
        s = station_scores(p[g, l], y[g, l], a.thr); bs = event_bootstrap(p[g, l], y[g, l], ev[g], "csi", a.B, a.thr)
        pv = paired_bootstrap_pvalue(p[g, l], pp[g, l], y[g, l], ev[g], "csi", a.B, a.thr)
        al = station_scores(p[:, l], y[:, l], a.thr); ps = station_scores(pp[:, l], y[:, l], a.thr)
        out["results"][key][lead] = dict(gen_csi=s["csi"], gen_pod=s["pod"], gen_far=s["far"], lo=bs["lo"], hi=bs["hi"],
                                         p_vs_station_persistence=pv, all_csi=al["csi"], persistence_all_csi=ps["csi"])
        print(key, lead, f"genesis CSI {s['csi']:.3f} [{bs['lo']:.3f},{bs['hi']:.3f}] POD {s['pod']:.2f} FAR {s['far']:.2f} p={pv:.4f} | all {al['csi']:.3f} vs persistence {ps['csi']:.3f}", flush=True)
Path(a.out).write_text(json.dumps(out, indent=1, default=float)); print("wrote", a.out)
