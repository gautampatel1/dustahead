"""Step 11: evaluate ENSEMBLES and the ALERT POLICY offline, from the per-station predictions that 05_evaluate saves.

    python scripts/11_ensemble_eval.py --runs runs/unet_all_v4 runs/gated_unet_v4 runs/gated_unet_v8   # max ensemble
    python scripts/11_ensemble_eval.py --runs runs/gated_unet_v9 --persistence-leads 15                # policy variant
    python scripts/11_ensemble_eval.py --pool runs/gated_unet_v8 runs/gated_unet_v8_fold1              # 2-fold CV pooled

--runs  : element-wise MAX over the runs' calibrated model probabilities (all runs must be evaluated on the same
          split and fold, so their station labels line up; this is checked).
--pool  : CONCATENATE runs evaluated on different folds of the same split (cross-validation): every 2025 event is
          then scored out of sample once, and the onset confidence intervals rest on twice the positives.
--persistence-leads : floor the forecast with station persistence at these leads (the product policy; default 15).
Reports, per lead: CSI at the best threshold (sweep 0.05..0.95), the threshold, POD/FAR there, and the GENESIS CSI
at that threshold with its event-bootstrap 95 % CI. No training data is touched, so this is safe to run on val as
often as needed; on test it is a second look and must be reported as such.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dustahead import load_config
from dustahead.eval.bootstrap import event_bootstrap, paired_bootstrap_pvalue
from dustahead.eval.metrics import station_scores

THRS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def load(run: str, split: str) -> dict:
    f = Path(run) / f"eval_{split}_preds.npz"
    if not f.exists():
        sys.exit(f"{f} missing: run 05_evaluate.py --split {split} on that checkpoint first")
    z = np.load(f, allow_pickle=True)
    return {k: z[k] for k in z.files}


def best_threshold(p, y):
    return max(((station_scores(p, y, t)["csi"], float(t)) for t in THRS), key=lambda z: (-1 if np.isnan(z[0]) else z[0]))


def report(name: str, p: np.ndarray, y: np.ndarray, gen: np.ndarray, ev: np.ndarray, leads, B: int,
           p_ref: np.ndarray | None = None) -> dict:
    out = {}
    for l, lead in enumerate(leads):
        csi, thr = best_threshold(p[:, l], y[:, l])
        s = station_scores(p[:, l], y[:, l], thr)
        g = station_scores(p[gen, l], y[gen, l], thr)
        bs = event_bootstrap(p[gen, l], y[gen, l], ev[gen], "csi", B, thr)
        ci = [bs["lo"], bs["hi"]]
        row = dict(csi_best=csi, thr=thr, pod=s["pod"], far=s["far"], genesis_csi=g["csi"], genesis_pod=g["pod"],
                   genesis_far=g["far"], genesis_ci=ci, n_pos=int((y[:, l] == 1).sum()), n_pos_genesis=int((y[gen, l] == 1).sum()))
        if p_ref is not None:
            row["p_vs_ref_genesis"] = paired_bootstrap_pvalue(p[gen, l], p_ref[gen, l], y[gen, l], ev[gen], "csi", B, thr)
        out[f"lead{lead}"] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[], help="run dirs to MAX-ensemble (same split/fold)")
    ap.add_argument("--pool", nargs="*", default=[], help="run dirs on different folds to CONCATENATE")
    ap.add_argument("--split", default="val")
    ap.add_argument("--persistence-leads", nargs="*", type=int, default=[15])
    ap.add_argument("--out", default=None, help="JSON path (default: results/<split>_ensemble_<names>.json)")
    a = ap.parse_args()
    cfg = load_config()
    leads, B = cfg["time"]["leads_min"], cfg["eval"]["bootstrap"]
    if bool(a.runs) == bool(a.pool):
        sys.exit("give --runs (ensemble) or --pool (cross-validation), not both")

    if a.runs:
        zs = [load(r, a.split) for r in a.runs]
        y, gen, ev = zs[0]["y"], zs[0]["genesis"], zs[0]["event"]
        for z in zs[1:]:
            if not np.array_equal(z["y"], y):
                sys.exit("station labels differ between runs: they were not evaluated on the same split/fold")
        p_model = np.max(np.stack([z["p_model"] for z in zs]), 0)
        p_pers = zs[0].get("p_station_persistence")
        members = {Path(r).name: z["p_model"] for r, z in zip(a.runs, zs)}
        tag = "+".join(Path(r).name for r in a.runs)
    else:
        zs = [load(r, a.split) for r in a.pool]
        y, gen, ev = (np.concatenate([z[k] for z in zs]) for k in ("y", "genesis", "event"))
        p_model = np.concatenate([z["p_model"] for z in zs])
        p_pers = np.concatenate([z["p_station_persistence"] for z in zs]) if "p_station_persistence" in zs[0] else None
        members = {}
        tag = "pool_" + "+".join(Path(r).name for r in a.pool)

    res = dict(split=a.split, runs=a.runs or a.pool, mode="max" if a.runs else "pool", leads=leads,
               n_stations=int(len(y)), n_events=int(len(np.unique(ev))), models={})
    if p_pers is not None:
        res["models"]["station_persistence"] = report("station_persistence", p_pers, y, gen, ev, leads, B)
    for n, pm in members.items():
        res["models"][n] = report(n, pm, y, gen, ev, leads, B, p_pers)
    res["models"]["ensemble" if a.runs else "pooled"] = report("ens", p_model, y, gen, ev, leads, B, p_pers)
    if p_pers is not None and a.persistence_leads:
        pp = p_model.copy()
        for lead in a.persistence_leads:
            if lead in leads:
                pp[:, leads.index(lead)] = np.maximum(pp[:, leads.index(lead)], p_pers[:, leads.index(lead)])
        res["models"][f"ensemble+persistence@{','.join(map(str, a.persistence_leads))}"] = report("pol", pp, y, gen, ev, leads, B, p_pers)

    print(f"\n{a.split} · {res['n_stations']} station-times · {res['n_events']} events · mode {res['mode']}")
    print(f"{'model':44s} " + " ".join(f"| CSI+{l} (thr)    gen CSI [95% CI]      " for l in leads))
    for n, r in res["models"].items():
        cells = []
        for l in leads:
            q = r[f"lead{l}"]
            cells.append(f"| {q['csi_best']:.3f} ({q['thr']:.2f})  {q['genesis_csi']:.3f} [{q['genesis_ci'][0]:.2f},{q['genesis_ci'][1]:.2f}]")
        print(f"{n:44s} " + " ".join(cells))
    out = Path(a.out) if a.out else Path("results") / f"{a.split}_ensemble_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    print("wrote", out)


if __name__ == "__main__":
    main()
