"""Step 31: SEED ENSEMBLE - average the probabilities of several runs of ONE recipe (numpy only, no torch).

    python3 scripts/31_seed_ensemble.py --fold runs/gated_unet_v11f runs/gated_unet_v11f_s7 runs/gated_unet_v11f_s11 \
        --fold runs/gated_unet_v11f_fold1 runs/gated_unet_v11f_fold1_s7 runs/gated_unet_v11f_fold1_s11 --out results/seed_ensemble_v11f.json

Why: 17 Nov showed that the onset score of one recipe moves by 0.03-0.10 between random seeds. That variance is
noise in each member's probabilities, and averaging members removes noise that no single run can remove (a "deep
ensemble"). Unlike the max ensembles tried before (different recipes, element-wise maximum, which adds the members'
false alarms), the mean of same-recipe seeds keeps what the members agree on.
Each --fold lists the runs trained with the SAME validation fold; their eval_val_preds.npz rows must be identical.
Folds are pooled, so with two folds every 2025 event is scored out of sample. Reported per lead, ASOS stations
(Automated Surface Observing System): CSI (Critical Success Index) at the best threshold for all cases and for
onsets, onset CSI / POD / FAR at 0.5, the average over the single members for comparison, station persistence,
an event-bootstrap 95 % interval for the ensemble's onset CSI at 0.5, and the paired bootstrap of
(ensemble - average member) on the same resampled events.
"""
import argparse
import json
from pathlib import Path

import numpy as np

THS = np.arange(0.05, 0.96, 0.05)


def sc(p, t, thr):
    a = p >= thr
    h = (a & t).sum()
    return h / max((a | t).sum(), 1), h / max(t.sum(), 1), 1 - h / max(a.sum(), 1)


def best(p, t):
    return max(sc(p, t, th)[0] for th in THS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", nargs="+", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    ens, mem, Y, G, EV, S, PERS = [], [], [], [], [], [], []
    for runs in a.fold:
        zs = [np.load(Path(r) / "eval_val_preds.npz", allow_pickle=True) for r in runs]
        y = zs[0]["y"]
        for r, z in zip(runs, zs):
            assert z["y"].shape == y.shape and (z["y"] == y).all(), f"{r}: rows differ from {runs[0]}"
        P = np.stack([z["p_model"].astype(np.float64) for z in zs])          # (members, rows, leads)
        ens.append(P.mean(0)); mem.append(P)
        Y.append(y); G.append(zs[0]["genesis"]); EV.append(zs[0]["event"]); S.append(zs[0]["src"])
        PERS.append(zs[0]["p_station_persistence"].astype(np.float64))
    n_mem = min(m.shape[0] for m in mem)
    ens, Y, G, EV, S, PERS = (np.concatenate(v) for v in (ens, Y, G, EV, S, PERS))
    MEM = [np.concatenate([m[k] for m in mem]) for k in range(n_mem)]       # member k of every fold, pooled
    asos = S == 0
    events = np.unique(EV)
    idx = {e: np.flatnonzero(EV == e) for e in events}
    rng = np.random.default_rng(0)
    draws = [np.concatenate([idx[e] for e in rng.choice(events, len(events))]) for _ in range(a.B)]
    out = dict(folds=a.fold, members_per_fold=[m.shape[0] for m in mem], n_events=int(len(events)), leads={})
    for l, lead in enumerate((15, 30, 60)):
        t = Y[:, l] == 1
        m_all, m_on = asos & (Y[:, l] >= 0), asos & (Y[:, l] >= 0) & G
        e5 = sc(ens[m_on, l], t[m_on], 0.5)
        mem5 = [sc(p[m_on, l], t[m_on], 0.5) for p in MEM]
        d, ci = [], []
        for dr in draws:
            k = dr[m_on[dr]]
            ce = sc(ens[k, l], t[k], 0.5)[0]
            ci.append(ce); d.append(ce - np.mean([sc(p[k, l], t[k], 0.5)[0] for p in MEM]))
        r = dict(n_pos=int(t[m_all].sum()), n_pos_onset=int(t[m_on].sum()),
                 persistence_all_best=best(PERS[m_all, l], t[m_all]),
                 ensemble_all_best=best(ens[m_all, l], t[m_all]), members_all_best=[best(p[m_all, l], t[m_all]) for p in MEM],
                 ensemble_onset_best=best(ens[m_on, l], t[m_on]), members_onset_best=[best(p[m_on, l], t[m_on]) for p in MEM],
                 ensemble_onset_05=dict(csi=e5[0], pod=e5[1], far=e5[2], ci=[float(np.percentile(ci, 2.5)), float(np.percentile(ci, 97.5))]),
                 members_onset_05=[dict(csi=q[0], pod=q[1], far=q[2]) for q in mem5],
                 gain_onset_05=dict(mean=float(e5[0] - np.mean([q[0] for q in mem5])), lo=float(np.percentile(d, 2.5)),
                                    hi=float(np.percentile(d, 97.5)), p_le_0=float((np.array(d) <= 0).mean())))
        out["leads"][str(lead)] = r
        print(f"+{lead:<2d} all: persistence {r['persistence_all_best']:.3f} | members {np.mean(r['members_all_best']):.3f} | ENSEMBLE {r['ensemble_all_best']:.3f}"
              f"   onset@0.5: members {np.mean([q['csi'] for q in r['members_onset_05']]):.3f} | ENSEMBLE {e5[0]:.3f} "
              f"[{r['ensemble_onset_05']['ci'][0]:.3f}, {r['ensemble_onset_05']['ci'][1]:.3f}] POD {e5[1]:.2f} FAR {e5[2]:.2f}"
              f"   gain {r['gain_onset_05']['mean']:+.3f} [{r['gain_onset_05']['lo']:+.3f}, {r['gain_onset_05']['hi']:+.3f}] p {r['gain_onset_05']['p_le_0']:.3f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
