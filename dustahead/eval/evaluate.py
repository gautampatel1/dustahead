"""Full evaluation of a checkpoint (and the baselines) on one split: station metrics, genesis
stratification, FSS, calibration, bootstrap CIs. Writes <run_dir>/eval_<split>.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.sampler import load_sample
from ..models.baselines import BASELINES
from ..train.dataset import DustDataset
from ..train.train import load_checkpoint
from .bootstrap import event_bootstrap, holm, paired_bootstrap_pvalue
from .metrics import fss_np, station_scores, temperature_scale, apply_temperature


def _station_values(p, st_ij, st_y):
    i, j = st_ij[:, 0], st_ij[:, 1]
    return p[:, i, j].T, st_y          # (S, L), (S, L)


def collect(model, ds: DustDataset, device, baselines=()):
    """Returns per-model station probabilities (N, L), labels (N, L), genesis flags, event ids, FSS lists."""
    out = {name: [] for name in ("model",) + tuple(baselines)}
    ys, gen, ev, fss_pix = [], [], [], {name: [] for name in out}
    for k in range(len(ds)):
        s = load_sample(ds.files[k])
        x = s["x"].astype(np.float32)
        with torch.no_grad():
            p = model(torch.from_numpy(x)[None].to(device))[0].cpu().numpy()
        preds = {"model": p, **{b: BASELINES[b](x) for b in baselines}}
        ij, sy = s["st_ij"].astype(int), s["st_y"].astype(int)
        for name, pp in preds.items():
            sp, _ = _station_values(pp, ij, sy)
            out[name].append(sp)
            fss_pix[name].append([fss_np(pp[l], s["y"][l]) for l in range(pp.shape[0])])
        ys.append(sy)
        gen.append(np.full(len(sy), bool(s["genesis"])))
        ev.append(np.full(len(sy), str(ds.files[k].stem).rsplit("_", 1)[0]))
    return ({n: np.concatenate(v) for n, v in out.items()}, np.concatenate(ys), np.concatenate(gen),
            np.concatenate(ev), {n: np.nanmean(v, 0).tolist() for n, v in fss_pix.items()})


def run(ckpt: str | Path, cfg: dict, split: str = "test", baselines=("persistence", "advected", "hrrr_rule"),
        device: str | None = None, calibrate_on: str = "val") -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(ckpt, device)
    leads = cfg["time"]["leads_min"]
    thr, B = cfg["eval"]["threshold"], cfg["eval"]["bootstrap"]

    # temperature scaling fitted on the validation split, applied to the evaluated split
    T = 1.0
    if calibrate_on and calibrate_on != split:
        pv, yv, *_ = collect(model, DustDataset(cfg["paths"]["samples"], calibrate_on), device)
        T = temperature_scale(pv["model"].ravel(), yv.ravel())
    preds, y, gen, ev, fss_pix = collect(model, DustDataset(cfg["paths"]["samples"], split), device, baselines)
    preds["model"] = apply_temperature(preds["model"], T)

    report = dict(split=split, checkpoint=str(ckpt), temperature=T, leads=leads, models={})
    for name, p in preds.items():
        r = dict(fss_pixels=fss_pix[name])
        for l, lead in enumerate(leads):
            r[f"all_{lead}"] = station_scores(p[:, l], y[:, l], thr)
            r[f"genesis_{lead}"] = station_scores(p[gen, l], y[gen, l], thr)
            r[f"genesis_{lead}"]["csi_ci"] = event_bootstrap(p[gen, l], y[gen, l], ev[gen], "csi", B, thr)
        report["models"][name] = r
    pv = {}
    for name in baselines:
        l = leads.index(30)
        pv[name] = paired_bootstrap_pvalue(preds["model"][gen, l], preds[name][gen, l], y[gen, l], ev[gen], "csi", B, thr)
    report["model_vs_baselines_p_genesis30"] = pv
    report["holm_corrected"] = holm(pv) if pv else {}
    out = Path(ckpt).parent / f"eval_{split}.json"
    out.write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != "models"}, indent=1, default=float))
    for name, r in report["models"].items():
        print(name, {k: round(v["csi"], 3) for k, v in r.items() if k.startswith("genesis")})
    return report
