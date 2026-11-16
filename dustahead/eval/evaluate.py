"""Full evaluation of a checkpoint (and the baselines) on one split: station metrics, genesis
stratification, FSS, calibration, bootstrap CIs. Writes <run_dir>/eval_<split>.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.sampler import genesis_per_station as _genesis_ij, input_stack, load_sample, radar_drop_newest, radar_key, wants_forecast, wants_radar, wants_wind
from ..models.baselines import BASELINES, SAMPLE_BASELINES
from ..train.dataset import DustDataset
from ..train.train import load_checkpoint
from .bootstrap import event_bootstrap, holm, paired_bootstrap_pvalue
from .metrics import fss_np, station_scores, temperature_scale, apply_temperature, platt_scale, apply_platt


def genesis_per_station(s: dict, radius_cells: int = 12) -> np.ndarray:
    """Per station: True when no station within `radius_cells` (incl. itself) reports dust at t0."""
    return _genesis_ij(s["st_ij"], s["st_now"], radius_cells)


def _station_values(p, st_ij, st_y):
    i, j = st_ij[:, 0], st_ij[:, 1]
    return p[:, i, j].T, st_y          # (S, L), (S, L)


def collect(model, ds: DustDataset, device, baselines=(), with_forecast: bool = False, with_radar: bool = False,
            radar_drop: int = 0, rkey: str = "xr", with_wind: bool = False, wind_out: dict | None = None):
    """Returns per-model station probabilities (N, L), labels (N, L), genesis flags, event ids, FSS lists.
    wind_out (optional dict) is filled with the wind head's station probabilities `p` (N, L), the observed peak wind
    at t0 + lead `kt` (N, L, NaN = no report) and at t0 `now` (N,), for models that have a wind head."""
    out = {name: [] for name in ("model",) + tuple(baselines)}
    ys, gen, ev, src, fss_pix = [], [], [], [], {name: [] for name in out}
    for k in range(len(ds)):
        s = load_sample(ds.files[k])
        x = input_stack(s, None, with_forecast, with_radar, radar_drop, rkey, with_wind)
        with torch.no_grad():
            p = model(torch.from_numpy(x)[None].to(device))[0].cpu().numpy()
        wl = getattr(getattr(model, "unet", None), "last_wind", None)
        if wind_out is not None and wl is not None and "st_wy" in s:
            pw = torch.sigmoid(wl[0].float()).cpu().numpy()
            wind_out.setdefault("p", []).append(pw[:, s["st_ij"][:, 0].astype(int), s["st_ij"][:, 1].astype(int)].T)
            wind_out.setdefault("kt", []).append(s["st_wy"].astype(np.float32))
            wind_out.setdefault("now", []).append(s["st_wnow"][:, 0].astype(np.float32))
        preds = {"model": p, **{b: (BASELINES[b](x) if b in BASELINES else SAMPLE_BASELINES[b](s)) for b in baselines}}
        ij, sy = s["st_ij"].astype(int), s["st_y"].astype(int)
        for name, pp in preds.items():
            sp, _ = _station_values(pp, ij, sy)
            out[name].append(sp)
            fss_pix[name].append([fss_np(pp[l], s["y"][l]) for l in range(pp.shape[0])])
        ys.append(sy)
        gen.append(genesis_per_station(s, radius_cells=12))      # 50 km / 4 km
        ev.append(np.full(len(sy), str(ds.files[k].stem).rsplit("_", 1)[0]))
        src.append(s["st_src"].astype(int) if "st_src" in s else np.zeros(len(sy), int))   # 0 = ASOS (see 16_extend_labels)
    return ({n: np.concatenate(v) for n, v in out.items()}, np.concatenate(ys), np.concatenate(gen),
            np.concatenate(ev), {n: np.nanmean(v, 0).tolist() for n, v in fss_pix.items()}, np.concatenate(src))


def wind_scores(p: np.ndarray, kt: np.ndarray, now: np.ndarray, leads, thr_kt: float) -> dict:
    """Verification of the wind head. Event = peak wind (max of sustained and gust) >= thr_kt at t0 + lead.
    `all`: every station-time with a report; `onset`: stations whose wind at t0 was below the threshold (or not
    reported), where wind persistence scores zero by construction. CSI at the best probability threshold."""
    out = dict(threshold_kt=thr_kt)
    calm = ~(np.nan_to_num(now, nan=0.0) >= thr_kt)
    for l, lead in enumerate(leads):
        ok = np.isfinite(kt[:, l])
        y = (kt[:, l] >= thr_kt).astype(int)
        pers = (np.nan_to_num(now, nan=0.0) >= thr_kt).astype(float)
        for tag, m in (("all", ok), ("onset", ok & calm)):
            best = max((dict(station_scores(p[m, l], y[m], t), thr=float(t)) for t in np.arange(0.05, 0.96, 0.05)),
                       key=lambda z: -1 if np.isnan(z["csi"]) else z["csi"])
            out[f"{tag}_{lead}"] = dict(csi_best=best["csi"], thr=best["thr"], pod=best["pod"], far=best["far"],
                                        n_pos=int(y[m].sum()), n=int(m.sum()),
                                        persistence_csi=station_scores(pers[m], y[m], 0.5)["csi"])
    return out


def run(ckpt: str | Path, cfg: dict, split: str = "test", baselines=("persistence", "advected", "hrrr_rule"),
        device: str | None = None, calibrate_on: str = "val") -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(ckpt, device)
    wf = wants_forecast(ck["cfg"])          # the checkpoint knows whether it was trained with the forecast block
    wr = wants_radar(ck["cfg"])
    rd = radar_drop_newest(ck["cfg"])
    rk = radar_key(ck["cfg"])       # a latency-trained model is also evaluated without the newest frame(s)
    ww = wants_wind(ck["cfg"])
    wind_out: dict = {}
    leads = cfg["time"]["leads_min"]
    thr, B = cfg["eval"]["threshold"], cfg["eval"]["bootstrap"]

    # temperature scaling fitted on the validation split, applied to the evaluated split
    T, t_note = 1.0, "none"
    if str(ck["cfg"].get("splits", {}).get("val_fold", 0)).lower() == "none" and calibrate_on == "val":
        # a final model (splits.val_fold none) trained on every validation event: fitting a temperature there would be
        # in-sample, so its scores stay raw (T = 1)
        calibrate_on, t_note = None, "none (model trained on the validation events)"
    if calibrate_on and calibrate_on != split:
        pv, yv, *_ = collect(model, DustDataset(cfg["paths"]["samples"], calibrate_on, cfg=cfg), device, with_forecast=wf, with_radar=wr, radar_drop=rd, rkey=rk, with_wind=ww)
        T, t_note = temperature_scale(pv["model"].ravel(), yv.ravel()), f"fitted on {calibrate_on}"
    preds, y, gen, ev, fss_pix, src = collect(model, DustDataset(cfg["paths"]["samples"], split, cfg=cfg), device, baselines, with_forecast=wf, with_radar=wr, radar_drop=rd, rkey=rk, with_wind=ww, wind_out=wind_out)
    asos = src == 0
    if calibrate_on == split:            # two-parameter Platt fit on the evaluated split itself; say so in the report
        a, b = platt_scale(preds["model"].ravel(), y.ravel())
        T, t_note = 1.0 / max(a, 1e-3), f"Platt a={a:.3f} b={b:.3f} fitted in-sample on {split} (2 parameters)"
        preds["model_raw"] = preds["model"]
        fss_pix["model_raw"] = fss_pix["model"]      # same pixels: Platt is monotone, FSS is threshold-free here
        preds["model"] = apply_platt(preds["model"], a, b)
    else:
        preds["model"] = apply_temperature(preds["model"], T)
    # product policy variant: the model floored by station persistence at the short leads (app/predict.py does the
    # same on the map; at station level the element-wise max is identical). Evaluated whenever the baseline is present.
    pl = cfg.get("product", {}).get("persistence_leads", [15]) or []
    if "station_persistence" in preds and pl:
        pp = preds["model"].copy()
        for lead in pl:
            if lead in leads:
                pp[:, leads.index(lead)] = np.maximum(pp[:, leads.index(lead)], preds["station_persistence"][:, leads.index(lead)])
        preds["model+persistence"] = pp
        fss_pix["model+persistence"] = fss_pix["model"]      # pixel FSS is not recomputed for the policy variant

    report = dict(split=split, checkpoint=str(ckpt), temperature=T, temperature_note=t_note, leads=leads,
                  n_stations=int(len(y)), n_asos=int(asos.sum()), extended_labels=bool(not asos.all()), models={})
    for name, p in preds.items():
        r = dict(fss_pixels=fss_pix[name])
        for l, lead in enumerate(leads):
            r[f"all_{lead}"] = station_scores(p[:, l], y[:, l], thr)
            r[f"all_{lead}"]["csi_best"], r[f"all_{lead}"]["thr_best"] = max(
                ((station_scores(p[:, l], y[:, l], t)["csi"], float(t)) for t in np.arange(0.05, 0.96, 0.05)),
                key=lambda z: (-1 if np.isnan(z[0]) else z[0]))
            r[f"genesis_{lead}"] = station_scores(p[gen, l], y[gen, l], thr)
            r[f"genesis_{lead}"]["csi_ci"] = event_bootstrap(p[gen, l], y[gen, l], ev[gen], "csi", B, thr)
            if not asos.all():       # extended benchmark: also the ASOS-only view, comparable with earlier tables
                r[f"asos_{lead}"] = station_scores(p[asos, l], y[asos, l], thr)
                r[f"asos_{lead}"]["csi_best"] = max((station_scores(p[asos, l], y[asos, l], t)["csi"] for t in np.arange(0.05, 0.96, 0.05)),
                                                    key=lambda z: -1 if np.isnan(z) else z)
                ga = gen & asos
                r[f"asos_genesis_{lead}"] = station_scores(p[ga, l], y[ga, l], thr)
                for name_src, code in (("rwis", 1), ("vis_other", 2), ("pm10", 3)):
                    m = src == code
                    if m.any():
                        r[f"{name_src}_{lead}"] = station_scores(p[m, l], y[m, l], thr)
        report["models"][name] = r
    if wind_out.get("p"):       # wind head: skill at "peak wind >= wind_kt at t0 + lead", against wind persistence
        report["wind"] = wind_scores(np.concatenate(wind_out["p"]), np.concatenate(wind_out["kt"]),
                                     np.concatenate(wind_out["now"]), leads, float(ck["cfg"].get("loss", {}).get("wind_kt", 30.0)))
    pv = {}
    for name in baselines:
        l = leads.index(30)
        pv[name] = paired_bootstrap_pvalue(preds["model"][gen, l], preds[name][gen, l], y[gen, l], ev[gen], "csi", B, thr)
    report["model_vs_baselines_p_genesis30"] = pv
    report["holm_corrected"] = holm(pv) if pv else {}
    # per-station predictions for offline analysis (hybrids, genesis threshold sweeps) without re-running inference
    extra = dict(wind_p=np.concatenate(wind_out["p"]), wind_kt=np.concatenate(wind_out["kt"]),
                 wind_now=np.concatenate(wind_out["now"])) if wind_out.get("p") else {}     # wind head, same row order
    np.savez_compressed(Path(ckpt).parent / f"eval_{split}_preds.npz", y=y, genesis=gen, event=ev, src=src,
                        **{f"p_{n}": v for n, v in preds.items()}, **extra)
    out = Path(ckpt).parent / f"eval_{split}.json"
    out.write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != "models"}, indent=1, default=float))
    if "wind" in report:
        print("wind head", {k: (round(v["csi_best"], 3), round(v["persistence_csi"], 3), v["n_pos"]) for k, v in report["wind"].items() if isinstance(v, dict)})
    for name, r in report["models"].items():
        print(name, {k: round(v["csi"], 3) for k, v in r.items() if k.startswith("genesis")})
    return report
