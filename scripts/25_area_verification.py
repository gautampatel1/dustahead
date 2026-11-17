"""Step 25: AREA verification - is the model right about the neighbourhood when it is wrong about the point?

    python scripts/25_area_verification.py --runs runs/gated_unet_v8 runs/gated_unet_v8_fold1 \
        --configs default configs/fold1.yaml --out results/area_v8_2fold.json

The benchmark asks for dust at one station cell. A dust wall is tens of kilometres wide but patchy, and the 17 Nov
lead-time study found that 55 % of ASOS stations (Automated Surface Observing System) inside an official Dust Storm
Warning polygon never report strict dust: the forecasters warn an area, and so does a road alert. This script
re-scores stored predictions with a neighbourhood rule, for radius R in --radii-km:
  truth    = strict dust at ANY station within R of the station at t0 + lead
  forecast = the highest station probability within R (same rule for station persistence)
R = 0 is the point benchmark. Onset rows are the usual genesis rows (no dust within 50 km at t0), so persistence
still scores zero there for R <= 50 km. CSI (Critical Success Index) at the best threshold, with POD / FAR, an
event-bootstrap interval for the onset CSI, and the false alarm ratio at fixed thresholds.
Only station points are used (the stored predictions hold no maps), so "area" means "the stations of the area".
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dustahead.eval.metrics import station_scores  # noqa: E402


def load(run: str, cfg: dict, cell_km: tuple, radii_km):
    """Neighbourhood-max probabilities and neighbourhood-any labels per radius, sample by sample."""
    from dustahead.train.dataset import split_files
    files, _ = split_files(cfg["paths"]["samples"], "val", cfg)
    z = np.load(Path(run) / "eval_val_preds.npz", allow_pickle=True)
    y, gen, ev = z["y"], z["genesis"], z["event"]
    src = z["src"] if "src" in z.files else np.zeros(len(y), int)
    names = [k[2:] for k in z.files if k in ("p_model", "p_station_persistence")]
    P = {n: z[f"p_{n}"] for n in names}
    out = {r: dict(y=np.full(y.shape, -1, np.int8), **{n: np.zeros(y.shape, np.float32) for n in names}) for r in radii_km}
    k0 = 0
    for f in files:
        with np.load(f, allow_pickle=False) as s:
            ij = s["st_ij"].astype(np.float32)
        n = len(ij)
        sl = slice(k0, k0 + n)
        asos = src[sl] == 0
        d = np.hypot((ij[:, None, 0] - ij[None, :, 0]) * cell_km[0], (ij[:, None, 1] - ij[None, :, 1]) * cell_km[1])
        yy = y[sl]
        for r in radii_km:
            nb = (d <= r) & asos[None, :]                     # neighbours: ASOS stations within r (self included)
            valid = (yy >= 0)
            anyv = (nb[:, :, None] & valid[None, :, :]).any(1)
            anyp = (nb[:, :, None] & (yy == 1)[None, :, :]).any(1)
            out[r]["y"][sl] = np.where(anyv, anyp.astype(np.int8), -1)
            for nme in names:
                pp = P[nme][sl]
                out[r][nme][sl] = np.where(nb[:, :, None], pp[None, :, :], 0.0).max(1)
        k0 += n
    assert k0 == len(y), f"{run}: {k0} rebuilt rows against {len(y)} stored (wrong --configs entry?)"
    return out, gen, ev, src == 0


def main():
    from dustahead import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--radii-km", type=float, nargs="+", default=[0, 25, 50])
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    cfgs = [load_config(None if c == "default" else c) for c in a.configs]
    D = cfgs[0]["domain"]                      # cell size in km: rows are latitude, columns longitude
    lat0 = np.deg2rad((D["lat_min"] + D["lat_max"]) / 2)
    cell_km = ((D["lat_max"] - D["lat_min"]) / D["ny"] * 111.2, (D["lon_max"] - D["lon_min"]) / D["nx"] * 111.2 * float(np.cos(lat0)))
    parts = [load(r, c, cell_km, a.radii_km) for r, c in zip(a.runs, cfgs)]
    gen = np.concatenate([p[1] for p in parts]); ev = np.concatenate([p[2] for p in parts]); asos = np.concatenate([p[3] for p in parts])
    events = np.unique(ev)
    idx = {e: np.flatnonzero(ev == e) for e in events}
    rng = np.random.default_rng(0)
    draws = [np.concatenate([idx[e] for e in rng.choice(events, len(events))]) for _ in range(a.B)]
    ths = np.arange(0.05, 0.96, 0.05)
    res = dict(runs=a.runs, n_events=int(len(events)), radii_km=a.radii_km, results={})
    for r in a.radii_km:
        Y = np.concatenate([p[0][r]["y"] for p in parts])
        row = {}
        for l, lead in enumerate((15, 30, 60)):
            for tag, m in (("all", asos & (Y[:, l] >= 0)), ("onset", asos & gen & (Y[:, l] >= 0))):
                e = dict(n=int(m.sum()), n_pos=int((Y[m, l] == 1).sum()))
                for name in ("model", "station_persistence"):
                    if name not in parts[0][0][r]:
                        continue
                    p = np.concatenate([q[0][r][name] for q in parts])
                    sc = [station_scores(p[m, l], Y[m, l], t) for t in ths]
                    b = int(np.nanargmax([s_["csi"] for s_ in sc])) if any(np.isfinite(s_["csi"]) for s_ in sc) else 0
                    e[name] = dict(csi_best=sc[b]["csi"], thr=float(ths[b]), pod=sc[b]["pod"], far=sc[b]["far"])
                    if name == "model":
                        for t in (0.5, 0.8):
                            s_ = station_scores(p[m, l], Y[m, l], t)
                            e[name][f"at_{t}"] = dict(csi=s_["csi"], pod=s_["pod"], far=s_["far"])
                        if tag == "onset":
                            bs = [station_scores(p[dr[m[dr]], l], Y[dr[m[dr]], l], ths[b])["csi"] for dr in draws]
                            bs = np.array(bs, float); bs = bs[np.isfinite(bs)]
                            e[name]["ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
                row[f"{tag}_{lead}"] = e
        res["results"][str(r)] = row
        for k, e in row.items():
            mo, pe = e.get("model", {}), e.get("station_persistence", {})
            print(f"R={r:>4} km {k:9s} n_pos {e['n_pos']:5d} | model CSI {mo.get('csi_best', float('nan')):.3f} POD {mo.get('pod', float('nan')):.2f} "
                  f"FAR {mo.get('far', float('nan')):.2f} ci {mo.get('ci', '')} | persistence CSI {pe.get('csi_best', float('nan')):.3f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
