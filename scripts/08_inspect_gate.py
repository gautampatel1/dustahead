"""Inspect the gated model on stored samples: how open is the ground gate, what does the U-Net branch say,
and what do the physics prior / soil moisture look like? Used to understand the monsoon-haboob misses.

    python scripts/08_inspect_gate.py --ckpt runs/gated_unet_v4/best.pt --split val --event dust_20250825_2236 [--every 6]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dustahead import load_config
from dustahead.data.sampler import CH, load_sample, paint_station_now
from dustahead.train.train import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/gated_unet_v4/best.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--event", action="append", required=True)
    ap.add_argument("--every", type=int, default=6)
    a = ap.parse_args()
    cfg = load_config()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, ck = load_checkpoint(a.ckpt, dev)
    soil_ch = CH["hrrr"].start + 6          # u, v, gust, cape, dcape, rh, soilm, rain7d
    rain_ch = CH["hrrr"].start + 7
    phys_ch = CH["physics"].start
    print("event/time               gate: mean  p90  max | sig(z) max 15/30/60 | prior mean max | soilm mean | rain7d mean | P max 15/30/60 | st dust")
    for ev in a.event:
        files = sorted(Path(cfg["paths"]["samples"], a.split).glob(f"{ev}_*.npz"))
        for f in files[::a.every]:
            s = load_sample(f)
            x = s["x"].astype(np.float32)
            x = np.concatenate([x, paint_station_now(s["st_ij"], s["st_now"], x.shape[1:])[None]], 0)
            with torch.no_grad():
                p, z, h = m(torch.from_numpy(x)[None].to(dev), return_parts=True)
            g = torch.sigmoid(h)[0, 0].float().cpu().numpy()
            zz = torch.sigmoid(z)[0].float().cpu().numpy().reshape(3, -1).max(1)
            pp = p[0].float().cpu().numpy().reshape(3, -1).max(1)
            print(f"{f.stem[-18:]:22s}  {g.mean():.3f} {np.percentile(g, 90):.3f} {g.max():.3f} | "
                  f"{zz[0]:.2f} {zz[1]:.2f} {zz[2]:.2f} | {x[phys_ch].mean():.3f} {x[phys_ch].max():.2f} | "
                  f"{x[soil_ch].mean():.3f} | {x[rain_ch].mean():.3f} | {pp[0]:.2f} {pp[1]:.2f} {pp[2]:.2f} | {int((s['st_now'] == 1).sum())}")


if __name__ == "__main__":
    main()
