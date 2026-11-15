"""Step 21: sanity check of the station-wind keys written by 20_add_wind.py, on a random subset of samples.

    python scripts/21_check_wind.py [--n 600]

Prints how many samples carry the keys, the share of ASOS station rows with a wind report at t0 and at each lead,
and the base rate of the wind event (peak wind >= loss.wind_kt). Exit code 1 if the keys are missing or mostly
empty, so a job chain stops before training on zeros.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from dustahead import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=600)
    a = ap.parse_args()
    cfg = load_config(a.config)
    kt = float(cfg.get("loss", {}).get("wind_kt", 30.0))
    files = sorted(Path(cfg["paths"]["samples"]).glob("*/*.npz"))
    files = [files[k] for k in np.random.default_rng(0).choice(len(files), min(a.n, len(files)), replace=False)]
    have, now_ok, tgt_ok, pos, n_asos, onset = 0, 0, 0, 0, 0, 0
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            if "st_wy" not in z.files:
                continue
            have += 1
            asos = (z["st_src"] == 0) if "st_src" in z.files else np.ones(len(z["st_ij"]), bool)
            wn, wy = z["st_wnow"][asos], z["st_wy"][asos]
        n_asos += len(wn)
        now_ok += int(np.isfinite(wn[:, 0]).sum())
        tgt_ok += int(np.isfinite(wy).sum())
        pos += int((wy >= kt).sum())
        onset += int(((wy >= kt) & ~(np.nan_to_num(wn[:, :1], nan=0.0) >= kt)).sum())
    L = len(cfg["time"]["leads_min"])
    print(f"{have}/{len(files)} samples carry wind; ASOS rows {n_asos}: wind at t0 {now_ok / max(n_asos, 1):.1%}, "
          f"targets {tgt_ok / max(n_asos * L, 1):.1%}; wind >= {kt:.0f} kt in {pos / max(tgt_ok, 1):.2%} of targets "
          f"({pos} positives, {onset} of them onsets) in this subset", flush=True)
    ok = have == len(files) and now_ok > 0.3 * n_asos and pos > 0
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
