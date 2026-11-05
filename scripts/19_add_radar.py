"""Step 19: add the NEXRAD composite-reflectivity block `xr` to every existing sample, in place (data/radar.py).

    python -u scripts/19_add_radar.py [--workers 4] [--splits train val test] [--limit N] [--dry-run]

Per sample: the IEM N0Q mosaic (cropped to the domain by the IEM WMS, ~0.6 MB per 5-min frame) at t0-20, t0-10 and
t0, reduced to the maximum dBZ per 4-km cell, normalised, stored as `xr` (3, ny, nx) float16 plus `xr_missing`
(number of frames IEM had no image for). Phase 1 fetches every UNIQUE frame time once with a few polite workers
(the cache under data/cache/radar is reused by live mode too); phase 2 rewrites the samples. Resumable.
"""
import argparse
import datetime as dt
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dustahead.keep_awake import keep_awake  # noqa: E402
keep_awake(True)

OFFSETS = (20, 10, 0)
_G = {}


def _init(cfg_path):
    from dustahead import load_config
    from dustahead.grid import Grid
    cfg = load_config(cfg_path)
    _G.update(cfg=cfg, grid=Grid.from_config(cfg))


def _t0_of(path: Path):
    with np.load(path, allow_pickle=False) as z:
        if "xr" in z.files:
            return None
        return pd.Timestamp(z["t0"][()]).to_pydatetime()


def fetch_one(iso):
    from dustahead.data import radar
    try:
        fr = radar.frame_cached(dt.datetime.fromisoformat(iso), _G["grid"], _G["cfg"]["paths"]["cache"])
        return "ok" if fr is not None else "missing"
    except Exception as e:  # noqa: BLE001
        return f"{iso}: {type(e).__name__}: {str(e)[:100]}"


def augment_one(path_str):
    from dustahead.data import radar
    from dustahead.data.sampler import assemble_radar
    path = Path(path_str)
    try:
        with np.load(path, allow_pickle=False) as z:
            if "xr" in z.files:
                return path.name, "skip"
            s = {k: z[k] for k in z.files}
        t0 = pd.Timestamp(s["t0"][()]).to_pydatetime()
        st = radar.radar_stack(t0, _G["grid"], _G["cfg"]["paths"]["cache"], OFFSETS)
        s["xr"] = assemble_radar(st).astype(np.float16)
        s["xr_missing"] = np.array(int(np.isnan(st).all(axis=(1, 2)).sum()), np.int8)
        tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, **s)
        os.replace(tmp, path)
        return path.name, int(s["xr_missing"])
    except Exception as e:  # noqa: BLE001
        return path.name, f"{type(e).__name__}: {str(e)[:160]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=4, help="concurrent IEM requests in phase 1 / processes in phase 2")
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    from dustahead import load_config
    cfg = load_config(a.config)
    root = Path(cfg["paths"]["samples"])
    todo = []
    for split in a.splits:
        for f in sorted((root / split).glob("*.npz")):
            t0 = _t0_of(f)
            if t0 is not None:
                todo.append((f, t0))
    if a.limit:
        todo = todo[: a.limit]
    times = set()
    for _, t0 in todo:
        for off in OFFSETS:
            t = t0 - dt.timedelta(minutes=off)
            times.add((t - dt.timedelta(minutes=t.minute % 5)).replace(second=0, microsecond=0).isoformat())
    print(f"{len(todo)} samples need xr; {len(times)} unique 5-min frames to fetch (~{0.6 * len(times) / 1024:.1f} GB); workers {a.workers}", flush=True)
    if a.dry_run or not todo:
        return
    _init(a.config)
    t = time.time()
    n_ok = n_miss = 0
    bad = []
    with ThreadPoolExecutor(a.workers) as ex:              # network-bound: threads are enough and gentler on IEM
        for k, r in enumerate(ex.map(fetch_one, sorted(times))):
            n_ok += r == "ok"; n_miss += r == "missing"
            if r not in ("ok", "missing"):
                bad.append(r)
            if (k + 1) % 1000 == 0:
                print(f"  {k + 1}/{len(times)} frames ({time.time() - t:.0f} s, {n_miss} missing, {len(bad)} errors)", flush=True)
    print(f"phase 1 done in {time.time() - t:.0f} s: {n_ok} frames, {n_miss} missing at IEM, {len(bad)} errors" + ("" if not bad else ":\n  " + "\n  ".join(bad[:10])), flush=True)
    t = time.time()
    n_ok = n_part = n_err = 0
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.config,)) as ex:
        for k, (name, r) in enumerate(ex.map(augment_one, [str(f) for f, _ in todo], chunksize=8)):
            if isinstance(r, int):
                n_ok += 1; n_part += r > 0
            elif r != "skip":
                n_err += 1; print(f"  {name}: {r}", flush=True)
            if (k + 1) % 1000 == 0:
                print(f"  {k + 1}/{len(todo)} rewritten ({time.time() - t:.0f} s)", flush=True)
    print(f"phase 2 done in {time.time() - t:.0f} s: {n_ok} samples augmented, {n_part} with a missing frame, {n_err} errors", flush=True)


if __name__ == "__main__":
    main()
