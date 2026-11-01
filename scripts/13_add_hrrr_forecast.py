"""Step 13: add the HRRR short-range FORECAST block `xf` to every existing sample, in place. Nothing else is refetched.

    python -u scripts/13_add_hrrr_forecast.py [--workers 6] [--splits train val test] [--limit N] [--dry-run]

For each data/samples/<split>/*.npz without an `xf` key: read its t0, fetch (or take from cache) the HRRR forecast
fields of the run initialised 2 h before the hour of t0 (fxx 3 and 4: see data/hrrr.py forecast_stack), normalise
them (sampler.assemble_forecast) and rewrite the .npz with `xf` (6, ny, nx) float16 and `xf_missing` (number of
NaN fields, 0 = complete). Resumable: files that already carry `xf` are skipped, cached fields are reused.
Phase 1 prefetches every UNIQUE (run, fxx, field) over all samples so each GRIB message is read once (about 24 per
event); phase 2 rewrites the samples. Both phases use worker PROCESSES. The test split is augmented too (it is still
never evaluated until the end); the station labels and every other array are untouched.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dustahead.keep_awake import keep_awake  # noqa: E402
keep_awake(True)

_G = {}


def _init(cfg_path):
    from dustahead import load_config
    from dustahead.data import hrrr
    from dustahead.grid import Grid
    cfg = load_config(cfg_path)
    hrrr.set_cache(cfg["paths"]["cache"])
    _G.update(cfg=cfg, grid=Grid.from_config(cfg))


def _t0_of(path: Path):
    with np.load(path, allow_pickle=False) as z:
        if "xf" in z.files:
            return None
        return pd.Timestamp(z["t0"][()]).to_pydatetime()       # 0-d datetime64 array


def prefetch_one(args):
    """(run_iso, fxx, fields) -> cache those regridded fields of one GRIB file in a single Herbie call."""
    import datetime as dt
    from dustahead.data import hrrr
    run_iso, fxx, fields = args
    try:
        hrrr.fields_on_grid(dt.datetime.fromisoformat(run_iso), tuple(fields), _G["grid"], fxx=fxx)
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"{run_iso} f{fxx:02d} {fields}: {type(e).__name__}: {str(e)[:100]}"


def augment_one(path_str):
    """Rewrite one sample with the forecast block. Returns (path, n_missing) or (path, error string)."""
    from dustahead.data import hrrr
    from dustahead.data.sampler import assemble_forecast
    path = Path(path_str)
    try:
        with np.load(path, allow_pickle=False) as z:
            if "xf" in z.files:
                return path.name, "skip"
            s = {k: z[k] for k in z.files}
        t0 = pd.Timestamp(s["t0"][()]).to_pydatetime()
        fc = hrrr.forecast_stack(t0, _G["grid"])
        n_missing = int(sum(np.isnan(v).all() for v in fc.values()))
        s["xf"] = assemble_forecast(fc).astype(np.float16)
        s["xf_missing"] = np.array(n_missing, np.int8)
        tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, **s)
        os.replace(tmp, path)
        return path.name, n_missing
    except Exception as e:  # noqa: BLE001
        return path.name, f"{type(e).__name__}: {str(e)[:160]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None, help="first N samples that still need xf (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="count the work, fetch nothing")
    a = ap.parse_args()
    from dustahead import load_config
    from dustahead.data import hrrr
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
    runs = set()
    for _, t0 in todo:
        run, fxx = hrrr.forecast_runs(t0)
        runs.add((run.isoformat(), fxx, hrrr.FORECAST_FIELDS))
        runs.add((run.isoformat(), fxx + 1, ("gust",)))
    print(f"{len(todo)} samples need xf; {len(runs)} unique (run, fxx) GRIB files to read; workers {a.workers}", flush=True)
    if a.dry_run or not todo:
        return

    t = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.config,)) as ex:
        bad = [r for r in ex.map(prefetch_one, sorted(runs), chunksize=4) if r != "ok"]
    print(f"phase 1 (prefetch) done in {time.time() - t:.0f} s; {len(bad)} items failed" + ("" if not bad else ":\n  " + "\n  ".join(bad[:20])), flush=True)

    t = time.time()
    n_ok = n_missing = n_err = 0
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.config,)) as ex:
        for k, (name, r) in enumerate(ex.map(augment_one, [str(f) for f, _ in todo], chunksize=8)):
            if isinstance(r, int):
                n_ok += 1
                n_missing += r > 0
            elif r != "skip":
                n_err += 1
                print(f"  {name}: {r}", flush=True)
            if (k + 1) % 500 == 0:
                print(f"  {k + 1}/{len(todo)} rewritten ({time.time() - t:.0f} s)", flush=True)
    print(f"phase 2 done in {time.time() - t:.0f} s: {n_ok} samples augmented, {n_missing} with a missing field, {n_err} errors", flush=True)


if __name__ == "__main__":   # Windows: workers re-import this file
    main()
