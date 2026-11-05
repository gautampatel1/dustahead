"""Step 20: rebuild the station-observation channel `st_now` of every existing sample from reports available at t0.

    python -u scripts/20_causal_station_now.py --dry-run          # measure only: how many inputs change, nothing written
    python -u scripts/20_causal_station_now.py [--workers 6] [--splits train val test]
    python scripts/06_index_samples.py                            # afterwards: the genesis column changes

Why: samples built before this fix stored the station LABEL at t0 as the input `st_now`. The label needs a Dust Storm
Warning within ±60 min, so a dust report at t0 was painted +1 only if a warning was issued up to an hour LATER (and the
report itself could be up to 5 min after t0). That is future information in the strongest input (hard rule 3), and it
also decided which stations count as genesis. The live product never had it: it reads visibility and present weather
only. This script applies the one shared rule (events.stations.station_now) to the per-event label tables written by
03_build_samples.py, so nothing is downloaded. Samples already extended by 16_extend_labels.py (they carry st_src) are
matched to its ASOS + extended station order; their extended stations have no weather codes and become "no report" (-1)
in the input, as in the live product (16 itself now writes the causal channel, so new extensions need no fix).
Each rewritten sample gets:
  st_now         causal channel (1 dust, 0 clear, -1 no report)           genesis        recomputed from it
  st_now_label   the old label-based values, kept for analysis            st_sid         station ids in st_ij order
  st_now_causal  True (files that already carry it are skipped: resumable)
Targets (y, w, st_y) and every other array are untouched. GOES frame timing and the HRRR analysis hour are NOT fixed
here - those need 03_build_samples.py (they change the inputs themselves, not just the station channel).
"""
import argparse
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_G = {}


def _init(cfg_path):
    from dustahead import load_config
    cfg = load_config(cfg_path)
    L = Path(cfg["paths"]["labels"])
    _G.update(cfg=cfg, stations=pd.read_csv(L / "stations.csv"),
              stations_ext=pd.read_csv(L / "stations_ext.csv") if (L / "stations_ext.csv").is_file() else None)


def _labels_csv(labels_dir: Path, event_id: str) -> Path | None:
    for p in (labels_dir / "events" / f"{event_id}.csv", labels_dir / f"labels_{event_id}.csv"):
        if p.is_file():
            return p
    return None


def station_order(labels: pd.DataFrame, stations: pd.DataFrame) -> list[str]:
    """Station ids in the row order build_sample used: station_targets groups labels by station (sorted ids),
    then an inner merge with the station table keeps that order."""
    return sorted(set(labels.station.astype(str)) & set(stations.sid.astype(str)))


def fix_event(args):
    """(event_id, [sample paths], dry_run) -> (event_id, Counter of changes, [errors])."""
    from dustahead.data.sampler import genesis_per_station
    from dustahead.events.stations import station_now
    event_id, paths, dry = args
    cfg, stations = _G["cfg"], _G["stations"]
    c, errors = Counter(), []
    L = Path(cfg["paths"]["labels"])
    f = _labels_csv(L, event_id)
    if f is None:
        return event_id, c, [f"{event_id}: no labels table under {L}"]
    labels = pd.read_csv(f, low_memory=False)          # raw ASOS reports: the only source of the input channel
    # station order of a sample: ASOS-only (03_build_samples) or ASOS + extended (16_extend_labels, has st_src)
    tables = {False: (labels, stations)}
    ext_csv = L / "events_ext" / f"{event_id}.csv"
    if ext_csv.is_file() and _G.get("stations_ext") is not None:
        tables[True] = (pd.read_csv(ext_csv, low_memory=False), _G["stations_ext"])
    order = {}
    for ext, (lab, sta) in tables.items():
        order[ext] = (station_order(lab, sta), sta.assign(sid=sta.sid.astype(str)).set_index("sid")[["i", "j"]])
    for p in map(Path, paths):
        try:
            with np.load(p, allow_pickle=False) as z:
                if "st_now_causal" in z.files:
                    c["skipped"] += 1
                    continue
                s = {k: z[k] for k in z.files}
            ext = "st_src" in s
            if ext not in order:
                errors.append(f"{p.name}: extended sample but no {ext_csv} / stations_ext.csv")
                continue
            sids, ij_of = order[ext]
            if len(sids) != len(s["st_ij"]) or not np.array_equal(ij_of.loc[sids].values, s["st_ij"].astype(int)):
                errors.append(f"{p.name}: station order does not match st_ij ({len(sids)} ids vs {len(s['st_ij'])} rows)")
                continue
            old = s["st_now"].astype(np.int8)
            new = station_now(labels, sids, pd.Timestamp(s["t0"][()]), cfg)
            if ext:
                new[s["st_src"] != 0] = -1          # extended stations carry no weather codes: targets only, as live
            gen_old, gen_new = genesis_per_station(s["st_ij"], old), genesis_per_station(s["st_ij"], new)
            pos30 = s["st_y"][:, 1] == 1                         # +30 min station positives (the primary lead)
            c["samples"] += 1
            c["changed"] += int((old != new).any())
            c["dust_old"] += int((old == 1).sum())
            c["dust_new"] += int((new == 1).sum())
            c["dust_lost"] += int(((old == 1) & (new != 1)).sum())          # warning-only "dust now" that vanishes
            c["dust_gained"] += int(((old != 1) & (new == 1)).sum())        # dust reported, warning not (yet) issued
            c["sample_genesis_old"] += int(not (old == 1).any())
            c["sample_genesis_new"] += int(not (new == 1).any())
            c["pos30"] += int(pos30.sum())
            c["pos30_genesis_old"] += int((pos30 & gen_old).sum())
            c["pos30_genesis_new"] += int((pos30 & gen_new).sum())
            if dry:
                continue
            s.update(st_now=new, genesis=np.array(not (new == 1).any()), st_now_label=old,
                     st_sid=np.array(sids, dtype="U8"), st_now_causal=np.array(True))
            tmp = p.with_suffix(f".{os.getpid()}.tmp.npz")
            np.savez_compressed(tmp, **s)
            os.replace(tmp, p)
            c["written"] += 1
        except Exception as e:  # noqa: BLE001
            errors.append(f"{p.name}: {type(e).__name__}: {str(e)[:160]}")
    return event_id, c, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None, help="first N events (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="measure the change, write nothing")
    a = ap.parse_args()
    from dustahead import load_config
    cfg = load_config(a.config)
    root = Path(cfg["paths"]["samples"])

    for split in a.splits:
        by_event: dict[str, list[str]] = {}
        for f in sorted((root / split).glob("*.npz")):
            by_event.setdefault(f.stem.rsplit("_", 1)[0], []).append(str(f))
        jobs = [(ev, ps, a.dry_run) for ev, ps in sorted(by_event.items())][: a.limit]
        if not jobs:
            continue
        t, total, errors = time.time(), Counter(), []
        with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.config,)) as ex:
            for ev, c, err in ex.map(fix_event, jobs):
                total.update(c)
                errors += err
        n = max(total["samples"], 1)
        print(f"[{split}] {len(jobs)} events, {total['samples']} samples checked, {total['skipped']} already causal, "
              f"{total['written']} written, {len(errors)} errors ({time.time() - t:.0f} s)")
        print(f"  samples whose station channel changes: {total['changed']} ({100 * total['changed'] / n:.1f} %)")
        print(f"  'dust now' station-times: label-based {total['dust_old']} -> causal {total['dust_new']} "
              f"(lost {total['dust_lost']}, gained {total['dust_gained']})")
        print(f"  genesis samples (no dust anywhere at t0): {total['sample_genesis_old']} -> {total['sample_genesis_new']}")
        print(f"  +30 min station positives at genesis stations: {total['pos30_genesis_old']} -> {total['pos30_genesis_new']} "
              f"of {total['pos30']}")
        for e in errors[:10]:
            print("   ", e)


if __name__ == "__main__":   # Windows: workers re-import this file
    main()
