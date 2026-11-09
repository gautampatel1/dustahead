"""Step 21: add the sky-branch OUTFLOW target `st_wind` to every existing sample, in place. Nothing is downloaded.

    python -u scripts/21_add_wind_targets.py --dry-run            # count positives per lead, write nothing
    python -u scripts/21_add_wind_targets.py [--workers 6] [--splits train val test]

st_wind (S, L) int8, one row per station in st_ij order: 1 when the ASOS report nearest t0 + lead (±5 min) has wind or
gust >= label.outflow_kt, 0 below, -1 no wind reading (events.stations.station_wind_targets). It trains the sky branch
sigmoid(z) of the gated U-Net on its own (loss.w_sky), so the factor means P(outflow arrives) rather than "anything
the gate does not veto". The wind comes from the per-event ASOS tables 03_build_samples.py wrote (data/labels/events/,
columns sknt and gust). Samples extended by 16_extend_labels.py (they carry st_src) are matched to its ASOS + extended
station order; extended stations have no wind reading and get -1. Resumable: files that already carry st_wind are
skipped (16 and 03 now write it themselves). Inputs and every other target are untouched.
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


def station_order(labels: pd.DataFrame, stations: pd.DataFrame) -> list[str]:
    """Station ids in the row order build_sample / 16_extend_labels used: station_targets groups labels by station
    (sorted ids), then an inner merge with the station table keeps that order."""
    return sorted(set(labels.station.astype(str)) & set(stations.sid.astype(str)))


def add_event(args):
    """(event_id, [sample paths], dry_run) -> (event_id, Counter, [errors])."""
    from dustahead.events.stations import station_wind_targets
    event_id, paths, dry = args
    cfg = _G["cfg"]
    L = Path(cfg["paths"]["labels"])
    leads = cfg["time"]["leads_min"]
    c, errors = Counter(), []
    asos_csv = next((p for p in (L / "events" / f"{event_id}.csv", L / f"labels_{event_id}.csv") if p.is_file()), None)
    if asos_csv is None:
        return event_id, c, [f"{event_id}: no ASOS labels table under {L}"]
    asos = pd.read_csv(asos_csv, low_memory=False)
    if "sknt" not in asos:
        return event_id, c, [f"{event_id}: {asos_csv.name} has no wind column (sknt)"]
    tables = {False: (asos, _G["stations"])}
    ext_csv = L / "events_ext" / f"{event_id}.csv"
    if ext_csv.is_file() and _G["stations_ext"] is not None:
        tables[True] = (pd.read_csv(ext_csv, low_memory=False), _G["stations_ext"])
    order = {ext: (station_order(lab, sta), sta.assign(sid=sta.sid.astype(str)).set_index("sid")[["i", "j"]])
             for ext, (lab, sta) in tables.items()}
    for p in map(Path, paths):
        try:
            with np.load(p, allow_pickle=False) as z:
                if "st_wind" in z.files:
                    c["skipped"] += 1
                    continue
                s = {k: z[k] for k in z.files}
            ext = "st_src" in s
            if ext not in order:
                errors.append(f"{p.name}: extended sample but no {ext_csv.name} / stations_ext.csv")
                continue
            sids, ij_of = order[ext]
            if len(sids) != len(s["st_ij"]) or not np.array_equal(ij_of.loc[sids].values, s["st_ij"].astype(int)):
                errors.append(f"{p.name}: station order does not match st_ij ({len(sids)} ids vs {len(s['st_ij'])} rows)")
                continue
            wind = station_wind_targets(asos, sids, pd.Timestamp(s["t0"][()]), leads, cfg)
            if ext:
                wind[s["st_src"] != 0] = -1
            c["samples"] += 1
            c["known"] += int((wind >= 0).sum())
            for l, lead in enumerate(leads):
                c[f"wind_pos_{lead}"] += int((wind[:, l] == 1).sum())
                c[f"dust_pos_{lead}"] += int((s["st_y"][:, l] == 1).sum())
            if dry:
                continue
            s["st_wind"] = wind
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
    ap.add_argument("--dry-run", action="store_true", help="count, write nothing")
    a = ap.parse_args()
    from dustahead import load_config
    cfg = load_config(a.config)
    root = Path(cfg["paths"]["samples"])
    leads = cfg["time"]["leads_min"]

    for split in a.splits:
        by_event: dict[str, list[str]] = {}
        for f in sorted((root / split).glob("*.npz")):
            by_event.setdefault(f.stem.rsplit("_", 1)[0], []).append(str(f))
        jobs = [(ev, ps, a.dry_run) for ev, ps in sorted(by_event.items())][: a.limit]
        if not jobs:
            continue
        t, total, errors = time.time(), Counter(), []
        with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.config,)) as ex:
            for _, c, err in ex.map(add_event, jobs):
                total.update(c)
                errors += err
        print(f"[{split}] {len(jobs)} events, {total['samples']} samples, {total['skipped']} already had st_wind, "
              f"{total['written']} written, {len(errors)} errors ({time.time() - t:.0f} s); "
              f"{total['known']} station-lead wind readings")
        print("  outflow positives (wind >= %g kt) vs dust positives: " % cfg["label"].get("outflow_kt", 30)
              + ", ".join(f"+{l}: {total[f'wind_pos_{l}']} / {total[f'dust_pos_{l}']}" for l in leads))
        for e in errors[:10]:
            print("   ", e)


if __name__ == "__main__":   # Windows: workers re-import this file
    main()
