"""Step 3: labels + samples for every event in data/labels/events.csv.

    python -u scripts/03_build_samples.py [--limit 2] [--only dust|nodust] [--workers 6] [--no-adp] [--event ID] [--max-samples N]

Per event: fetch 5-min ASOS reports, DS.W polygons and Storm Events, build station labels, then for
each forecast time t0 (every 10 min from 3 h before the first warning to 2 h after the last expiry)
read 6 GOES frames + GLM + HRRR, compute the physics prior, paint targets and write
data/samples/<split>/<event_id>_<t0>.npz. Resumable: existing files and cached frames are skipped.
Workers are PROCESSES (regridding and compression are CPU-bound; threads would serialise on the GIL).
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dustahead.keep_awake import keep_awake  # noqa: E402
keep_awake(True)   # long job: stop Windows from sleeping while this process runs

_G = {}   # per-process globals (config, grid, static, stations)


def _init(cfg_path):
    from dustahead import load_config
    from dustahead.data.static import load_static
    from dustahead.grid import Grid
    cfg = load_config(cfg_path)
    grid = Grid.from_config(cfg)
    _G.update(cfg=cfg, grid=grid, static=load_static(cfg["paths"]["static"], grid),
              stations=pd.read_csv(Path(cfg["paths"]["labels"]) / "stations.csv"))


def prefetch_one(args):
    """('goes'|'glm'|'hrrr'|'adp', iso_time) -> fills the cache for that item. Runs in a worker process.
    Doing this over UNIQUE items first means each satellite frame is downloaded once per event instead of
    once per sample (six samples share five of their six frames)."""
    kind, iso = args
    import datetime as dt
    from dustahead.data import glm, goes, hrrr
    cfg, grid = _G["cfg"], _G["grid"]
    P, S = cfg["paths"], cfg["satellite"]
    t = pd.Timestamp(iso).to_pydatetime()
    try:
        if kind == "goes":
            f = goes.nearest_file(S["bucket"], S["product"], t)
            if f:
                goes.frame_cached(f, grid, Path(P["cache"]) / "goes", goes.BANDS)
        elif kind == "glm":
            glm.flash_density_at(t, grid, cfg)
        elif kind == "hrrr":
            hrrr.set_cache(P["cache"])
            for k in hrrr.ANALYSIS_FIELDS:
                hrrr.field_on_grid(t, k, grid)
        elif kind == "rain":
            hrrr.set_cache(P["cache"])
            hrrr.field_on_grid(t, "apcp6", grid, fxx=6)
        elif kind == "adp":
            f = goes.nearest_file(S["bucket"], S["dust_product"], t)
            if f:
                goes.read_adp_dust(f, grid, cache_dir=Path(P["cache"]) / "adp")
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"{kind} {iso} failed: {type(e).__name__}: {str(e)[:120]}"


def prefetch_jobs(times, cfg):
    """Unique (kind, time) items needed by a list of forecast times."""
    import datetime as dt
    T = cfg["time"]
    goes_t, glm_t, hrrr_t, rain_t, adp_t = set(), set(), set(), set(), set()
    for t0 in times:
        for k in range(T["history_frames"]):
            tk = t0 - dt.timedelta(minutes=T["step_min"] * k)
            goes_t.add(tk); glm_t.add(tk)
        hrrr_t.add(t0.replace(minute=0, second=0, microsecond=0))
        last = t0.replace(minute=0, second=0, microsecond=0)
        last = last - dt.timedelta(hours=last.hour % 6)
        for k in range(28):
            rain_t.add(last - dt.timedelta(hours=6 * (k + 1)))
        for lead in T["leads_min"]:
            adp_t.add(t0 + dt.timedelta(minutes=lead))
    jobs = [("goes", t.isoformat()) for t in sorted(goes_t)] + [("glm", t.isoformat()) for t in sorted(glm_t)]
    jobs += [("hrrr", t.isoformat()) for t in sorted(hrrr_t)] + [("rain", t.isoformat()) for t in sorted(rain_t)]
    jobs += [("adp", t.isoformat()) for t in sorted(adp_t)]
    return jobs


def build_one(args):
    """(event_id, split, t0_iso, labels_csv, use_adp) -> status string. Runs in a worker process."""
    event_id, split, t0_iso, labels_csv, use_adp = args
    from dustahead.data.sampler import build_sample, save_sample
    cfg, grid = _G["cfg"], _G["grid"]
    t0 = pd.Timestamp(t0_iso).to_pydatetime()
    out = Path(cfg["paths"]["samples"]) / split / f"{event_id}_{t0:%Y%m%dT%H%M}.npz"
    if out.exists():
        return "skip"
    labels = pd.read_csv(labels_csv, parse_dates=["valid"])
    t = time.time()
    try:
        s = build_sample(t0, grid, cfg, _G["static"], labels, _G["stations"], use_adp=use_adp)
    except Exception as e:  # noqa: BLE001
        return f"{t0:%Y-%m-%d %H:%M} failed: {type(e).__name__}: {str(e)[:160]}"
    if s is None:
        return f"{t0:%Y-%m-%d %H:%M} no satellite data"
    save_sample(out, s)
    return f"ok {time.time() - t:.0f}s"


def main():
    from dustahead import load_config
    from dustahead.data.sampler import sample_times
    from dustahead.events.stations import fetch_asos, fetch_station_table, label_reports
    from dustahead.events.storm_events import fetch_dust_storm_events
    from dustahead.events.warnings import fetch_polygon
    from dustahead.grid import Grid

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="max events (for a first test)")
    ap.add_argument("--only", choices=["dust", "nodust"], default=None)
    ap.add_argument("--workers", type=int, default=6, help="parallel worker processes")
    ap.add_argument("--no-adp", action="store_true")
    ap.add_argument("--event", default=None, help="build only this event_id")
    ap.add_argument("--max-samples", type=int, default=None, help="per event, for quick tests")
    ap.add_argument("--null-ratio", type=float, default=1.0, help="null days per dust event (per year), sampled with seed 0")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    grid = Grid.from_config(cfg)
    L = Path(cfg["paths"]["labels"])
    events = pd.read_csv(L / "events.csv", parse_dates=["t_start", "t_end"])
    warnings = pd.read_csv(L / "warnings.csv", parse_dates=["issue", "expire"])
    if a.event:
        events = events[events.event_id == a.event]
    if a.only:
        events = events[events.kind == a.only]
    keep = []
    for yr, grp in events.groupby("year"):
        n_dust = int((grp.kind == "dust").sum())
        nulls = grp[grp.kind == "nodust"].sample(min(int(a.null_ratio * n_dust), int((grp.kind == "nodust").sum())), random_state=0) if n_dust else grp.iloc[0:0]
        keep.append(pd.concat([grp[grp.kind == "dust"], nulls]))
    events = pd.concat(keep).sort_values("t_start") if keep else events
    if a.limit:
        events = events.head(a.limit)
    print(f"building {int((events.kind == 'dust').sum())} dust events + {int((events.kind == 'nodust').sum())} null days "
          f"with {a.workers} worker processes", flush=True)

    if not (L / "stations.csv").exists():
        fetch_station_table(grid).to_csv(L / "stations.csv", index=False)
    stations = pd.read_csv(L / "stations.csv")
    split_of = {y: s for s, ys in cfg["splits"].items() for y in ys}
    storm_events = {}
    t_all = time.time()
    n_done = 0

    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.config,)) as ex:
        for ev in events.itertuples():
            split = split_of.get(ev.year)
            if split is None:
                continue
            labels_csv = L / "events" / f"{ev.event_id}.csv"
            labels_csv.parent.mkdir(parents=True, exist_ok=True)
            old = L / f"labels_{ev.event_id}.csv"
            if old.is_file() and not labels_csv.exists():
                labels_csv.write_bytes(old.read_bytes())          # reuse labels from the first build
            if not labels_csv.exists():
                t_lo = ev.t_start - pd.Timedelta(hours=cfg["time"]["hours_before"] + 1)
                t_hi = ev.t_end + pd.Timedelta(hours=cfg["time"]["hours_after"] + 1)
                obs = fetch_asos(list(stations.sid), t_lo, t_hi, tuple(cfg["label"]["asos_report_types"]))
                ds_w = warnings[(warnings.phenomena == "DS") & (warnings.expire >= t_lo) & (warnings.issue <= t_hi)]
                polys = {(w.wfo, w.year, w.eventid): fetch_polygon(w.wfo, w.year, w.eventid) for w in ds_w.itertuples()}
                if ev.year not in storm_events:
                    try:
                        storm_events[ev.year] = fetch_dust_storm_events(ev.year, cache_dir=cfg["paths"]["raw"] + "/storm_events")
                    except Exception as e:  # noqa: BLE001
                        print(f"[storm events {ev.year}] unavailable: {e}", flush=True)
                        storm_events[ev.year] = None
                labels = label_reports(obs, ds_w, stations, cfg, polys, storm_events[ev.year])
                labels.to_csv(labels_csv, index=False)
            else:
                labels = pd.read_csv(labels_csv)
            print(f"[{ev.event_id}] {ev.kind}: {len(labels)} reports from {labels.station.nunique()} stations, "
                  f"{int(labels.label.sum())} dust-positive, {int(labels.severe.sum())} severe", flush=True)
            times = sample_times(ev, cfg)[: a.max_samples]
            todo = [t0 for t0 in times if not (Path(cfg["paths"]["samples"]) / split / f"{ev.event_id}_{t0:%Y%m%dT%H%M}.npz").exists()]
            t_ev = time.time()
            if todo:
                pre = prefetch_jobs(todo, cfg)
                if a.no_adp:
                    pre = [j for j in pre if j[0] != "adp"]
                pre_res = list(ex.map(prefetch_one, pre, chunksize=2))
                n_bad = sum(not r.startswith("ok") for r in pre_res)
                print(f"  prefetched {len(pre)} items ({sum(j[0] == 'goes' for j in pre)} frames) in {time.time() - t_ev:.0f}s"
                      + (f", {n_bad} failed: {[r for r in pre_res if not r.startswith('ok')][:2]}" if n_bad else ""), flush=True)
            jobs = [(ev.event_id, split, t0.isoformat(), str(labels_csv), not a.no_adp) for t0 in times]
            results = list(ex.map(build_one, jobs))
            ok = sum(r.startswith("ok") for r in results)
            n_done += ok
            bad = [r for r in results if not (r.startswith("ok") or r == "skip")]
            print(f"  {ok} written, {results.count('skip')} existed, {len(bad)} failed in {time.time() - t_ev:.0f}s "
                  f"({(time.time() - t_ev) / max(ok, 1):.1f} s/sample); total {n_done} in {(time.time() - t_all) / 60:.0f} min", flush=True)
            for r in bad[:5]:
                print("   ", r, flush=True)


if __name__ == "__main__":
    main()
