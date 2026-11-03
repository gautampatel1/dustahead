"""Step 16: EXTEND the ground truth with Synoptic Data stations (road-weather visibility, PM10) and rewrite every
sample's station table and targets in place. Inputs x (satellite, lightning, HRRR, static, forecast block) are untouched.

    python -u scripts/16_extend_labels.py [--workers 6] [--limit N] [--event ID] [--dry-run]

Per event (dust days, null days and the test year alike - a label improvement applies to the benchmark too, and the
test split is still evaluated once at the end):
  1. fetch the Synoptic time series (visibility, PM10, 1-h precip) of the extended stations over the event window
     (cached in data/labels/synoptic/<event>.csv),
  2. apply the extended label rule (events/synoptic.label_extended: sensor threshold AND Dust Storm Warning / Storm
     Events cross-check), write data/labels/events_ext/<event>.csv = ASOS labels + extended labels,
  3. for every sample of the event: recompute the station rows for its t0 (station_targets over ASOS + extended),
     paint the new stations' 12 km discs into y / w where no ASOS disc already is (ASOS keeps precedence),
     and replace st_ij / st_y / st_now / genesis; add st_src (0 asos, 1 rwis, 2 vis_other, 3 pm10) for
     per-source reporting. Resumable: samples that already carry st_src are skipped.
Writes data/labels/stations_ext.csv (the combined table every later step uses).
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

SRC = {"asos": 0, "rwis": 1, "vis_other": 2, "pm10": 3}


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
_G = {}


FORCE = False


def _init(cfg_path, force=False):
    global FORCE
    FORCE = force
    from dustahead import load_config
    from dustahead.grid import Grid
    cfg = load_config(cfg_path)
    L = Path(cfg["paths"]["labels"])
    _G.update(cfg=cfg, grid=Grid.from_config(cfg), stations=pd.read_csv(L / "stations_ext.csv"))


def rewrite_one(args):
    """(sample path, labels_ext csv) -> status. Repaints targets and the station table of one stored sample."""
    from dustahead.events.stations import station_targets
    path, labels_csv = Path(args[0]), args[1]
    cfg, grid, stations = _G["cfg"], _G["grid"], _G["stations"]
    try:
        with np.load(path, allow_pickle=False) as z:
            if "st_src" in z.files and not FORCE:
                return "skip"
            s = {k: z[k] for k in z.files}
        labels = pd.read_csv(labels_csv, parse_dates=["valid"])
        t0 = pd.Timestamp(s["t0"][()]).tz_localize("UTC")
        leads = cfg["time"]["leads_min"]
        st = station_targets(labels, stations, t0, leads)
        st = st.merge(stations[["sid", "lon", "lat", "i", "j", "source"]], left_on="station", right_on="sid")
        y, w = s["y"].astype(np.float32), s["w"].astype(np.float32)
        R = cfg["label"]["station_radius_km"]
        if FORCE and "st_src" in s:
            # samples already extended: drop every extended disc first (back to ADP weight 0.3 or unknown 0),
            # then paint the ASOS discs again so precedence is the same as in a first pass
            w_ext = np.zeros_like(w, bool)
            for li in range(len(leads)):
                for (i, j), src_ in zip(s["st_ij"], s["st_src"]):
                    if src_ != 0:
                        w_ext[li] |= grid.disk_mask(float(grid.lons()[j]), float(grid.lats()[i]), R)
            adp = (w > 0) & (w < 1.0)
            w[w_ext & ~adp] = 0.0
            y[w_ext & ~adp] = 0
            for li, lead in enumerate(leads):
                for r in st[st.source == "asos"].itertuples():
                    v = getattr(r, f"y_{lead}")
                    if v < 0:
                        continue
                    m = grid.disk_mask(r.lon, r.lat, R)
                    y[li][m] = v
                    w[li][m] = 1.0
        asos_disc = np.zeros_like(w, bool)
        for li, lead in enumerate(leads):
            for r in st[st.source == "asos"].itertuples():
                if getattr(r, f"y_{lead}") >= 0:
                    asos_disc[li] |= grid.disk_mask(r.lon, r.lat, R)
        for li, lead in enumerate(leads):
            for r in st[st.source != "asos"].itertuples():
                v = getattr(r, f"y_{lead}")
                if v < 0:
                    continue
                m = grid.disk_mask(r.lon, r.lat, R) & ~asos_disc[li]
                y[li][m] = v
                w[li][m] = 1.0
        s["y"], s["w"] = y.astype(np.uint8), w.astype(np.float16)
        s["st_ij"] = st[["i", "j"]].values.astype(np.int16)
        s["st_y"] = st[[f"y_{l}" for l in leads]].values.astype(np.int8)
        s["st_now"] = st.dust_now.values.astype(np.int8)
        s["st_src"] = st.source.map(SRC).values.astype(np.int8)
        s["genesis"] = np.array(not bool((st.dust_now.values == 1).any()))
        tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, **s)
        os.replace(tmp, path)
        return f"ok {len(st)} stations, {int((st.source != 'asos').sum())} extended"
    except Exception as e:  # noqa: BLE001
        return f"{path.name}: {type(e).__name__}: {str(e)[:160]}"


def main():
    from dustahead import load_config
    from dustahead.events.storm_events import fetch_dust_storm_events
    from dustahead.events.synoptic import extended_station_table, label_extended, station_inventory, timeseries
    from dustahead.events.warnings import fetch_polygon
    from dustahead.grid import Grid

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--event", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-label from the cached Synoptic time series (new thresholds in configs) and rewrite every sample again")
    a = ap.parse_args()
    cfg = load_config(a.config)
    grid = Grid.from_config(cfg)
    L = Path(cfg["paths"]["labels"])
    D = cfg["domain"]

    asos = pd.read_csv(L / "stations.csv")
    inv = station_inventory((D["lon_min"], D["lat_min"], D["lon_max"], D["lat_max"]))
    ext = extended_station_table(inv, asos, grid)
    asos["source"] = "asos"
    stations = pd.concat([asos, ext], ignore_index=True)
    stations.to_csv(L / "stations_ext.csv", index=False)
    print(f"stations: {len(asos)} ASOS + {len(ext)} extended {ext.source.value_counts().to_dict()}", flush=True)

    events = pd.read_csv(L / "events.csv", parse_dates=["t_start", "t_end"])
    warnings = pd.read_csv(L / "warnings.csv", parse_dates=["issue", "expire"])
    if a.event:
        events = events[events.event_id == a.event]
    if a.limit:
        events = events.head(a.limit)
    root = Path(cfg["paths"]["samples"])
    files_by_event = {}
    for f in root.glob("*/*.npz"):
        files_by_event.setdefault(f.stem.rsplit("_", 1)[0], []).append(f)
    events = events[events.event_id.isin(files_by_event)]
    print(f"{len(events)} events with stored samples", flush=True)
    if a.dry_run:
        return

    (L / "synoptic").mkdir(exist_ok=True)
    (L / "events_ext").mkdir(exist_ok=True)
    (L / "polygons").mkdir(exist_ok=True)
    storm_events = {}
    T = cfg["time"]
    t_all = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.config, a.force)) as ex:
        for n_ev, ev in enumerate(events.itertuples()):
            t_ev = time.time()
            out_csv = L / "events_ext" / f"{ev.event_id}.csv"
            if not out_csv.exists() or a.force:
                t_lo = ev.t_start - pd.Timedelta(hours=T["hours_before"] + 1)
                t_hi = ev.t_end + pd.Timedelta(hours=T["hours_after"] + 1)
                raw = L / "synoptic" / f"{ev.event_id}.csv"
                if raw.exists():
                    ts = pd.read_csv(raw, parse_dates=["valid"])
                    ts["valid"] = ts.valid.map(_utc)
                else:
                    try:
                        ts = timeseries(list(ext.sid), _utc(t_lo), _utc(t_hi))
                    except Exception as e:  # noqa: BLE001
                        print(f"[{ev.event_id}] synoptic fetch failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
                        continue
                    ts.to_csv(raw, index=False)
                ds_w = warnings[(warnings.phenomena == "DS") & (warnings.expire >= t_lo) & (warnings.issue <= t_hi)]
                polys = {}
                for w_ in ds_w.itertuples():
                    key = (w_.wfo, w_.year, w_.eventid)
                    pf = L / "polygons" / f"{w_.wfo}_{w_.year}_{w_.eventid}.npz"
                    if pf.exists():
                        with np.load(pf) as z:
                            polys[key] = [z[k] for k in z.files] or None
                    else:
                        rings = fetch_polygon(w_.wfo, w_.year, w_.eventid)
                        polys[key] = rings
                        np.savez(pf, *(rings or []))
                if ev.year not in storm_events:
                    try:
                        storm_events[ev.year] = fetch_dust_storm_events(ev.year, cache_dir=cfg["paths"]["raw"] + "/storm_events")
                    except Exception:  # noqa: BLE001
                        storm_events[ev.year] = None
                lab_ext = label_extended(ts, ext, ds_w, cfg, polys, storm_events[ev.year])
                old = L / "events" / f"{ev.event_id}.csv"
                lab_asos = pd.read_csv(old, parse_dates=["valid"])
                lab_asos["valid"] = lab_asos.valid.map(_utc)
                lab_asos["source"] = "asos"
                lab_asos["tol_min"] = 5
                keep = ["station", "valid", "label", "severe", "source", "tol_min"]
                both = pd.concat([lab_asos[keep], lab_ext[keep]], ignore_index=True)
                both.to_csv(out_csv, index=False)
                print(f"[{ev.event_id}] {ev.kind}: synoptic {len(ts)} obs -> {len(lab_ext)} rows, "
                      f"{int(lab_ext.label.sum())} dust-positive at {lab_ext[lab_ext.label == 1].station.nunique()} stations "
                      f"(ASOS positives {int(lab_asos.label.sum())})", flush=True)
            jobs = [(str(f), str(out_csv)) for f in sorted(files_by_event[ev.event_id])]
            res = list(ex.map(rewrite_one, jobs, chunksize=4))
            ok = sum(r.startswith("ok") for r in res)
            bad = [r for r in res if not (r.startswith("ok") or r == "skip")]
            print(f"  {ok} samples rewritten, {res.count('skip')} skipped, {len(bad)} failed in {time.time() - t_ev:.0f}s; "
                  f"event {n_ev + 1}/{len(events)}, {(time.time() - t_all) / 60:.0f} min", flush=True)
            for r in bad[:3]:
                print("   ", r, flush=True)


if __name__ == "__main__":
    main()
