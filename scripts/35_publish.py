"""Publish forecasts for the web and phone apps: predict -> AlertTracker -> bundle -> (R2 upload) -> heartbeat.

    python scripts/35_publish.py --live --once [--upload]          # one forecast for now (Task Scheduler, every 10 min)
    python scripts/35_publish.py --live --loop [--upload]          # long-running worker, wakes at :02, :12, :22 ...
    python scripts/35_publish.py --replay data/samples/val --event <event> [--out web/public/replay/<event>]
    python scripts/35_publish.py --fixture                         # illustrative storm, no model or data needed

Terms: ASOS = Automated Surface Observing System; NWS = National Weather Service; R2 = Cloudflare's
S3-compatible object store; t0 = forecast time (rounded down to 10 min).

--live      runs the product ensemble (configs/default.yaml product.checkpoints, or --ckpt) on live GOES/GLM/HRRR/ASOS.
            The AlertTracker state is saved in <out>/tracker.json between runs, so hysteresis (30-min minimum hold,
            release below 0.50) also holds when Task Scheduler starts a fresh process every 10 minutes. A state older
            than three forecast steps is discarded: after an outage the alerts restart from the thresholds.
--replay    replays the saved samples of one event through the same ensemble and policy into runs/<t0>/ folders plus
            replay.json, which the web app steps through with ?replay=<folder>/replay.json. Station observations
            come from the stored reports; NWS products are not part of a sample, so replays carry nws = null.
--fixture   writes web/public/demo/: an analytic storm (the blueprint's example: outflow from storms over Tucson
            and Marana carrying dust north-west up I-10, a weaker area near Lordsburg) run through the real road
            segments, thresholds, genesis rule and AlertTracker. It is marked source = "fixture" and the apps label
            it as an illustration. It is for building the interface on a laptop, never for any result.
Secrets come from the environment only: R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET for
--upload, and DUSTAHEAD_HC_URL (a Healthchecks.io ping URL) for the dead-man's switch.
"""
from __future__ import annotations

import argparse
import datetime as dt
import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dustahead import ROOT, load_config  # noqa: E402
from dustahead.app import bundle as B  # noqa: E402
from dustahead.app.predict import AlertTracker, DustAhead  # noqa: E402

STEP = dt.timedelta(minutes=10)


# ---------------------------------------------------------------------------------------------- helpers
def heartbeat(ok: bool, msg: str = "") -> None:
    url = os.environ.get("DUSTAHEAD_HC_URL")
    if not url:
        return
    try:
        import requests
        requests.post(url + ("" if ok else "/fail"), data=msg[-9000:].encode(), timeout=10)
    except Exception as e:  # noqa: BLE001
        print(f"[publish] heartbeat failed: {type(e).__name__}")


def load_tracker(cfg: dict, path: Path, t0: dt.datetime) -> AlertTracker:
    tr = AlertTracker(cfg)
    try:
        saved = json.loads(path.read_text())
        last = dt.datetime.fromisoformat(saved["t0"])
        if dt.timedelta(0) < t0 - last <= 3 * STEP:
            tr.state = saved["state"]
        else:
            print(f"[publish] tracker state from {saved['t0']} is not the previous step: starting fresh")
    except FileNotFoundError:
        pass
    except Exception as e:  # noqa: BLE001  - a corrupt state file must not stop the forecast
        print(f"[publish] could not read {path.name} ({type(e).__name__}): starting fresh")
    return tr


def save_tracker(tr: AlertTracker, path: Path, t0: dt.datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(t0=t0.isoformat(), state=tr.state)))


def station_names(cfg: dict, engine=None) -> dict:
    """(i, j) -> ASOS id from the live station table or labels/stations.csv; {} when neither exists."""
    import pandas as pd
    st = getattr(engine, "stations", None)
    if st is None:
        f = ROOT / cfg["paths"]["labels"] / "stations.csv"
        st = pd.read_csv(f) if f.exists() else None
    if st is None or not {"sid", "i", "j"} <= set(st.columns):
        return {}
    return {(int(i), int(j)): str(s) for s, i, j in zip(st.sid, st.i, st.j)}


def engine_from(args, cfg) -> DustAhead:
    ckpts = args.ckpt or cfg.get("product", {}).get("checkpoints", ["runs/latest/best.pt"])
    return DustAhead(ckpts, cfg)


# ---------------------------------------------------------------------------------------------- modes
def publish_live(args, cfg, out: Path) -> None:
    from dustahead.app import nws
    E = engine_from(args, cfg)

    def once():
        t_start = time.time()
        fc = E.predict()
        t0 = fc["t0"].replace(tzinfo=None) if fc["t0"].tzinfo else fc["t0"]
        tr = load_tracker(cfg, out / "tracker.json", t0)
        fc["alerts"] = tr.update(fc["alerts"], fc["probs"], E)
        warn = nws.fetch_active()
        _, files = B.write_bundle(fc, E, out, source="live", nws=warn, station_names=station_names(cfg, E))
        save_tracker(tr, out / "tracker.json", t0)
        n_up = B.upload_r2(files, out) if args.upload else 0
        msg = f"t0 {t0:%Y-%m-%d %H:%M}Z  alerts {len(fc['alerts'])}  uploaded {n_up}  {time.time() - t_start:.0f}s"
        print("[publish]", msg)
        heartbeat(True, msg)

    while True:
        try:
            once()
        except Exception:  # noqa: BLE001  - report and keep the loop alive; the apps go "No forecast" after 25 min
            tb = traceback.format_exc()
            print(tb)
            heartbeat(False, tb)
            if args.once:
                sys.exit(1)
        if args.once:
            return
        now = dt.datetime.now(dt.timezone.utc)                 # next run 2 min past the next 10-min mark (GOES latency)
        nxt = now.replace(minute=now.minute // 10 * 10, second=0, microsecond=0) + STEP + dt.timedelta(minutes=2)
        time.sleep(max(5.0, (nxt - now).total_seconds()))


def publish_replay(args, cfg, out: Path) -> None:
    from dustahead.data.sampler import load_sample, paint_station_now
    files = sorted(Path(args.replay).glob("*.npz"))
    events = sorted({f.stem.rsplit("_", 1)[0] for f in files})
    if args.event not in events:
        sys.exit(f"event {args.event!r} not in {args.replay}; have e.g. {events[-5:]}")
    ev_files = [f for f in files if f.stem.rsplit("_", 1)[0] == args.event]
    E = engine_from(args, cfg)
    accepts = set(inspect.signature(E.predict_array).parameters)
    names = station_names(cfg, None)
    tr = AlertTracker(cfg)
    bundles = []
    for f in ev_files:
        s = load_sample(f)
        kw = {k: s[k] for k in ("xf", "xr") if k in s and k in accepts}
        x = s["x"].astype(np.float32)
        fc = E.predict_array(x, paint_station_now(s["st_ij"], s["st_now"], x.shape[1:])[None], s["st_ij"], s["st_now"], **kw)
        fc.update(t0=s["t0"], station_ij=s["st_ij"], station_now=s["st_now"])
        fc["alerts"] = tr.update(fc["alerts"], fc["probs"], E)
        t0 = np.asarray(s["t0"])[()].astype("datetime64[s]").astype(dt.datetime)
        bpath, _ = B.write_bundle(fc, E, out, source="replay", issued=t0 + dt.timedelta(minutes=2), nws=None,
                                  station_names=names, update_latest=False)
        bundles.append(bpath)
        print(f"[replay] {t0:%Y-%m-%d %H:%M}Z  alerts {len(fc['alerts'])}")
    idx = B.write_replay_index(out, bundles, title=f"Replay of {args.event}", source="replay")
    print(f"[replay] {len(bundles)} forecasts -> {idx}")


# The blueprint's example storm, in its map units (60 px per degree from lon -115, lat 36.5) and minutes after t0.
def fixture_p(lon: np.ndarray, lat: np.ndarray, t: float) -> np.ndarray:
    x, y = (lon + 115) * 60, (36.5 - lat) * 60
    f = t / 60
    cx, cy, s = 240 - 44 * f, 250 - 44 * f, 18 + 16 * f
    p = 0.98 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))
    if t > 25:
        g = min(1.0, (t - 25) / 35)
        s2 = 6 + 10 * g
        p = np.maximum(p, 0.9 * g * np.exp(-((x - 380) ** 2 + (y - 250) ** 2) / (2 * s2 * s2)))
    return p.astype(np.float32)


FIXTURE_STATIONS = {"TUS": (-110.94, 32.13), "MZJ": (-111.33, 32.51), "CGZ": (-111.77, 32.95),
                    "PHX": (-112.01, 33.43), "LSB": (-108.69, 32.33), "DMN": (-107.72, 32.26), "ELP": (-106.38, 31.81)}


def publish_fixture(args, cfg, out: Path) -> None:
    E = DustAhead.policy_only(cfg)
    g = E.grid
    lon, lat = g.mesh()
    sids = list(FIXTURE_STATIONS)
    si, sj = g.to_ij([FIXTURE_STATIONS[s][0] for s in sids], [FIXTURE_STATIONS[s][1] for s in sids])
    st_ij = np.stack([si, sj], 1)
    names = {(int(i), int(j)): s for s, i, j in zip(sids, si, sj)}
    t_start = dt.datetime(2026, 8, 25, 23, 10)                          # 4:10 PM MST, as in the blueprint
    tr = AlertTracker(cfg)
    bundles = []
    for k in range(args.frames):
        t0 = t_start + k * STEP
        tm = 10.0 * k
        # Stations see what is at the station NOW (t = tm); TUS and MZJ start the storm, as in the blueprint.
        st_now = np.array([1 if fixture_p(np.array(FIXTURE_STATIONS[s][0]), np.array(FIXTURE_STATIONS[s][1]), tm) >= 0.5
                           else -1 for s in sids], np.int8)
        probs = np.stack([fixture_p(lon, lat, tm + lead) for lead in E.leads])
        gmap = E.genesis_map(st_ij, st_now, probs.shape[1:])
        fc = dict(probs=probs, genesis=gmap, t0=t0, station_ij=st_ij, station_now=st_now,
                  alerts=E.road_alerts(probs, genesis_map=gmap))
        fc["alerts"] = tr.update(fc["alerts"], probs, E)
        nws = [dict(type="Dust Storm Warning", vtec="fixture", area="Pinal County",
                    onset=B._iso(t_start), expires=B._iso(t_start + dt.timedelta(minutes=50)),
                    polygon=[[-111.9, 32.5], [-111.0, 32.5], [-111.0, 33.2], [-111.9, 33.2], [-111.9, 32.5]])]
        nws = nws if t0 < t_start + dt.timedelta(minutes=50) else []
        bpath, _ = B.write_bundle(fc, E, out, source="fixture", issued=t0 + dt.timedelta(minutes=2), nws=nws,
                                  station_names=names, update_latest=(k == 0))
        bundles.append(bpath)
        print(f"[fixture] {t0:%H:%M}Z  alerts {[(a['segment'], a['level'], a['lead_min']) for a in fc['alerts']]}")
    B.write_replay_index(out, bundles, title="Illustration: outflow dust moving up I-10 from Tucson", source="fixture")
    print(f"[fixture] {len(bundles)} forecasts -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--replay", metavar="SAMPLES_DIR")
    mode.add_argument("--fixture", action="store_true")
    ap.add_argument("--event", help="event name for --replay (sample file stem without the _HHMM suffix)")
    ap.add_argument("--once", action="store_true", help="--live: one forecast, then exit")
    ap.add_argument("--loop", action="store_true", help="--live: run every 10 minutes")
    ap.add_argument("--ckpt", action="append", help="checkpoint(s); default = product.checkpoints")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="publish root (default: publish.out_dir, or web/public/demo for --fixture)")
    ap.add_argument("--upload", action="store_true", help="upload to R2 (credentials from the environment)")
    ap.add_argument("--frames", type=int, default=9, help="--fixture: number of 10-min forecasts")
    args = ap.parse_args()
    cfg = load_config(args.config)
    pub = cfg.get("publish", {})
    if args.fixture:
        out = Path(args.out or ROOT / "web" / "public" / "demo")
        publish_fixture(args, cfg, out)
    elif args.replay:
        if not args.event:
            ap.error("--replay needs --event")
        out = Path(args.out or ROOT / pub.get("out_dir", "runs/publish") / "replay" / args.event)
        publish_replay(args, cfg, out)
    else:
        if args.once == args.loop:
            ap.error("--live needs exactly one of --once / --loop")
        out = Path(args.out or ROOT / pub.get("out_dir", "runs/publish"))
        publish_live(args, cfg, out)


if __name__ == "__main__":
    main()
