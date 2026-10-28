"""Live-mode smoke test: fetch GOES / GLM / HRRR / ASOS for a recent time and run the product ensemble end to end.

    python scripts/10_live_smoke.py [--t0 2026-10-28T12:00] [--out results/live_smoke]

Default t0 = now minus 2 h (floored to 10 min) so that the HRRR analysis and the GOES files have landed on AWS.
Writes <out>/live_<t0>.json (timings, alerts, max probabilities, station channel summary) and a 2x3 panel PNG.
Late October has no dust; the test only has to run through every stage and report where time goes.
"""
import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dustahead import load_config
from dustahead.app.predict import AlertTracker, DustAhead, I10
from dustahead.data import goes
from dustahead.data.sampler import paint_station_now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", default=None, help="UTC, e.g. 2026-10-28T12:00")
    ap.add_argument("--out", default="results/live_smoke")
    a = ap.parse_args()
    cfg = load_config()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.t0:
        t0 = dt.datetime.fromisoformat(a.t0).replace(tzinfo=dt.timezone.utc)
    else:
        t0 = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=2)
    t0 = t0.replace(minute=t0.minute // 10 * 10, second=0, microsecond=0)
    rep = dict(t0=t0.isoformat(), stages={})

    def stage(name, fn):
        t = time.perf_counter()
        try:
            r = fn()
            rep["stages"][name] = dict(ok=True, seconds=round(time.perf_counter() - t, 1))
            print(f"[{name}] ok in {rep['stages'][name]['seconds']} s")
            return r
        except Exception as e:  # noqa: BLE001
            rep["stages"][name] = dict(ok=False, seconds=round(time.perf_counter() - t, 1), error=f"{type(e).__name__}: {str(e)[:300]}")
            print(f"[{name}] FAILED: {rep['stages'][name]['error']}")
            return None

    E = stage("load_models", lambda: DustAhead(cfg["product"]["checkpoints"], cfg))
    if E is None:
        sys.exit(1)
    x = stage("build_live_input (GOES + GLM + HRRR + static + prior)", lambda: E.build_live_input(t0))
    st = stage("station_now_live (ASOS)", lambda: E.station_now_live(t0))
    if x is None:
        (out / f"live_{t0:%Y%m%dT%H%M}.json").write_text(json.dumps(rep, indent=1))
        sys.exit(1)
    ij, now = st if st is not None else (None, None)
    chan = paint_station_now(ij, now, x.shape[1:])[None] if ij is not None else None
    r = stage("predict (3 checkpoints, max)", lambda: E.predict_array(x, chan, ij, now))
    if r is None:
        (out / f"live_{t0:%Y%m%dT%H%M}.json").write_text(json.dumps(rep, indent=1))
        sys.exit(1)
    alerts = AlertTracker(cfg).update(r["alerts"], r["probs"], E)
    rep.update(
        input_shape=list(x.shape), input_finite_fraction=float(np.isfinite(x).mean()),
        input_channel_nan_fraction=[round(float(np.isnan(x[c]).mean()), 3) for c in range(x.shape[0])],
        stations=dict(n=int(len(now)) if now is not None else 0, dust=int((now == 1).sum()) if now is not None else 0,
                      clear=int((now == -1).sum()) if now is not None else 0),
        p_max=[round(float(r["probs"][l].max()), 3) for l in range(3)],
        p_mean=[round(float(r["probs"][l].mean()), 4) for l in range(3)],
        gate_mean=round(float(r["gate"].mean()), 3), alerts=alerts, n_alerts=len(alerts),
        total_seconds=round(sum(s["seconds"] for s in rep["stages"].values()), 1))
    (out / f"live_{t0:%Y%m%dT%H%M}.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("input_channel_nan_fraction",)}, indent=1, default=str))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext = [E.grid.lon_min, E.grid.lon_max, E.grid.lat_min, E.grid.lat_max]
    lon10, lat10 = zip(*I10)
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs[0, 0].imshow(E.last_rgb, origin="lower", extent=ext); axs[0, 0].set_title(f"Dust RGB, live, {t0:%Y-%m-%d %H:%M}Z")
    for l, lead in enumerate(E.leads):
        ax = axs[0, l + 1] if l < 2 else axs[1, 0]
        im = ax.imshow(r["probs"][l], origin="lower", extent=ext, vmin=0, vmax=1, cmap="magma"); ax.set_title(f"P(dust) +{lead} min"); plt.colorbar(im, ax=ax, fraction=0.035)
    axs[1, 1].imshow(r["gate"], origin="lower", extent=ext, vmin=0, vmax=1, cmap="YlOrBr"); axs[1, 1].set_title("Ground gate (max over gated members)")
    axs[1, 2].imshow(chan[0] if chan is not None else np.zeros(x.shape[1:]), origin="lower", extent=ext, vmin=-1, vmax=1, cmap="coolwarm"); axs[1, 2].set_title("Station channel (red = dust now)")
    for ax in axs.ravel():
        ax.plot(lon10, lat10, color="white", lw=1.4, alpha=0.9); ax.set_axis_off()
    fig.suptitle(f"DustAhead live smoke test · {len(alerts)} alert(s) · {rep['total_seconds']} s end to end", fontsize=14)
    fig.tight_layout(); fig.savefig(out / f"live_{t0:%Y%m%dT%H%M}.png", dpi=110); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
