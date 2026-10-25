"""Streamlit dashboard: admin view (maps, gate, alerts) and user view (road alert feed).

    streamlit run dustahead/app/dashboard.py -- --ckpt runs/<run>/best.pt [--demo]

--demo replays saved samples from data/samples/test (or synthetic ones) instead of live data,
so the dashboard works without internet access. Live mode calls DustAhead.predict() every 10 min.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dustahead import load_config  # noqa: E402
from dustahead.app.predict import DustAhead, I10  # noqa: E402
from dustahead.data.sampler import load_sample  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/latest/best.pt")
ap.add_argument("--demo", action="store_true")
ap.add_argument("--samples", default="data/samples/test")
args, _ = ap.parse_known_args()

st.set_page_config(page_title="DustAhead", layout="wide")
cfg = load_config()


@st.cache_resource
def engine():
    return DustAhead(args.ckpt, cfg)


E = engine()
view = st.sidebar.radio("View", ["Admin (maps)", "User (alerts)"])
lead = st.sidebar.select_slider("Lead time (min)", options=E.leads, value=E.leads[1])
thr = st.sidebar.slider("Alert threshold", 0.1, 0.9, float(cfg["eval"]["threshold"]), 0.05)

if args.demo:
    files = sorted(Path(args.samples).glob("*.npz"))
    k = st.sidebar.slider("Replay sample", 0, max(len(files) - 1, 0), 0)
    s = load_sample(files[k])
    from dustahead.data.sampler import paint_station_now
    out = E.predict_array(s["x"].astype(np.float32), paint_station_now(s["st_ij"], s["st_now"], s["x"].shape[1:])[None])
    out["t0"] = str(s["t0"])
    out["dust_rgb"] = s["dust_rgb"]
    truth = s["y"]
else:
    if st.sidebar.button("Refresh now") or "live" not in st.session_state:
        st.session_state["live"] = E.predict()
    out, truth = st.session_state["live"], None

li = E.leads.index(lead)
probs, gate = out["probs"], out["gate"]
alerts = E.road_alerts(probs, thr)


def road_overlay(ax):
    lon, lat = zip(*I10)
    ax.plot(lon, lat, color="white", lw=1.5, alpha=0.8)


if view.startswith("Admin"):
    import matplotlib.pyplot as plt
    st.title(f"DustAhead admin · t0 = {out['t0']}")
    c1, c2, c3 = st.columns(3)
    ext = [E.grid.lon_min, E.grid.lon_max, E.grid.lat_min, E.grid.lat_max]
    with c1:
        st.caption("Dust RGB at t0 (pink = dust)")
        fig, ax = plt.subplots(figsize=(5, 4)); ax.imshow(out["dust_rgb"], origin="lower", extent=ext); road_overlay(ax); ax.set_axis_off(); st.pyplot(fig)
    with c2:
        st.caption(f"P(dust) at +{lead} min")
        fig, ax = plt.subplots(figsize=(5, 4)); im = ax.imshow(probs[li], origin="lower", extent=ext, vmin=0, vmax=1, cmap="magma"); road_overlay(ax); ax.set_axis_off(); plt.colorbar(im, ax=ax, fraction=0.04); st.pyplot(fig)
    with c3:
        st.caption("Ground gate σ(h): can the surface emit dust?")
        fig, ax = plt.subplots(figsize=(5, 4)); ax.imshow(gate, origin="lower", extent=ext, vmin=0, vmax=1, cmap="YlOrBr"); road_overlay(ax); ax.set_axis_off(); st.pyplot(fig)
    if truth is not None:
        st.caption(f"Painted target at +{lead} min (demo only)")
        fig, ax = plt.subplots(figsize=(5, 4)); ax.imshow(truth[li], origin="lower", extent=ext, cmap="Greys"); road_overlay(ax); ax.set_axis_off(); st.pyplot(fig)
    st.subheader(f"Road alerts ({len(alerts)})")
    st.dataframe(alerts)
else:
    st.title("DustAhead · Interstate 10 dust alerts")
    st.caption(f"Updated {out['t0']} UTC · forecasts every 10 min")
    if not alerts:
        st.success("No dust expected on I-10 in the next 60 minutes.")
    for a in alerts:
        (st.error if a["level"] == "severe" else st.warning)(
            f"**{a['segment']}** · dust likely in **{a['lead_min']} min** (confidence {a['p_max']:.0%}). Slow down, pull aside, lights off.")
