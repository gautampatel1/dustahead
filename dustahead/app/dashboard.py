"""Streamlit dashboard: admin view (maps, gate, alerts) and user view (road alert feed).

    streamlit run dustahead/app/dashboard.py -- [--ckpt a.pt --ckpt b.pt] [--demo] [--samples data/samples/val]

Without --ckpt the ensemble in configs/default.yaml (product.checkpoints) is used.
--demo replays saved samples (default data/samples/val, newest event first) instead of live data, with the
station-observation channel painted from the stored reports, so it works without internet access.
Live mode calls DustAhead.predict() every 10 min.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dustahead import load_config  # noqa: E402
from dustahead.app.predict import AlertTracker, DustAhead, I10  # noqa: E402
from dustahead.data.sampler import load_sample  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", action="append", default=None, help="checkpoint(s); repeat for an ensemble")
ap.add_argument("--demo", action="store_true")
ap.add_argument("--samples", default="data/samples/val")
args, _ = ap.parse_known_args()

st.set_page_config(page_title="DustAhead", layout="wide")
cfg = load_config()
CKPTS = args.ckpt or cfg.get("product", {}).get("checkpoints", ["runs/latest/best.pt"])


@st.cache_resource
def engine():
    return DustAhead(CKPTS, cfg)


E = engine()
view = st.sidebar.radio("View", ["Admin (maps)", "User (alerts)"])
lead = st.sidebar.select_slider("Lead time (min)", options=E.leads, value=E.leads[1])
default_thr = float(cfg.get("product", {}).get("thresholds", [cfg["eval"]["threshold"]])[0])
thr = st.sidebar.slider("Alert threshold", 0.1, 0.95, default_thr, 0.05)
st.sidebar.caption("Models: " + ", ".join(Path(c).parent.name for c in CKPTS) + (" (max ensemble)" if len(CKPTS) > 1 else ""))

if args.demo:
    files = sorted(Path(args.samples).glob("*.npz"))
    events = sorted({f.stem.rsplit("_", 1)[0] for f in files}, reverse=True)
    ev = st.sidebar.selectbox("Event", events, index=0)
    ev_files = [f for f in files if f.stem.rsplit("_", 1)[0] == ev]
    k = st.sidebar.slider("Forecast time within event (10-min steps)", 0, max(len(ev_files) - 1, 0), min(18, len(ev_files) - 1))
    s = load_sample(ev_files[k])
    from dustahead.data.sampler import paint_station_now
    out = E.predict_array(s["x"].astype(np.float32), paint_station_now(s["st_ij"], s["st_now"], s["x"].shape[1:])[None],
                          s["st_ij"], s["st_now"])
    out["t0"] = str(s["t0"])
    out["dust_rgb"] = s["dust_rgb"]
    out["station_ij"], out["station_now"] = s["st_ij"], s["st_now"]
    truth = s["y"]
else:
    if st.sidebar.button("Refresh now") or "live" not in st.session_state:
        st.session_state.setdefault("tracker", AlertTracker(cfg))
        live = E.predict()
        live["alerts"] = st.session_state["tracker"].update(live["alerts"], live["probs"], E)
        st.session_state["live"] = live
    out, truth = st.session_state["live"], None

li = E.leads.index(lead)
probs, gate = out["probs"], out["gate"]
alerts = out["alerts"] if not args.demo else E.road_alerts(probs, thr, genesis_map=out.get("genesis"))


def road_overlay(ax):
    lon, lat = zip(*I10)
    ax.plot(lon, lat, color="white", lw=1.5, alpha=0.8)
    ij, now = out.get("station_ij"), out.get("station_now")
    if ij is not None and now is not None:      # stations: red = reporting dust now, grey = clear, none = no report
        lons, lats = E.grid.lons()[ij[:, 1]], E.grid.lats()[ij[:, 0]]
        ax.scatter(lons[now == 1], lats[now == 1], s=18, c="red", edgecolors="white", lw=0.5, zorder=5)
        ax.scatter(lons[now == -1], lats[now == -1], s=6, c="lightgrey", alpha=0.6, zorder=4)


DARK = "#0e1117"


def panel(draw, caption):
    """One map panel: dark figure matching the Streamlit theme, no axes, road + stations overlaid."""
    import matplotlib.pyplot as plt
    st.caption(caption)
    fig, ax = plt.subplots(figsize=(6, 3.6), facecolor=DARK)
    draw(ax); road_overlay(ax); ax.set_axis_off()
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    st.pyplot(fig, clear_figure=True)


if view.startswith("Admin"):
    st.title(f"DustAhead admin · t0 = {str(out['t0'])[:16]} UTC")
    ext = [E.grid.lon_min, E.grid.lon_max, E.grid.lat_min, E.grid.lat_max]
    c1, c2, c3 = st.columns(3)
    with c1:
        panel(lambda ax: ax.imshow(out["dust_rgb"], origin="lower", extent=ext), "Dust RGB at t0 · pink = dust, orange = thick cloud · red dots = stations reporting dust")
    with c2:
        panel(lambda ax: ax.imshow(probs[li], origin="lower", extent=ext, vmin=0, vmax=1, cmap="magma"), f"P(dust) at +{lead} min, max ensemble · black 0 → yellow 1")
    with c3:
        panel(lambda ax: ax.imshow(gate, origin="lower", extent=ext, vmin=0, vmax=1, cmap="YlOrBr"), "Ground gate σ(h): can the surface emit dust? · pale = closed, brown = open")
    members = out.get("members", {})
    if truth is not None or len(members) > 1:
        cols = st.columns(1 + len(members) if truth is not None else len(members))
        k = 0
        if truth is not None:
            with cols[0]:
                panel(lambda ax: ax.imshow(truth[li], origin="lower", extent=ext, cmap="Greys_r"), f"Target at +{lead} min (demo only: station discs + ADP pixels)")
            k = 1
        for c, (name, pm) in zip(cols[k:], members.items()):
            with c:
                panel(lambda ax, pm=pm: ax.imshow(pm[li], origin="lower", extent=ext, vmin=0, vmax=1, cmap="magma"), f"Member {name} at +{lead} min")
    st.subheader(f"Road alerts ({len(alerts)})")
    gmap = out.get("genesis")
    st.caption(f"threshold {thr:.2f} · stations reporting dust now: {int((out.get('station_now') == 1).sum()) if out.get('station_now') is not None else 'n/a'} · "
               f"genesis cells (no dust within 50 km): {100 * float(gmap.mean()):.0f} % of the domain" if gmap is not None else f"threshold {thr:.2f}")
    if alerts:
        import pandas as pd
        df = pd.DataFrame(alerts)
        df["p_max"] = df["p_max"].map(lambda v: f"{v:.2f}")
        df["genesis"] = df["genesis"].map(lambda g: "onset (no dust nearby)" if g else "dust already nearby")
        st.dataframe(df[["segment", "lead_min", "p_max", "level", "genesis"] + (["held"] if "held" in df else [])], hide_index=True, use_container_width=True)
    else:
        st.info("No segment above the alert threshold.")
else:
    st.title("DustAhead · Interstate 10 dust alerts")
    st.caption(f"Updated {str(out['t0'])[:16]} UTC · forecasts every 10 min · alerts hold at least 30 min")
    if not alerts:
        st.success("No dust expected on I-10 in the next 60 minutes.")
    for a in alerts:
        box = st.error if a["level"] == "severe" else st.warning
        when = "now" if a["lead_min"] <= 15 else f"in about {a['lead_min']} min"
        box(f"### {a['segment']}\n**Dust {when}** · confidence {a['p_max']:.0%}"
            + (" · forming, no station reports dust yet" if a.get("genesis") else " · dust already reported nearby")
            + (" · holding" if a.get("held") else "")
            + "\n\nPull aside, turn off lights, stay buckled, wait it out.")
