"""The product interface: predict(t0) -> dust probability maps + road alerts.

Live mode fetches the last hour of GOES/GLM/HRRR for t0 = now (rounded down to 10 min), builds
the 62-channel stack exactly as in training, paints the station-observation channel from the
latest ASOS (Automated Surface Observing System) reports, runs the model ENSEMBLE and returns:
  probs   (L, ny, nx) float32    P(dust) at +15/+30/+60 min
  gate    (ny, nx)               sigmoid(h): where the ground can emit (for the admin view)
  alerts  list of dicts          road segments whose max P over the next 60 min exceeds the threshold

Ensemble: `checkpoint` may be a list; the product probability is the element-wise MAX of the members
(validated: max(unet_all v4, gated_unet v4) beats either member and station persistence at every lead).
Alert policy: one threshold per lead (cfg.product.thresholds) and, optionally, a lower one for GENESIS
road cells - cells with no station reporting dust within 50 km (cfg.product.genesis_thresholds).
AlertTracker adds hysteresis across consecutive forecasts: an alert switches on at the threshold, stays on
for at least `min_hold` forecasts and is released only when P drops below `release`, so a road segment is not
warned and un-warned every 10 minutes when a station report is late.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import torch

from .. import load_config
from ..data import goes, glm, hrrr, physics
from ..data.sampler import BANDS, assemble, genesis_per_station, paint_station_now
from ..data.static import NAMES as STATIC_NAMES, load_static
from ..events.stations import fetch_asos, fetch_station_table, has_code
from ..grid import Grid
from ..train.train import load_checkpoint

# Interstate 10 waypoints (lon, lat) across the domain, west -> east; refined from OSM in Phase 2
I10 = [(-114.72, 33.61), (-113.90, 33.63), (-113.20, 33.55), (-112.70, 33.43), (-112.07, 33.46), (-111.95, 33.38),
       (-111.75, 33.10), (-111.55, 32.90), (-111.30, 32.70), (-111.05, 32.42), (-110.96, 32.20), (-110.75, 32.00),
       (-110.40, 32.07), (-110.08, 32.17), (-109.78, 32.26), (-109.45, 32.25), (-109.04, 32.27), (-108.70, 32.33),
       (-108.20, 32.28), (-107.75, 32.26), (-107.30, 32.26), (-106.90, 32.30), (-106.65, 32.20), (-106.45, 31.95),
       (-106.38, 31.81)]


class DustAhead:
    def __init__(self, checkpoint, cfg: dict | None = None, device: str | None = None):
        self.cfg = cfg or load_config()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        paths = [checkpoint] if isinstance(checkpoint, (str, Path)) else list(checkpoint)
        self.members = [load_checkpoint(p, self.device) for p in paths]      # [(model, ck), ...]
        self.model = self.members[0][0]                                      # kept for single-model callers
        self.temperature = self.members[0][1].get("temperature", 1.0)
        self.grid = Grid.from_config(self.cfg)
        self.static = load_static(self.cfg["paths"]["static"], self.grid)
        self.leads = self.cfg["time"]["leads_min"]
        self.stations = None            # station table, loaded on first live call
        self.last_station = None        # (st_ij, st_now) of the last live station fetch

    # ---- inputs ----
    def build_live_input(self, t0: dt.datetime) -> np.ndarray:
        T, S, P = self.cfg["time"], self.cfg["satellite"], self.cfg["paths"]
        hrrr.set_cache(P["cache"])
        sat, flashes = [], []
        for t in goes.frame_times(t0, T["history_frames"], T["step_min"]):
            f = goes.nearest_file(S["bucket"], S["product"], t)
            if f is None:
                raise RuntimeError(f"no GOES file near {t}")
            sat.append(goes.frame_cached(f, self.grid, Path(P["cache"]) / "goes", BANDS))
            flashes.append(glm.flash_density_at(t, self.grid, self.cfg))
        hr = hrrr.hrrr_stack(t0, self.grid)
        hr["rain7d"] = hrrr.hrrr_7day_rain(t0, self.grid, days=7)
        clay, bare = self.static[STATIC_NAMES.index("clay")], self.static[STATIC_NAMES.index("bare")]
        prior = physics.erodibility_prior(hr["wind10"], hr["soilm"] * 25.0, clay, bare)
        self.last_rgb = goes.dust_rgb(sat[-1])
        return assemble(sat, flashes, hr, self.static, prior)

    def station_now_live(self, t0: dt.datetime, window_min: int = 20) -> tuple[np.ndarray, np.ndarray]:
        """What the ASOS stations say at t0: +1 dust (vis <= 1 mi and a dust code in a report within
        `window_min` before t0), -1 clear report in the window, 0 no report. Live analogue of st_now."""
        L = self.cfg["label"]
        if self.stations is None:
            f = Path(self.cfg["paths"]["labels"]) / "stations.csv"
            import pandas as pd
            self.stations = pd.read_csv(f) if f.exists() else fetch_station_table(self.grid)
        st = self.stations
        import pandas as pd
        t0p = pd.Timestamp(t0).tz_localize("UTC") if pd.Timestamp(t0).tzinfo is None else pd.Timestamp(t0)
        now = np.zeros(len(st), np.int8)
        try:
            obs = fetch_asos(list(st.sid), t0p - pd.Timedelta(minutes=window_min), t0p, tuple(L["asos_report_types"]))
            obs = obs[obs.valid <= t0p].sort_values("valid").groupby("station").tail(3)
            dust = obs[(obs.vsby <= L["vis_max_mi"]) & obs.wxcodes.apply(lambda w: has_code(w, L["dust_codes"]))]
            reported = set(obs.station)
            dusty = set(dust.station)
            for k, sid in enumerate(st.sid):
                now[k] = 1 if sid in dusty else (-1 if sid in reported else 0)
        except Exception as e:  # noqa: BLE001  - no network: run with "no report" everywhere and say so
            print(f"[DustAhead] live ASOS fetch failed ({type(e).__name__}): station channel = no report")
        ij = np.stack([st.i.values, st.j.values], 1).astype(np.int64)
        self.last_station = (ij, now)
        return ij, now

    # ---- inference ----
    @torch.no_grad()
    def predict_array(self, x: np.ndarray, station_now: np.ndarray | None = None, st_ij=None, st_now=None) -> dict:
        from ..data.sampler import N_CH_IN
        if x.shape[0] < N_CH_IN:   # live mode: append the painted station channel (zeros = no report anywhere)
            x = np.concatenate([x, (station_now if station_now is not None else np.zeros((1,) + x.shape[1:], np.float32))], 0)
        xt = torch.from_numpy(x.astype(np.float32))[None].to(self.device)
        probs, gate, members = None, None, {}
        for model, ck in self.members:
            p, z, h = model(xt, return_parts=True)
            pm = p[0].float().cpu().numpy()
            T = ck.get("temperature", 1.0)
            if T != 1.0:
                from ..eval.metrics import apply_temperature
                pm = apply_temperature(pm, T)
            members[ck.get("name", "model")] = pm
            probs = pm if probs is None else np.maximum(probs, pm)
            if h is not None:                       # gate shown = most permissive member (max), consistent with the max ensemble
                g = torch.sigmoid(h)[0, 0].float().cpu().numpy()
                gate = g if gate is None else np.maximum(gate, g)
        if gate is None:
            gate = np.ones(probs.shape[1:], np.float32)
        genesis_map = self.genesis_map(st_ij, st_now, probs.shape[1:])
        return dict(probs=probs, gate=gate, members=members, genesis=genesis_map,
                    alerts=self.road_alerts(probs, genesis_map=genesis_map))

    def genesis_map(self, st_ij, st_now, shape, radius_cells: int = 12) -> np.ndarray:
        """True where no station within ~50 km reports dust now (unknown stations -> genesis)."""
        g = np.ones(shape, bool)
        if st_ij is None or st_now is None:
            return g
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        for (i, j), n in zip(st_ij, st_now):
            if n == 1:
                g &= np.hypot(yy - i, xx - j) > radius_cells
        return g

    def predict(self, t0: dt.datetime | None = None) -> dict:
        t0 = t0 or dt.datetime.now(dt.timezone.utc)
        t0 = t0.replace(minute=t0.minute // 10 * 10, second=0, microsecond=0)
        x = self.build_live_input(t0)
        ij, now = self.station_now_live(t0)
        out = self.predict_array(x, paint_station_now(ij, now, x.shape[1:])[None], ij, now)
        out.update(t0=t0, dust_rgb=self.last_rgb, station_ij=ij, station_now=now)
        return out

    # ---- road alerts ----
    def road_cells(self, n_per_segment: int = 12) -> list[tuple[str, np.ndarray, np.ndarray]]:
        segs = []
        for k in range(len(I10) - 1):
            (x0, y0), (x1, y1) = I10[k], I10[k + 1]
            lon = np.linspace(x0, x1, n_per_segment)
            lat = np.linspace(y0, y1, n_per_segment)
            i, j = self.grid.to_ij(lon, lat)
            segs.append((f"I-10 seg {k:02d} ({x0:.2f},{y0:.2f})->({x1:.2f},{y1:.2f})", i, j))
        return segs

    def road_alerts(self, probs: np.ndarray, threshold: float | None = None, genesis_map: np.ndarray | None = None) -> list[dict]:
        """One alert per road segment at the earliest lead whose max P exceeds that lead's threshold.
        `threshold` (a single number) overrides the per-lead policy, e.g. from the dashboard slider."""
        P = self.cfg.get("product", {})
        thr = [threshold] * len(self.leads) if threshold else P.get("thresholds", [self.cfg["eval"]["threshold"]] * len(self.leads))
        thr_g = [threshold] * len(self.leads) if threshold else P.get("genesis_thresholds", thr)
        severe = P.get("severe_margin", 0.1)
        alerts = []
        for name, i, j in self.road_cells():
            gen = bool(genesis_map[i, j].all()) if genesis_map is not None else False
            for l, lead in enumerate(self.leads):
                t = thr_g[l] if gen else thr[l]
                pm = float(probs[l][i, j].max())
                if pm >= t:
                    alerts.append(dict(segment=name, lead_min=lead, p_max=pm, genesis=gen,
                                       level="severe" if pm >= t + severe else "watch"))
                    break   # report the earliest lead that triggers
        return alerts


class AlertTracker:
    """Hysteresis over consecutive forecasts (10-min cadence). State per road segment."""

    def __init__(self, cfg: dict):
        P = cfg.get("product", {})
        self.release = float(P.get("release_threshold", 0.5))
        self.min_hold = int(P.get("min_hold_steps", 3))          # 3 x 10 min = 30 min minimum warning
        self.state: dict[str, dict] = {}

    def update(self, alerts: list[dict], probs: np.ndarray, engine: "DustAhead") -> list[dict]:
        """`alerts` = fresh threshold alerts for this forecast; returns the alerts to SHOW now."""
        fresh = {a["segment"]: a for a in alerts}
        pmax = {name: float(probs[:, i, j].max()) for name, i, j in engine.road_cells()}
        shown = []
        for seg, p in pmax.items():
            st = self.state.get(seg)
            if seg in fresh:
                a = dict(fresh[seg], held=False)
                self.state[seg] = dict(alert=a, steps=(st["steps"] + 1 if st else 1))
                shown.append(a)
            elif st and (st["steps"] < self.min_hold or p >= self.release):
                st["steps"] += 1
                a = dict(st["alert"], p_max=p, held=True, level="watch")
                shown.append(a)
            elif st:
                del self.state[seg]
        return shown
