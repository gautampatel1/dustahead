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
from ..data.sampler import BANDS, N_CH_IN, assemble, assemble_forecast, genesis_per_station, paint_station_now, wants_forecast
from ..data.static import NAMES as STATIC_NAMES, load_static
from ..events.stations import fetch_asos, fetch_station_table, has_code
from ..grid import Grid
from ..train.train import load_checkpoint

# Interstate 10 waypoints, west -> east: (name, state, exit number, lon, lat). Each waypoint IS the I-10 exit for that
# place (OpenStreetMap highway=motorway_junction nodes, (c) OpenStreetMap contributors, ODbL; Overpass extract of
# 2026-07-24), so the name, the position and the milepost (exit numbers are mileposts in AZ, NM and TX) cannot drift
# apart. Until 2026-11 the coordinates and names were kept as two separate lists and the names sat about one waypoint
# east of their positions from Chandler on (e.g. "Eloy -> Picacho Peak" covered Picacho Peak -> north Tucson).
# Straight lines between these exits stay within ~3 km of the road (~5 km between Anthony and El Paso), under one
# 4-km grid cell; OSM road geometry replaces the straight lines in Phase 2.
I10_EXITS = [
    ("Ehrenberg (CA line)", "AZ", 1, -114.519, 33.603), ("Quartzsite", "AZ", 17, -114.240, 33.660),
    ("Tonopah", "AZ", 94, -112.937, 33.497), ("Buckeye", "AZ", 114, -112.591, 33.434),
    ("Phoenix", "AZ", 145, -112.070, 33.462), ("Tempe", "AZ", 153, -111.984, 33.411),
    ("Chandler", "AZ", 160, -111.972, 33.303), ("Casa Grande", "AZ", 194, -111.687, 32.884),
    ("Eloy", "AZ", 208, -111.555, 32.737), ("Picacho Peak", "AZ", 219, -111.399, 32.651),
    ("Marana", "AZ", 236, -111.222, 32.466), ("Tucson", "AZ", 258, -110.982, 32.228),
    ("Vail", "AZ", 279, -110.724, 32.029), ("Benson", "AZ", 304, -110.312, 31.978),
    ("Dragoon", "AZ", 318, -110.097, 32.040), ("Willcox", "AZ", 340, -109.847, 32.262),
    ("Bowie", "AZ", 362, -109.519, 32.332), ("San Simon", "AZ", 378, -109.256, 32.275),
    ("Lordsburg", "NM", 22, -108.717, 32.344), ("Deming", "NM", 82, -107.762, 32.270),
    ("Akela", "NM", 102, -107.423, 32.239), ("Las Cruces West", "NM", 135, -106.890, 32.280),
    ("Las Cruces", "NM", 142, -106.769, 32.281), ("Anthony", "NM", 162, -106.590, 32.028),
    ("El Paso", "TX", 19, -106.496, 31.760),
]
I10 = [(lon, lat) for _, _, _, lon, lat in I10_EXITS]
I10_NAMES = [name for name, *_ in I10_EXITS]


class DustAhead:
    def __init__(self, checkpoint, cfg: dict | None = None, device: str | None = None):
        self.cfg = cfg or load_config()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        paths = [checkpoint] if isinstance(checkpoint, (str, Path)) else list(checkpoint)
        self.members = [load_checkpoint(p, self.device) for p in paths]      # [(model, ck), ...]
        self.member_names = [Path(p).parent.name for p in paths]            # e.g. unet_all_v4, gated_unet_v8
        self.model = self.members[0][0]                                      # kept for single-model callers
        self.temperature = self.members[0][1].get("temperature", 1.0)
        self.grid = Grid.from_config(self.cfg)
        self.static = load_static(self.cfg["paths"]["static"], self.grid)
        self.leads = self.cfg["time"]["leads_min"]
        self.stations = None            # station table, loaded on first live call
        self.last_station = None        # (st_ij, st_now) of the last live station fetch
        self.needs_forecast = any(wants_forecast(ck["cfg"]) for _, ck in self.members)   # HRRR forecast block wanted
        self.last_xf = None             # forecast block of the last build_live_input

    @classmethod
    def policy_only(cls, cfg: dict | None = None) -> "DustAhead":
        """An engine without models: road segments and the alert policy only. Used by the bundle tests and the
        web fixture so they apply exactly the thresholds, genesis rule and hysteresis the product uses."""
        self = cls.__new__(cls)
        self.cfg = cfg or load_config()
        self.grid = Grid.from_config(self.cfg)
        self.leads = self.cfg["time"]["leads_min"]
        self.members, self.member_names, self.stations, self.last_station = [], [], None, None
        return self

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
        self.last_xf = assemble_forecast(hrrr.forecast_stack(t0, self.grid)) if self.needs_forecast else None
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
    def predict_array(self, x: np.ndarray, station_now: np.ndarray | None = None, st_ij=None, st_now=None,
                      xf: np.ndarray | None = None) -> dict:
        """x: the stored/live 62-channel input (the station channel is appended here if missing). xf: the HRRR
        forecast block (6, ny, nx), required when any member was trained with it (stored samples carry it as `xf`;
        live mode takes it from the last build_live_input). Members index their own channels, so the full stack
        is safe to pass to every member."""
        if x.shape[0] < N_CH_IN:   # live mode: append the painted station channel (zeros = no report anywhere)
            x = np.concatenate([x, (station_now if station_now is not None else np.zeros((1,) + x.shape[1:], np.float32))], 0)
        if self.needs_forecast and x.shape[0] == N_CH_IN:
            xf = xf if xf is not None else self.last_xf
            if xf is None:
                raise ValueError("a member needs the HRRR forecast block: pass xf (sample['xf']) or call build_live_input first")
            x = np.concatenate([x, xf.astype(np.float32)], 0)
        xt = torch.from_numpy(x.astype(np.float32))[None].to(self.device)
        probs, gate, members = None, None, {}
        for (model, ck), mname in zip(self.members, self.member_names):
            p, z, h = model(xt, return_parts=True)
            pm = p[0].float().cpu().numpy()
            T = ck.get("temperature", 1.0)
            if T != 1.0:
                from ..eval.metrics import apply_temperature
                pm = apply_temperature(pm, T)
            members[mname] = pm
            probs = pm if probs is None else np.maximum(probs, pm)
            if h is not None:                       # gate shown = most permissive member (max), consistent with the max ensemble
                g = torch.sigmoid(h)[0, 0].float().cpu().numpy()
                gate = g if gate is None else np.maximum(gate, g)
        if gate is None:
            gate = np.ones(probs.shape[1:], np.float32)
        # Lead-dependent policy: at the short leads listed in product.persistence_leads the forecast is floored by
        # station persistence (P = 1 within 12 km of a station reporting dust NOW). The network already sees those
        # reports through the station channel but assigns them P ~ 0.6-0.9; at +15 min "dust here now" is the best
        # available forecast, and the floor stops the deployed threshold from dropping a storm that is already on the road.
        for lead in self.cfg.get("product", {}).get("persistence_leads", []) or []:
            if lead in self.leads and st_ij is not None and st_now is not None:
                probs[self.leads.index(lead)] = np.maximum(probs[self.leads.index(lead)],
                                                           self.station_persistence_map(st_ij, st_now, probs.shape[1:]))
        genesis_map = self.genesis_map(st_ij, st_now, probs.shape[1:])
        return dict(probs=probs, gate=gate, members=members, genesis=genesis_map,
                    alerts=self.road_alerts(probs, genesis_map=genesis_map))

    @staticmethod
    def station_persistence_map(st_ij, st_now, shape, radius_cells: int = 3) -> np.ndarray:
        """1.0 within ~12 km (3 cells) of every station reporting dust now, else 0 (models/baselines.station_persistence)."""
        m = np.zeros(shape, np.float32)
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        for (i, j), n in zip(st_ij, st_now):
            if n == 1:
                m[np.hypot(yy - i, xx - j) <= radius_cells] = 1.0
        return m

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
            segs.append((f"I-10 {I10_NAMES[k]} \u2192 {I10_NAMES[k + 1]}", i, j))
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
