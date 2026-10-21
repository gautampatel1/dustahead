"""The product interface: predict(t0) -> dust probability maps + road alerts.

Live mode fetches the last hour of GOES/GLM/HRRR for t0 = now (rounded down to 10 min), builds
the 62-channel stack exactly as in training (no station labels needed), runs the model and
returns:
  probs   (L, ny, nx) float32    P(dust) at +15/+30/+60 min
  gate    (ny, nx)               sigmoid(h): where the ground can emit (for the admin view)
  alerts  list of dicts          road segments whose max P over the next 60 min exceeds the threshold
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import torch

from .. import load_config
from ..data import goes, glm, hrrr, physics
from ..data.sampler import BANDS, assemble
from ..data.static import NAMES as STATIC_NAMES, load_static
from ..grid import Grid
from ..train.train import load_checkpoint

# Interstate 10 waypoints (lon, lat) across the domain, west -> east; refined from OSM in Phase 2
I10 = [(-114.72, 33.61), (-113.90, 33.63), (-113.20, 33.55), (-112.70, 33.43), (-112.07, 33.46), (-111.95, 33.38),
       (-111.75, 33.10), (-111.55, 32.90), (-111.30, 32.70), (-111.05, 32.42), (-110.96, 32.20), (-110.75, 32.00),
       (-110.40, 32.07), (-110.08, 32.17), (-109.78, 32.26), (-109.45, 32.25), (-109.04, 32.27), (-108.70, 32.33),
       (-108.20, 32.28), (-107.75, 32.26), (-107.30, 32.26), (-106.90, 32.30), (-106.65, 32.20), (-106.45, 31.95),
       (-106.38, 31.81)]


class DustAhead:
    def __init__(self, checkpoint: str | Path, cfg: dict | None = None, device: str | None = None):
        self.cfg = cfg or load_config()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, ck = load_checkpoint(checkpoint, self.device)
        self.temperature = ck.get("temperature", 1.0)
        self.grid = Grid.from_config(self.cfg)
        self.static = load_static(self.cfg["paths"]["static"], self.grid)
        self.leads = self.cfg["time"]["leads_min"]

    # ---- inputs ----
    def build_live_input(self, t0: dt.datetime) -> np.ndarray:
        T, S, P = self.cfg["time"], self.cfg["satellite"], self.cfg["paths"]
        sat, flashes = [], []
        for t in goes.frame_times(t0, T["history_frames"], T["step_min"]):
            f = goes.nearest_file(S["bucket"], S["product"], t)
            if f is None:
                raise RuntimeError(f"no GOES file near {t}")
            sat.append(goes.read_bands(goes.download(f, P["raw"] + "/goes"), self.grid, BANDS))
            flashes.append(glm.flash_density_at(t, self.grid, self.cfg))
        hr = hrrr.hrrr_stack(t0, self.grid)
        hr["rain7d"] = hrrr.hrrr_7day_rain(t0, self.grid, 168, 6)
        clay, bare = self.static[STATIC_NAMES.index("clay")], self.static[STATIC_NAMES.index("bare")]
        prior = physics.erodibility_prior(hr["wind10"], hr["soilm"] * 100, clay, bare)
        self.last_rgb = goes.dust_rgb(sat[-1])
        return assemble(sat, flashes, hr, self.static, prior)

    # ---- inference ----
    @torch.no_grad()
    def predict_array(self, x: np.ndarray) -> dict:
        xt = torch.from_numpy(x.astype(np.float32))[None].to(self.device)
        p, z, h = self.model(xt, return_parts=True)
        probs = p[0].cpu().numpy()
        if self.temperature != 1.0:
            from ..eval.metrics import apply_temperature
            probs = apply_temperature(probs, self.temperature)
        gate = torch.sigmoid(h)[0, 0].cpu().numpy() if h is not None else np.ones(self.grid.shape, np.float32)
        return dict(probs=probs, gate=gate, alerts=self.road_alerts(probs))

    def predict(self, t0: dt.datetime | None = None) -> dict:
        t0 = t0 or dt.datetime.now(dt.timezone.utc)
        t0 = t0.replace(minute=t0.minute // 10 * 10, second=0, microsecond=0)
        x = self.build_live_input(t0)
        out = self.predict_array(x)
        out.update(t0=t0, dust_rgb=self.last_rgb)
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

    def road_alerts(self, probs: np.ndarray, threshold: float | None = None) -> list[dict]:
        thr = threshold or self.cfg["eval"]["threshold"]
        alerts = []
        for name, i, j in self.road_cells():
            for l, lead in enumerate(self.leads):
                pm = float(probs[l][i, j].max())
                if pm >= thr:
                    alerts.append(dict(segment=name, lead_min=lead, p_max=pm, level="severe" if pm >= 0.8 else "watch"))
                    break   # report the earliest lead that triggers
        return alerts
