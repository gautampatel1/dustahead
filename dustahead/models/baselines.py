"""Non-learned baselines, all producing (L, ny, nx) probabilities from one sample's input x.

persistence      dust now (ADP-like proxy: 12.3-10.3 um BTD < -0.5 K over bare ground) stays put
advected         the same mask shifted by the HRRR 10 m wind x lead time (ADP-advected baseline)
hrrr_rule        wind-threshold rule: p = erodibility prior E (physics channel) - no satellite at all
pysteps_flow     optical-flow extrapolation of the dust mask (needs `pip install pysteps`)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import shift as nd_shift

from ..data.sampler import CH, BANDS

B = {b: k for k, b in enumerate(BANDS)}   # band -> offset within a frame


def dust_now_mask(x: np.ndarray, btd_thresh: float = -2.5) -> np.ndarray:
    """Split-window brightness-temperature difference at t0 (frame 5): dust has BT12.3 << BT10.3.
    Clear desert is already slightly negative (water vapour), so the threshold is -2.5 K, and the pixel
    must be warmer than 265 K (not cloud) and over mostly bare ground."""
    f = 5 * 6
    btd = (x[f + B[15]] - x[f + B[13]]) * 30.0      # undo /30 normalisation
    bt13 = x[f + B[13]] * 30.0 + 273.15
    bare = x[CH["static"].start + 2]
    return ((btd < btd_thresh) & (bt13 > 265.0) & (bare > 0.3)).astype(np.float32)


def persistence(x: np.ndarray, leads=(15, 30, 60)) -> np.ndarray:
    m = dust_now_mask(x)
    return np.stack([m for _ in leads])


def advected(x: np.ndarray, leads=(15, 30, 60), cell_km=(4.5, 3.3)) -> np.ndarray:
    m = dust_now_mask(x)
    u = x[CH["hrrr"].start] * 15.0        # m/s
    v = x[CH["hrrr"].start + 1] * 15.0
    out = []
    for lead in leads:
        dx = np.nanmean(u) * lead * 60 / (cell_km[0] * 1000)    # cells
        dy = np.nanmean(v) * lead * 60 / (cell_km[1] * 1000)
        out.append(nd_shift(m, (dy, dx), order=0, mode="constant"))
    return np.stack(out)


def hrrr_rule(x: np.ndarray, leads=(15, 30, 60)) -> np.ndarray:
    e = x[CH["physics"].start]
    return np.stack([e for _ in leads])


def pysteps_flow(x: np.ndarray, leads=(15, 30, 60)) -> np.ndarray:
    from pysteps import motion, nowcasts
    frames = np.stack([(x[6 * t + B[15]] - x[6 * t + B[13]]) * -30.0 for t in range(6)])   # dust "intensity"
    frames = np.clip(frames, 0, None)
    V = motion.get_method("LK")(frames[-3:])
    n = max(leads) // 10
    fc = nowcasts.get_method("extrapolation")(frames[-1], V, n)
    return np.stack([(fc[lead // 10 - 1] > 0.5).astype(np.float32) for lead in leads])


def station_persistence(sample: dict, leads=(15, 30, 60)) -> np.ndarray:
    """Classic persistence at STATION level: forecast dust at t0+lead wherever a station reports dust at t0
    (12 km disc). Needs the sample dict (st_ij, st_now), not just x. The strongest honest baseline."""
    x = sample["x"]
    L, H, W = len(leads), x.shape[1], x.shape[2]   # x may carry the extra station channel; only its shape is used
    out = np.zeros((L, H, W), np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    for (i, j), now in zip(sample["st_ij"], sample["st_now"]):
        if now == 1:
            out[:, (np.hypot(yy - i, xx - j) <= 3)] = 1.0      # ~12 km at 4 km cells
    return out


BASELINES = {"persistence": persistence, "advected": advected, "hrrr_rule": hrrr_rule, "pysteps_flow": pysteps_flow}
SAMPLE_BASELINES = {"station_persistence": station_persistence}
