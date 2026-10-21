"""Verification metrics. Primary metric: genesis-stratified station CSI at +30 and +60 min.

Contingency counts from (forecast >= threshold) vs observed at stations:
  hits a, false alarms b, misses c, correct negatives d
  POD = a/(a+c)   FAR = b/(a+b)   CSI = a/(a+b+c)
  HSS = 2(ad - bc) / [(a+c)(c+d) + (a+b)(b+d)]
Probabilistic: Brier = mean (p-y)^2;  BSS = 1 - Brier / Brier_climatology;
  ECE (expected calibration error) over 10 bins = sum_k n_k/N |acc_k - conf_k|
Spatial: FSS_n on the painted pixel targets (see train/losses.py).
Lead time vs NWS: delta_e = t_NWS_warning - t_first_alert (positive = we were earlier).
Impact-weighted warned fraction W = sum_i AADT_i * warned_i / sum_i AADT_i over affected road cells.
"""
from __future__ import annotations

import numpy as np


def contingency(p: np.ndarray, y: np.ndarray, threshold: float = 0.5):
    m = y >= 0
    f = p[m] >= threshold
    o = y[m] >= 0.5
    a = int((f & o).sum()); b = int((f & ~o).sum()); c = int((~f & o).sum()); d = int((~f & ~o).sum())
    return a, b, c, d


def scores_from_counts(a, b, c, d) -> dict:
    def div(n, m):
        return float(n) / m if m else float("nan")
    hss_den = (a + c) * (c + d) + (a + b) * (b + d)
    return dict(pod=div(a, a + c), far=div(b, a + b), csi=div(a, a + b + c),
                hss=div(2 * (a * d - b * c), hss_den), n_pos=a + c, n=a + b + c + d)


def brier(p, y):
    m = y >= 0
    return float(np.mean((p[m] - y[m]) ** 2)) if m.any() else float("nan")


def bss(p, y):
    m = y >= 0
    if not m.any():
        return float("nan")
    clim = y[m].mean()
    b_clim = np.mean((clim - y[m]) ** 2)
    return float(1 - brier(p, y) / b_clim) if b_clim > 0 else float("nan")


def ece(p, y, n_bins: int = 10):
    m = y >= 0
    p, y = p[m], y[m]
    if len(p) == 0:
        return float("nan")
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    out = 0.0
    for k in range(n_bins):
        sel = bins == k
        if sel.any():
            out += sel.mean() * abs(y[sel].mean() - p[sel].mean())
    return float(out)


def station_scores(p, y, threshold=0.5) -> dict:
    s = scores_from_counts(*contingency(p, y, threshold))
    s.update(brier=brier(p, y), bss=bss(p, y), ece=ece(p, y))
    return s


def fss_np(p, y, n=5, threshold=0.5) -> float:
    from scipy.ndimage import uniform_filter
    pf = uniform_filter((p >= threshold).astype(float), n, mode="constant")
    of = uniform_filter((y >= 0.5).astype(float), n, mode="constant")
    den = (pf ** 2).sum() + (of ** 2).sum()
    return float(1 - ((pf - of) ** 2).sum() / den) if den > 0 else float("nan")


def temperature_scale(p: np.ndarray, y: np.ndarray, grid=np.linspace(0.5, 3.0, 26)) -> float:
    """Fit a temperature T on the logit of p minimising the Brier score on validation data."""
    m = y >= 0
    logit = np.log(np.clip(p[m], 1e-6, 1 - 1e-6) / (1 - np.clip(p[m], 1e-6, 1 - 1e-6)))
    best, best_b = 1.0, np.inf
    for T in grid:
        q = 1 / (1 + np.exp(-logit / T))
        b = np.mean((q - y[m]) ** 2)
        if b < best_b:
            best, best_b = float(T), b
    return best


def apply_temperature(p, T):
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return 1 / (1 + np.exp(-logit / T))


def lead_time_vs_nws(alert_times, nws_issue_time) -> float | None:
    """Minutes by which the first model alert precedes the NWS warning (positive = earlier)."""
    if alert_times is None or len(alert_times) == 0:
        return None
    return float((nws_issue_time - min(alert_times)).total_seconds() / 60)


def impact_weighted_warned_fraction(warned: np.ndarray, affected: np.ndarray, aadt: np.ndarray) -> float:
    """W = sum(AADT * warned & affected) / sum(AADT * affected) over road cells. AADT = annual average daily traffic."""
    aff = affected > 0
    den = (aadt * aff).sum()
    return float((aadt * (aff & (warned > 0))).sum() / den) if den > 0 else float("nan")
