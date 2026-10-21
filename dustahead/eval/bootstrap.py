"""Event-level bootstrap confidence intervals and Holm-corrected paired comparisons.

Samples from the same event are correlated, so we resample EVENTS (not samples) with replacement,
B = 1000 times, and report the 2.5-97.5 percentile interval of each metric.
"""
from __future__ import annotations

import numpy as np

from .metrics import station_scores


def event_bootstrap(p: np.ndarray, y: np.ndarray, event_ids: np.ndarray, metric: str = "csi",
                    B: int = 1000, threshold: float = 0.5, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    events = np.unique(event_ids)
    if len(events) == 0:
        return dict(metric=metric, point=float("nan"), lo=float("nan"), hi=float("nan"), B=B)
    idx_by_ev = {e: np.flatnonzero(event_ids == e) for e in events}
    vals = []
    for _ in range(B):
        pick = rng.choice(events, len(events), replace=True)
        idx = np.concatenate([idx_by_ev[e] for e in pick])
        vals.append(station_scores(p[idx], y[idx], threshold)[metric])
    vals = np.asarray(vals, float)
    point = station_scores(p, y, threshold)[metric]
    return dict(metric=metric, point=point, lo=float(np.nanpercentile(vals, 2.5)), hi=float(np.nanpercentile(vals, 97.5)), B=B)


def paired_bootstrap_pvalue(p_a, p_b, y, event_ids, metric="csi", B=1000, threshold=0.5, seed=0) -> float:
    """Two-sided p-value that model A beats model B on `metric` (difference's bootstrap distribution)."""
    rng = np.random.default_rng(seed)
    events = np.unique(event_ids)
    if len(events) == 0:
        return float("nan")
    idx_by_ev = {e: np.flatnonzero(event_ids == e) for e in events}
    diffs = []
    for _ in range(B):
        idx = np.concatenate([idx_by_ev[e] for e in rng.choice(events, len(events), replace=True)])
        diffs.append(station_scores(p_a[idx], y[idx], threshold)[metric] - station_scores(p_b[idx], y[idx], threshold)[metric])
    diffs = np.asarray(diffs)
    frac = np.mean(diffs <= 0)
    return float(2 * min(frac, 1 - frac))


def holm(pvalues: dict) -> dict:
    """Holm-Bonferroni step-down correction over a dict name -> p."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for k, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - k) * p))
        out[name] = running
    return out
