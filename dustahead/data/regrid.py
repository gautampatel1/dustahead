"""Nearest-neighbour regridding helpers shared by every data source."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..grid import Grid


def _xyz(lon, lat):
    lon, lat = np.deg2rad(np.asarray(lon, float)), np.deg2rad(np.asarray(lat, float))
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


class NearestRegridder:
    """Pre-computes, once, the index of the nearest source pixel for every target cell."""

    def __init__(self, src_lon: np.ndarray, src_lat: np.ndarray, grid: Grid, max_km: float = 8.0):
        ok = np.isfinite(src_lon) & np.isfinite(src_lat)
        self.src_shape = src_lon.shape
        self.src_idx = np.flatnonzero(ok)
        tree = cKDTree(_xyz(src_lon.ravel()[self.src_idx], src_lat.ravel()[self.src_idx]))
        glon, glat = grid.mesh()
        d, k = tree.query(_xyz(glon.ravel(), glat.ravel()))
        self.k = k
        self.valid = (d * 6371.0) <= max_km   # chord ~ arc for small distances
        self.grid = grid

    def __call__(self, src_vals: np.ndarray, fill=np.nan) -> np.ndarray:
        flat = np.asarray(src_vals).reshape(-1)[self.src_idx][self.k].astype(np.float32)
        flat[~self.valid] = fill
        return flat.reshape(self.grid.shape)
