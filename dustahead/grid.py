"""The DustAhead analysis grid: a regular latitude/longitude grid over Arizona and New Mexico.

Every data source (satellite, lightning, weather model, static maps, stations) is
mapped onto this grid, so a sample is a stack of (ny, nx) arrays.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EARTH_KM = 6371.0


@dataclass(frozen=True)
class Grid:
    lon_min: float = -115.0
    lon_max: float = -103.0
    lat_min: float = 30.5
    lat_max: float = 36.5
    nx: int = 250
    ny: int = 200

    @classmethod
    def from_config(cls, cfg: dict) -> "Grid":
        d = cfg["domain"]
        return cls(d["lon_min"], d["lon_max"], d["lat_min"], d["lat_max"], d["nx"], d["ny"])

    # ---- cell geometry -------------------------------------------------
    @property
    def dlon(self) -> float:
        return (self.lon_max - self.lon_min) / self.nx

    @property
    def dlat(self) -> float:
        return (self.lat_max - self.lat_min) / self.ny

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    def lons(self) -> np.ndarray:
        """Cell-centre longitudes, length nx."""
        return self.lon_min + (np.arange(self.nx) + 0.5) * self.dlon

    def lats(self) -> np.ndarray:
        """Cell-centre latitudes, length ny (row 0 is the SOUTH edge)."""
        return self.lat_min + (np.arange(self.ny) + 0.5) * self.dlat

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """2-D (lon, lat) arrays of shape (ny, nx)."""
        return np.meshgrid(self.lons(), self.lats())

    def cell_km(self) -> tuple[float, float]:
        """Approximate cell size (east-west km, north-south km) at the domain centre."""
        lat_c = np.deg2rad(0.5 * (self.lat_min + self.lat_max))
        return (np.deg2rad(self.dlon) * EARTH_KM * np.cos(lat_c), np.deg2rad(self.dlat) * EARTH_KM)

    # ---- coordinate transforms -----------------------------------------
    def to_ij(self, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        """Lon/lat -> (row, col) integer indices. Values outside the grid are clipped."""
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        j = np.floor((lon - self.lon_min) / self.dlon).astype(int)
        i = np.floor((lat - self.lat_min) / self.dlat).astype(int)
        return np.clip(i, 0, self.ny - 1), np.clip(j, 0, self.nx - 1)

    def inside(self, lon, lat) -> np.ndarray:
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        return (lon >= self.lon_min) & (lon < self.lon_max) & (lat >= self.lat_min) & (lat < self.lat_max)

    def distance_km(self, lon0: float, lat0: float) -> np.ndarray:
        """Great-circle distance from (lon0, lat0) to every cell centre, shape (ny, nx)."""
        lon, lat = self.mesh()
        p1, p2 = np.deg2rad(lat0), np.deg2rad(lat)
        dphi = p2 - p1
        dlmb = np.deg2rad(lon - lon0)
        a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
        return 2 * EARTH_KM * np.arcsin(np.sqrt(a))

    def disk_mask(self, lon0: float, lat0: float, radius_km: float) -> np.ndarray:
        return self.distance_km(lon0, lat0) <= radius_km

    def bin_points(self, lon, lat, weights=None) -> np.ndarray:
        """Count (or sum weights of) points per cell. Used for lightning flash density."""
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        keep = self.inside(lon, lat)
        out = np.zeros(self.shape, dtype=np.float32)
        if keep.sum() == 0:
            return out
        i, j = self.to_ij(lon[keep], lat[keep])
        w = None if weights is None else np.asarray(weights, float)[keep]
        np.add.at(out, (i, j), 1.0 if w is None else w)
        return out
