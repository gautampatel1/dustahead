"""Synthetic haboob samples for tests, smoke training and the dashboard demo (no downloads).

A storm cell appears, cools (IR brightness temperature drops), produces lightning, and pushes a
dust front outward over bare ground. Dust at t0+lead is where the front will be. The physics
prior and ground gate matter because dust is emitted only where bare fraction > 0.5.
Clearly NOT real data: use only to check that code runs end to end.
"""
from __future__ import annotations

import numpy as np

from ..grid import Grid
from .sampler import N_CH, CH, BANDS, assemble


def synthetic_sample(grid: Grid, rng: np.random.Generator, leads=(15, 30, 60), dust: bool = True) -> dict:
    ny, nx = grid.shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    cx, cy = rng.uniform(0.3, 0.7) * nx, rng.uniform(0.3, 0.7) * ny
    ux, uy = rng.uniform(-1, 1, 2)                       # outflow direction (cells / 10 min)
    speed = rng.uniform(1.5, 3.0)
    ux, uy = speed * np.array([ux, uy]) / (np.hypot(ux, uy) + 1e-6)

    # static ground: bare desert with a wet/irrigated patch and some hills
    bare = np.clip(0.8 + 0.2 * np.sin(xx / 17) * np.cos(yy / 23) - 0.9 * (np.hypot(xx - 0.2 * nx, yy - 0.3 * ny) < 25), 0, 1)
    elev = 400 + 600 * np.exp(-((xx - 0.8 * nx) ** 2 + (yy - 0.8 * ny) ** 2) / 400)
    static = np.stack([elev, np.zeros_like(elev), bare, np.full_like(elev, 0.52), np.full_like(elev, 60.0), np.full_like(elev, 12.0)]).astype(np.float32)

    sat, flashes = [], []
    for k in range(6):                                   # frames t0-50 ... t0
        age = k + 1
        r_cell = 6 + 2.0 * age
        cell = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r_cell ** 2))
        bt = 300 - 70 * cell + rng.normal(0, 0.5, (ny, nx))            # cold cloud top
        front = ((xx - cx) * ux + (yy - cy) * uy) / speed               # minutes ahead of the cell, in 10-min units
        plume = np.exp(-(front - age * 0.6) ** 2 / 8) * (np.hypot(xx - cx, yy - cy) < 8 * age) if dust else 0
        bands = {2: np.clip(0.3 + 0.5 * cell + 0.2 * plume, 0, 1), 7: bt + 10, 11: bt - 2 * plume,
                 13: bt - 1.0 * plume, 14: bt, 15: bt - 3.5 * plume * bare}       # dust: 12.3-10.3 negative
        sat.append({b: bands[b].astype(np.float32) for b in BANDS})
        flashes.append(rng.poisson(3 * cell * (age > 1)).astype(np.float32))

    wind10 = 3 + 12 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 25 ** 2))
    hr = dict(u10=wind10 * ux / speed, v10=wind10 * uy / speed, gust=wind10 * 1.6, cape=np.full_like(wind10, 1200.0),
              dcape=np.full_like(wind10, 900.0), rh2m=np.full_like(wind10, 20.0), soilm=np.full_like(wind10, 0.15),
              rain7d=np.zeros_like(wind10))
    from .physics import erodibility_prior
    prior = erodibility_prior(wind10, hr["soilm"] * 25.0, static[5], bare)
    x = assemble(sat, flashes, hr, static, prior)

    y = np.zeros((len(leads), ny, nx), np.float32)
    for li, lead in enumerate(leads):
        if dust:
            d = 6 * 0.6 + lead / 10 * 0.6
            front = ((xx - cx) * ux + (yy - cy) * uy) / speed
            y[li] = ((np.abs(front - d) < 2.5) & (np.hypot(xx - cx, yy - cy) < 8 * 6 + lead / 10 * 8) & (bare > 0.5))
    w = np.ones_like(y)
    n_st = 12
    si, sj = rng.integers(0, ny, n_st), rng.integers(0, nx, n_st)
    st_y = np.stack([y[li][si, sj] for li in range(len(leads))], 1).astype(np.int8)
    return dict(x=x.astype(np.float16), y=y.astype(np.uint8), w=w.astype(np.float16),
                st_ij=np.stack([si, sj], 1).astype(np.int16), st_y=st_y,
                st_now=np.zeros(n_st, np.int8), t0=np.datetime64("2025-08-26T02:00"),
                genesis=np.array(True), dust_rgb=np.zeros((ny, nx, 3), np.uint8))


def make_synthetic_dataset(root, grid: Grid, n: int = 24, seed: int = 0, small: bool = False) -> None:
    """Writes n .npz samples to root/{train,val,test}/ (70/15/15). small=True uses a 64x80 grid."""
    from pathlib import Path
    from .sampler import save_sample
    rng = np.random.default_rng(seed)
    g = Grid(grid.lon_min, grid.lon_max, grid.lat_min, grid.lat_max, 80, 64) if small else grid
    root = Path(root)
    for k in range(n):
        split = "train" if k < 0.7 * n else ("val" if k < 0.85 * n else "test")
        s = synthetic_sample(g, rng, dust=(k % 3 != 2))
        save_sample(root / split / f"synthetic_{k:04d}.npz", s)
