"""Physics prior: can the wind lift dust from this ground right now?

Friction velocity from 10 m wind (neutral log law):
    u* = kappa * U10 / ln(z / z0),        kappa = 0.4, z = 10 m, z0 = roughness length
Threshold friction velocity for a dry bare surface (Marticorena & Bergametti 1995 form, fixed
value per soil class here), raised by soil moisture using Fecan et al. (1999):
    f_w = sqrt(1 + 1.21 * (w - w')^0.68)   for w > w', else 1
    w'  = 0.0014 * clay^2 + 0.17 * clay    (clay in %, w in % gravimetric)
Erodibility prior (one channel, in [0, 1]):
    E = sigmoid((u* - u*_t) / s) * bare_fraction
Worked example: clay 10 % -> w' = 1.84 %; soil moisture 5 % -> f_w = 1.91, so the dry
threshold is nearly doubled.
"""
from __future__ import annotations

import numpy as np

KAPPA = 0.4
Z_REF = 10.0
U_STAR_T_DRY = 0.25   # m/s, typical for loose fine sand / silt (Shao 2008 range 0.2-0.4)


def friction_velocity(u10: np.ndarray, z0: np.ndarray | float = 0.01) -> np.ndarray:
    """u* from 10 m wind speed (m/s) and roughness length z0 (m)."""
    z0 = np.maximum(np.asarray(z0, float), 1e-4)
    return KAPPA * np.asarray(u10, float) / np.log(Z_REF / z0)


def moisture_factor(soil_moisture_pct: np.ndarray, clay_pct: np.ndarray) -> np.ndarray:
    """Fecan et al. (1999) threshold multiplier (>= 1)."""
    w = np.asarray(soil_moisture_pct, float)
    c = np.asarray(clay_pct, float)
    w_prime = 0.0014 * c ** 2 + 0.17 * c
    excess = np.clip(w - w_prime, 0, None)
    return np.sqrt(1.0 + 1.21 * excess ** 0.68)


def threshold_friction_velocity(soil_moisture_pct, clay_pct, u_star_t_dry: float = U_STAR_T_DRY) -> np.ndarray:
    return u_star_t_dry * moisture_factor(soil_moisture_pct, clay_pct)


def erodibility_prior(u10, soil_moisture_pct, clay_pct, bare_fraction, z0=0.01, s: float = 0.05) -> np.ndarray:
    """E in [0,1]: probability-like score that the surface can emit dust under this wind."""
    u_star = friction_velocity(u10, z0)
    u_t = threshold_friction_velocity(soil_moisture_pct, clay_pct)
    sig = 1.0 / (1.0 + np.exp(-(u_star - u_t) / s))
    return (sig * np.clip(np.asarray(bare_fraction, float), 0, 1)).astype(np.float32)
