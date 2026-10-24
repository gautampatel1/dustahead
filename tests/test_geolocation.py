"""ABI fixed-grid geolocation against a synthetic GOES-West CONUS (PACUS) sector header.
Real values from GOES-18 ABI-L2-MCMIPC files: lon_0 = -137.0, h = 35786023 m, 2 km pixel = 56 urad,
x in about [-0.030052, 0.109948] rad for the West (PACUS) sector (2500 cols; East CONUS is [-0.101332, 0.038612]), y in [0.128212, 0.044268] rad (1500 rows, north first)."""
import numpy as np
import xarray as xr

from dustahead import load_config
from dustahead.data.goes import AbiGeolocator
from dustahead.grid import Grid


def fake_pacus():
    nx, ny = 2500, 1500
    x = -0.030052 + np.arange(nx) * 0.000056
    y = 0.128212 - np.arange(ny) * 0.000056
    proj = xr.DataArray(0, attrs=dict(perspective_point_height=35786023.0, longitude_of_projection_origin=-137.0,
                                      sweep_angle_axis="x", semi_major_axis=6378137.0, semi_minor_axis=6356752.31414))
    cmi = xr.DataArray(np.zeros((ny, nx), np.float32), dims=("y", "x"))
    return xr.Dataset(dict(goes_imager_projection=proj, CMI_C13=cmi), coords=dict(x=("x", x), y=("y", y)))


def test_domain_inside_sector_and_orientation():
    g = Grid.from_config(load_config())
    ds = fake_pacus()
    geo = AbiGeolocator(ds, g)
    assert geo.valid.mean() > 0.99                       # whole AZ/NM domain is inside PACUS
    # north row of our grid must map to a SMALLER ABI row index (ABI rows run north -> south)
    assert geo.i[-1, :].mean() < geo.i[0, :].mean()
    # east column of our grid maps to a larger ABI column index
    assert geo.j[:, -1].mean() > geo.j[:, 0].mean()
    # the slab covers only a fraction of the full image
    assert (geo.i1 - geo.i0) < 700 and (geo.j1 - geo.j0) < 900
    # sampling a ramp image reproduces the ramp
    ramp = np.broadcast_to(np.arange(2500, dtype=np.float32), (1500, 2500))
    out = geo(ramp)
    assert np.all(np.diff(out[100, :]) >= 0)
