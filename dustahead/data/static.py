"""Static ground layers (do not change between samples): terrain, land cover, soil texture.

Six channels, each (ny, nx) float32, stored as data/static/<name>.npy:
  elevation   m          USGS 3DEP (via py3dep) ............ terrain channels drive outflow
  slope       degrees    derived from elevation            ... and shelter effects
  bare        0..1       fraction of bare / fallow ground  USDA Cropland Data Layer + NLCD
  landcover   class id   NLCD 2021 (normalised /100)
  sand        %          SoilGrids 2.0, 0-5 cm
  clay        %          SoilGrids 2.0, 0-5 cm  (also feeds the physics prior)

Any GeoTIFF can be warped onto the grid with warp_to_grid(); build_all() fetches the public
sources directly. Run `python scripts/02_build_static.py` once on the training machine.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..grid import Grid

NAMES = ("elevation", "slope", "bare", "landcover", "sand", "clay")

# NLCD classes counted as bare or fallow for the bare-fraction layer
NLCD_BARE = {31: 1.0, 52: 0.4, 71: 0.3, 82: 0.5, 81: 0.3}   # barren, shrub, grassland, cropland, pasture
# USDA CDL classes: fallow/idle (61), barren (131), and harvested row crops contribute bare ground
# Over AZ+NM the CDL is 62 % shrubland (152), 11 % evergreen forest (142), 7.5 % grass/pasture (176), 1 % barren (131);
# class 0 = no data (the Mexican part of the domain) and is filled with the shrubland value.
CDL_BARE = {152: 0.5, 176: 0.3, 131: 1.0, 61: 1.0, 1: 0.5, 2: 0.5, 4: 0.4, 24: 0.3, 36: 0.2, 37: 0.2, 42: 0.4, 44: 0.4,
            121: 0.2, 122: 0.15, 123: 0.1, 124: 0.05, 195: 0.1}
CDL_NODATA_FILL = 0.5


def warp_to_grid(tif: str | Path, grid: Grid, resampling: str = "average") -> np.ndarray:
    """Reproject a GeoTIFF onto the grid with rasterio (nearest / average / mode ...)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    dst = np.full(grid.shape, np.nan, np.float32)
    transform = rasterio.transform.from_bounds(grid.lon_min, grid.lat_min, grid.lon_max, grid.lat_max, grid.nx, grid.ny)
    with rasterio.open(tif) as src:
        reproject(rasterio.band(src, 1), dst, dst_transform=transform, dst_crs="EPSG:4326",
                  resampling=getattr(Resampling, resampling), dst_nodata=np.nan)
    return dst[::-1]   # rasterio rows run north->south; our row 0 is the south edge


def slope_from_elevation(elev: np.ndarray, grid: Grid) -> np.ndarray:
    dx_km, dy_km = grid.cell_km()
    gy, gx = np.gradient(np.nan_to_num(elev), dy_km * 1000, dx_km * 1000)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


def class_fraction(class_map: np.ndarray, weights: dict) -> np.ndarray:
    out = np.zeros_like(class_map, np.float32)
    for k, w in weights.items():
        out[class_map == k] = w
    return out


def fetch_elevation(grid: Grid, out_tif: Path) -> None:
    """USGS 3DEP DEM for the bbox at 1 km, saved as GeoTIFF (needs `pip install py3dep`)."""
    import os
    os.environ.setdefault("HYRIVER_CACHE_DISABLE", "true")   # py3dep's sqlite cache breaks on some mounts
    import py3dep
    dem = py3dep.get_dem((grid.lon_min, grid.lat_min, grid.lon_max, grid.lat_max), resolution=1000)   # 1 km is plenty for a 4 km grid
    dem.rio.to_raster(out_tif)


def fetch_soilgrids(prop: str, grid: Grid, out_tif: Path) -> None:
    """SoilGrids 2.0 WCS, 0-5 cm mean, 250 m. prop in {'sand','clay'}; values are g/kg (÷10 -> %)."""
    import requests
    url = "https://maps.isric.org/mapserv"
    params = {"map": f"/map/{prop}.map", "SERVICE": "WCS", "VERSION": "2.0.1", "REQUEST": "GetCoverage",
              "COVERAGEID": f"{prop}_0-5cm_mean", "FORMAT": "image/tiff",
              "SUBSET": [f"long({grid.lon_min},{grid.lon_max})", f"lat({grid.lat_min},{grid.lat_max})"],
              "SUBSETTINGCRS": "http://www.opengis.net/def/crs/EPSG/0/4326",
              "OUTPUTCRS": "http://www.opengis.net/def/crs/EPSG/0/4326"}
    r = requests.get(url, params=params, timeout=600)
    r.raise_for_status()
    out_tif.write_bytes(r.content)


def fetch_cdl(year: int, grid: Grid, out_tif: Path) -> None:
    """USDA NASS Cropland Data Layer for the bbox (CropScape GetCDLFile REST service)."""
    import requests
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    x0, y0 = tr.transform(grid.lon_min, grid.lat_min)
    x1, y1 = tr.transform(grid.lon_max, grid.lat_max)
    r = requests.get("https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile",
                     params=dict(year=year, bbox=f"{x0},{y0},{x1},{y1}"), timeout=600)
    r.raise_for_status()
    import re
    url = re.search(r"<returnURL>(.*?)</returnURL>", r.text).group(1)
    out_tif.write_bytes(requests.get(url, timeout=600).content)


def fetch_cdl_tiles(year: int, grid: Grid, out_dir: Path, nx: int = 6, ny: int = 3, budget_s: float = 150) -> list[Path]:
    """CropScape cannot serve the whole 12x6 degree box in one request, so fetch it as nx*ny tiles.
    Resumable: existing tiles are skipped; stops after budget_s seconds (call again to continue)."""
    import time
    import requests
    from pyproj import Transformer
    out_dir.mkdir(parents=True, exist_ok=True)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    lons = np.linspace(grid.lon_min, grid.lon_max, nx + 1)
    lats = np.linspace(grid.lat_min, grid.lat_max, ny + 1)
    t0 = time.time()
    done = []
    for a in range(nx):
        for b in range(ny):
            p = out_dir / f"cdl_{year}_{a}_{b}.tif"
            done.append(p)
            if p.exists() and p.stat().st_size > 1000:
                continue
            if time.time() - t0 > budget_s:
                return [q for q in done if q.exists()]
            xs, ys = [], []
            for lon in (lons[a], lons[a + 1]):
                for lat in (lats[b], lats[b + 1]):
                    x, y = tr.transform(lon, lat)
                    xs.append(x); ys.append(y)
            r = requests.get("https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile",
                             params=dict(year=year, bbox=f"{min(xs):.0f},{min(ys):.0f},{max(xs):.0f},{max(ys):.0f}"), timeout=600)
            r.raise_for_status()
            import re
            url = re.search(r"<returnURL>(.*?)</returnURL>", r.text)
            if not url:
                raise RuntimeError(r.text[:200])
            p.write_bytes(requests.get(url.group(1), timeout=600).content)
    return [q for q in done if q.exists()]


def mosaic_classes_to_grid(tifs: list[Path], grid: Grid, weights: dict) -> tuple[np.ndarray, np.ndarray]:
    """Warp class tiles onto the grid. Returns (class map by mode, weighted bare fraction by average)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    transform = rasterio.transform.from_bounds(grid.lon_min, grid.lat_min, grid.lon_max, grid.lat_max, grid.nx, grid.ny)
    cls = np.full(grid.shape, np.nan, np.float32)
    bare = np.full(grid.shape, np.nan, np.float32)
    for tif in tifs:
        with rasterio.open(tif) as src:
            # read decimated 4x (30 m -> 120 m): plenty for a 4 km grid and 16x less work
            f = 4
            h, w_ = src.height // f, src.width // f
            data = src.read(1, out_shape=(h, w_), resampling=Resampling.nearest)
            src_tr = src.transform * src.transform.scale(src.width / w_, src.height / h)
            nodata = src.nodata
            w = class_fraction(data.astype(int), weights)
            w[data == 0] = np.nan                       # CDL 0 = no data (outside the US)
            if nodata is not None:
                w[data == nodata] = np.nan
            c_dst = np.full(grid.shape, np.nan, np.float32)
            b_dst = np.full(grid.shape, np.nan, np.float32)
            reproject(data.astype(np.float32), c_dst, src_transform=src_tr, src_crs=src.crs, dst_transform=transform,
                      dst_crs="EPSG:4326", resampling=Resampling.nearest, src_nodata=nodata, dst_nodata=np.nan)
            reproject(w, b_dst, src_transform=src_tr, src_crs=src.crs, dst_transform=transform,
                      dst_crs="EPSG:4326", resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
            m = np.isfinite(c_dst)
            cls[m] = c_dst[m]
            m = np.isfinite(b_dst)
            bare[m] = b_dst[m]
    return cls[::-1], bare[::-1]


def build_all(grid: Grid, static_dir: str | Path, nlcd_tif: str | Path | None = None, cdl_year: int = 2024) -> dict[str, np.ndarray]:
    """Fetch, warp and save all six layers. NLCD must be downloaded manually from
    https://www.mrlc.gov/data (Land Cover, CONUS) and passed as nlcd_tif."""
    static_dir = Path(static_dir)
    static_dir.mkdir(parents=True, exist_ok=True)
    tifs = static_dir / "tif"
    tifs.mkdir(exist_ok=True)
    layers = {}

    fetch_elevation(grid, tifs / "elevation.tif")
    layers["elevation"] = warp_to_grid(tifs / "elevation.tif", grid, "average")
    layers["slope"] = slope_from_elevation(layers["elevation"], grid)

    for prop in ("sand", "clay"):
        fetch_soilgrids(prop, grid, tifs / f"{prop}.tif")
        layers[prop] = warp_to_grid(tifs / f"{prop}.tif", grid, "average") / 10.0

    tiles = fetch_cdl_tiles(cdl_year, grid, tifs / "cdl", budget_s=10 ** 6)
    cdl, bare = mosaic_classes_to_grid(tiles, grid, CDL_BARE)
    bare = np.nan_to_num(bare, nan=CDL_NODATA_FILL)
    if nlcd_tif:
        nlcd = warp_to_grid(nlcd_tif, grid, "mode")
        layers["landcover"] = np.nan_to_num(nlcd) / 100.0
        bare = np.maximum(bare, class_fraction(np.nan_to_num(nlcd).astype(int), NLCD_BARE))
    else:
        layers["landcover"] = np.nan_to_num(cdl) / 255.0
    layers["bare"] = bare

    for k, v in layers.items():
        np.save(static_dir / f"{k}.npy", np.nan_to_num(v).astype(np.float32))
    return layers


def load_static(static_dir: str | Path, grid: Grid) -> np.ndarray:
    """(6, ny, nx) float32 stack in NAMES order. Missing layers are zeros (with a warning)."""
    static_dir = Path(static_dir)
    out = np.zeros((len(NAMES),) + grid.shape, np.float32)
    for k, name in enumerate(NAMES):
        p = static_dir / f"{name}.npy"
        if p.exists():
            out[k] = np.load(p)
        else:
            print(f"[static] missing {p}; using zeros")
    return out
