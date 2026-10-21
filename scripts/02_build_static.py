"""Step 2 (once): build the six static ground layers into data/static/*.npy.

    python scripts/02_build_static.py [--nlcd path/to/nlcd_2021_land_cover.tif]

Needs: pip install rasterio py3dep. NLCD is optional (download CONUS land cover from https://www.mrlc.gov/data).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dustahead import load_config
from dustahead.data.static import build_all
from dustahead.grid import Grid

ap = argparse.ArgumentParser()
ap.add_argument("--nlcd", default=None)
ap.add_argument("--cdl-year", type=int, default=2024)
a = ap.parse_args()
cfg = load_config()
layers = build_all(Grid.from_config(cfg), cfg["paths"]["static"], a.nlcd, a.cdl_year)
for k, v in layers.items():
    print(f"{k:10s} min {v.min():8.2f} max {v.max():8.2f}")
