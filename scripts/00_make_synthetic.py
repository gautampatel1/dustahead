"""Step 0 (optional): synthetic samples so every later step can be tested with no downloads.

    python scripts/00_make_synthetic.py --n 24 --small
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dustahead import load_config
from dustahead.data.synthetic import make_synthetic_dataset
from dustahead.grid import Grid

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--small", action="store_true", help="64x80 grid instead of 200x250")
ap.add_argument("--out", default=None)
a = ap.parse_args()
cfg = load_config()
make_synthetic_dataset(a.out or cfg["paths"]["samples"], Grid.from_config(cfg), a.n, small=a.small)
print("synthetic samples written to", a.out or cfg["paths"]["samples"])
