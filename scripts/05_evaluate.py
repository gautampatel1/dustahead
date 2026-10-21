"""Step 5: evaluate a checkpoint against the baselines.

    python scripts/05_evaluate.py --ckpt runs/<run>/best.pt --split val
    python scripts/05_evaluate.py --ckpt runs/<run>/best.pt --split test     # ONCE, at the end
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dustahead import load_config
from dustahead.eval.evaluate import run

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--split", default="val")
ap.add_argument("--baselines", nargs="*", default=["persistence", "advected", "hrrr_rule"])
a = ap.parse_args()
run(a.ckpt, load_config(), a.split, tuple(a.baselines))
