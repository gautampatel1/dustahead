"""Step 5: evaluate a checkpoint against the baselines.

    python scripts/05_evaluate.py --ckpt runs/<run>/best.pt --split val
    python scripts/05_evaluate.py --ckpt runs/<run>/best.pt --split test     # ONCE, at the end
    python scripts/05_evaluate.py --ckpt runs/<run>/best.pt --config configs/fold1.yaml   # a run trained on the other fold
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dustahead import load_config
from dustahead.eval.evaluate import run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--baselines", nargs="*", default=["station_persistence", "persistence", "advected", "hrrr_rule"])
    ap.add_argument("--config", default=None, help="experiment overlay; must match the split the run was trained with")
    a = ap.parse_args()
    run(a.ckpt, load_config(a.config), a.split, tuple(a.baselines))


if __name__ == "__main__":
    main()
