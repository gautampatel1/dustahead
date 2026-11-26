"""Step 15: self-supervised pre-training of the sky branch (masked-patch reconstruction; train/pretrain.py).

    python scripts/15_pretrain.py [--config configs/gate_static.yaml] [--epochs 15] [--patch 16] [--ratio 0.5] --run-dir runs/pretrain_mae_v1

Then train the supervised model from it:  04_train.py --config configs/pretrain_v12.yaml --run-dir runs/gated_unet_v12
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dustahead.keep_awake import keep_awake  # noqa: E402
keep_awake(True)
from dustahead import load_config
from dustahead.train.pretrain import pretrain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--run-dir", default="runs/pretrain_mae_v1")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    out = pretrain(load_config(a.config), a.run_dir, a.epochs, a.patch, a.ratio, limit=a.limit)
    print("pre-trained U-Net:", out / "unet.pt")


if __name__ == "__main__":
    main()
