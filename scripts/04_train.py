"""Step 4: train a model.

    python scripts/04_train.py --model gated_unet            # our model
    python scripts/04_train.py --model unet_all              # ablation: no gate
    python scripts/04_train.py --model unet_dust_only        # ablation: satellite only
    python scripts/04_train.py --model convlstm              # baseline
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dustahead import load_config
from dustahead.train.train import train

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=None)
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--config", default=None)
ap.add_argument("--run-dir", default=None)
a = ap.parse_args()
cfg = load_config(a.config)
run_dir = train(cfg, a.model, a.run_dir, a.epochs)
latest = Path(cfg["paths"]["runs"]) / "latest"
if latest.is_symlink() or latest.exists():
    latest.unlink()
try:
    latest.symlink_to(run_dir.resolve(), target_is_directory=True)
except OSError:
    (Path(cfg["paths"]["runs"]) / "latest.txt").write_text(str(run_dir))
print("best checkpoint:", run_dir / "best.pt")
