"""Smoke test: synthetic data -> train 2 epochs -> evaluate -> predict + road alerts."""
import copy

import numpy as np
import torch

from dustahead import load_config
from dustahead.app.predict import DustAhead
from dustahead.data.sampler import load_sample
from dustahead.data.synthetic import make_synthetic_dataset
from dustahead.eval.evaluate import run as evaluate
from dustahead.grid import Grid
from dustahead.train.train import train


def test_end_to_end(tmp_path):
    cfg = copy.deepcopy(load_config())
    cfg["paths"].update(samples=str(tmp_path / "samples"), runs=str(tmp_path / "runs"), static=str(tmp_path / "static"))
    cfg["train"].update(batch_size=2, num_workers=0, crop=[48, 64], amp=False, patience=3)
    cfg["model"].update(base_channels=8, depth=2)
    cfg["eval"]["bootstrap"] = 20
    make_synthetic_dataset(cfg["paths"]["samples"], Grid.from_config(cfg), n=10, small=True)

    run_dir = train(cfg, "gated_unet", tmp_path / "runs" / "t", max_epochs=2, device="cpu")
    ck = torch.load(run_dir / "best.pt", weights_only=False)
    assert ck["epoch"] in (0, 1)

    rep = evaluate(run_dir / "best.pt", cfg, "test", ("persistence", "hrrr_rule"), device="cpu")
    assert "model" in rep["models"] and "genesis_30" in rep["models"]["model"]
    # in-sample Platt branch (what 05_evaluate --split val runs): must report raw and calibrated model
    rep2 = evaluate(run_dir / "best.pt", cfg, "val", ("persistence",), device="cpu", calibrate_on="val")
    assert "model_raw" in rep2["models"] and rep2["models"]["model_raw"]["fss_pixels"] == rep2["models"]["model"]["fss_pixels"]

    # product interface on one stored sample
    cfg["domain"].update(nx=80, ny=64)
    eng = DustAhead(run_dir / "best.pt", cfg, device="cpu")
    s = load_sample(sorted((tmp_path / "samples" / "test").glob("*.npz"))[0])
    out = eng.predict_array(s["x"].astype(np.float32))   # 62-channel input: the station channel is appended as zeros
    assert out["probs"].shape == (3, 64, 80) and out["gate"].shape == (64, 80)
    assert isinstance(out["alerts"], list)
