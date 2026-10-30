"""Smoke test: synthetic data -> train 2 epochs -> evaluate -> predict + road alerts."""
import copy
import sys
from pathlib import Path

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
    # ensemble of two checkpoints = element-wise max; genesis map from the stored station reports
    ens = DustAhead([run_dir / "best.pt", run_dir / "best.pt"], cfg, device="cpu")
    out2 = ens.predict_array(s["x"].astype(np.float32), st_ij=s["st_ij"], st_now=s["st_now"])
    assert np.allclose(out2["probs"], out["probs"]) and len(out2["members"]) == 1 and out2["genesis"].shape == (64, 80)
    assert all({"segment", "lead_min", "p_max", "genesis", "level"} <= set(a) for a in out2["alerts"])


def test_finetune_ema_fold_and_policy(tmp_path):
    """v9-style fine-tune (init_from + EMA) on the swapped 2025 fold, the station-persistence policy in eval and
    in the product, and the offline ensemble/policy script on the saved predictions."""
    import json
    import subprocess
    cfg = copy.deepcopy(load_config())
    cfg["paths"].update(samples=str(tmp_path / "samples"), runs=str(tmp_path / "runs"), static=str(tmp_path / "static"))
    cfg["train"].update(batch_size=2, num_workers=0, crop=[48, 64], amp=False, patience=3)
    cfg["model"].update(base_channels=8, depth=2)
    cfg["eval"]["bootstrap"] = 10
    make_synthetic_dataset(cfg["paths"]["samples"], Grid.from_config(cfg), n=10, small=True)
    base = train(cfg, "gated_unet", tmp_path / "runs" / "base", max_epochs=1, device="cpu")

    ft = copy.deepcopy(cfg)
    ft["train"].update(init_from=str(base / "best.pt"), lr=1e-4, ema_decay=0.9)
    ft["splits"]["val_fold"] = 1
    run_dir = train(ft, "gated_unet", tmp_path / "runs" / "ft", max_epochs=2, device="cpu")
    ck = torch.load(run_dir / "best.pt", weights_only=False)
    assert ck["ema"] is True and ck["init_from"] == str(base / "best.pt")
    # fold 1 validates the complement of fold 0 (when index.csv exists); the two never overlap
    from dustahead.train.dataset import split_files
    f0, _ = split_files(cfg["paths"]["samples"], "val", cfg)
    f1, _ = split_files(cfg["paths"]["samples"], "val", ft)
    assert not (set(map(str, f0)) & set(map(str, f1))) or f0 == f1   # equal only when there is no index.csv

    rep = evaluate(run_dir / "best.pt", ft, "val", ("station_persistence",), device="cpu", calibrate_on="val")
    assert "model+persistence" in rep["models"]
    p = np.load(run_dir / "eval_val_preds.npz", allow_pickle=True)
    l15 = ft["time"]["leads_min"].index(15)
    assert np.all(p["p_model+persistence"][:, l15] >= p["p_model"][:, l15])
    # the product applies the same floor on the map
    ft["domain"].update(nx=80, ny=64)
    ft["product"]["persistence_leads"] = [15]
    eng = DustAhead(run_dir / "best.pt", ft, device="cpu")
    s = load_sample(sorted((tmp_path / "samples" / "val").glob("*.npz"))[0])
    s["st_now"][0] = 1
    out = eng.predict_array(s["x"].astype(np.float32), st_ij=s["st_ij"], st_now=s["st_now"])
    i, j = s["st_ij"][0]
    assert out["probs"][l15, i, j] == 1.0
    # offline ensemble + policy script runs on the saved predictions
    r = subprocess.run([sys.executable, "scripts/11_ensemble_eval.py", "--runs", str(run_dir), "--split", "val",
                        "--out", str(tmp_path / "ens.json")], capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]))
    assert r.returncode == 0, r.stderr
    ens = json.loads((tmp_path / "ens.json").read_text())
    assert "ensemble+persistence@15" in ens["models"] and "station_persistence" in ens["models"]
