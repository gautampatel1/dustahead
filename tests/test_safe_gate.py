"""v15: a gate that cannot freeze shut (log-space loss, gate floor) and a sky branch with its own outflow target."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dustahead import load_config
from dustahead.data.synthetic import make_synthetic_dataset
from dustahead.events.stations import station_targets, station_wind_targets
from dustahead.grid import Grid
from dustahead.models.gated_unet import build_model, log_probs
from dustahead.train.losses import DustLoss, focal_loss, focal_loss_log, station_bce, station_bce_log
from dustahead.train.train import load_checkpoint, train

CFG = load_config()


def test_log_probs_match_the_product_and_stay_finite():
    z = torch.linspace(-6, 6, 13, dtype=torch.float64)[:, None]
    h = torch.linspace(-6, 6, 13, dtype=torch.float64)[None, :]
    for floor in (0.0, 0.02):
        g = floor + (1 - floor) * torch.sigmoid(h)
        p = torch.sigmoid(z) * g
        lp, l1p = log_probs(z, h, floor)
        assert torch.allclose(lp, p.log()) and torch.allclose(l1p, (1 - p).log())
    lp, l1p = log_probs(torch.tensor([60.0, -60.0]), torch.tensor([60.0, -80.0]))
    assert torch.isfinite(lp).all() and torch.isfinite(l1p).all()
    assert lp[1] < -100                        # P ~ 1e-61: the log is exact, not clamped at log(1e-6)


def _station_case(h_val):
    """One station reporting dust at all three leads, sky logit 0, gate logit h_val."""
    z = torch.zeros(1, 3, 4, 4, requires_grad=True)
    h = torch.full((1, 1, 4, 4), float(h_val), requires_grad=True)
    st_ij = torch.tensor([[[1, 2]]])
    st_y = torch.ones(1, 1, 3, dtype=torch.long)
    return z, h, st_ij, st_y


def test_shut_gate_gets_zero_gradient_in_probability_space_but_not_in_log_space():
    z, h, st_ij, st_y = _station_case(-30.0)            # sigmoid(-30) ~ 1e-13: the v4 monsoon gate
    station_bce(torch.sigmoid(z) * torch.sigmoid(h), st_ij, st_y, 10.0).backward()
    assert z.grad.abs().sum() == 0 and h.grad.abs().sum() == 0          # frozen: the 1e-6 clamp eats the gradient

    z, h, st_ij, st_y = _station_case(-30.0)
    station_bce_log(*log_probs(z, h), st_ij, st_y, 10.0).backward()
    assert h.grad[0, 0, 1, 2] < -5        # a dust report under a shut gate pushes it OPEN (d/dh = -pw (1 - sigma(h)))
    assert z.grad[0, :, 1, 2].lt(0).all()  # ... and the sky branch still learns, whatever the gate says


def test_log_space_terms_equal_the_clamped_ones_where_nothing_is_clamped():
    torch.manual_seed(0)
    z, h = torch.randn(2, 3, 8, 8), torch.randn(2, 1, 8, 8)
    p = torch.sigmoid(z) * torch.sigmoid(h)
    y = (torch.rand(2, 3, 8, 8) < 0.3).float()
    w = torch.rand(2, 3, 8, 8)
    lp, l1p = log_probs(z, h)
    assert torch.allclose(focal_loss(p, y, w, 0.75, 2.0), focal_loss_log(lp, l1p, y, w, 0.75, 2.0), atol=1e-5)
    st_ij = torch.tensor([[[1, 1], [5, 6]], [[0, 7], [3, 3]]])
    st_y = torch.tensor([[[1, 0, -1], [0, 0, 1]], [[-1, -1, -1], [1, 1, 0]]])
    assert torch.allclose(station_bce(p, st_ij, st_y, 10.0), station_bce_log(lp, l1p, st_ij, st_y, 10.0), atol=1e-5)


def test_gate_floor_bounds_the_forecast_from_below():
    cfg = copy.deepcopy(CFG)
    cfg["model"].update(base_channels=8, depth=2, gate_floor=0.05)
    m = build_model(cfg, "gated_unet").eval()
    with torch.no_grad():
        m.gate.net[-1].bias.fill_(-50.0)                # gate shut everywhere
        p, z, h = m(torch.randn(1, 63, 32, 32), return_parts=True)
    assert torch.allclose(p, 0.05 * torch.sigmoid(z))   # the ground damps, never vetoes
    assert torch.allclose(p.log(), log_probs(z, h, 0.05)[0], atol=1e-5)


def test_wind_targets_from_asos_reports():
    t0 = pd.Timestamp("2025-08-26T02:00Z")
    m = pd.Timedelta(minutes=1)
    obs = pd.DataFrame(dict(
        station=["PHX", "PHX", "PHX", "TUS", "TUS", "ELP"],
        valid=[t0 + 15 * m, t0 + 31 * m, t0 + 62 * m, t0 + 14 * m, t0 + 40 * m, t0 + 15 * m],
        sknt=[12.0, 22.0, np.nan, 8.0, 40.0, np.nan], gust=[np.nan, 34.0, 45.0, np.nan, np.nan, np.nan]))
    w = station_wind_targets(obs, ["PHX", "TUS", "ELP", "YUM"], t0, [15, 30, 60], CFG)
    assert w[0].tolist() == [0, 1, 1]       # 12 kt; gust 34 >= 30; gust only (sustained missing) still counts
    assert w[1].tolist() == [0, -1, -1]     # the 40 kt report is 10 min from +30: outside the ±5 min tolerance
    assert w[2].tolist() == [-1, -1, -1]    # no wind reading at all
    assert w[3].tolist() == [-1, -1, -1]    # no report
    no_wind = obs.drop(columns=["sknt", "gust"])
    assert (station_wind_targets(no_wind, ["PHX"], t0, [15, 30, 60], CFG) == -1).all()   # extended tables: unknown


def test_wind_script_rebuilds_the_sampler_station_order():
    spec = importlib.util.spec_from_file_location("add21", Path(__file__).parents[1] / "scripts" / "21_add_wind_targets.py")
    add21 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(add21)
    t0 = pd.Timestamp("2025-08-26T02:00Z")
    stations = pd.DataFrame(dict(sid=["TUS", "A39", "PHX", "ELP"], lon=0.0, lat=0.0, i=[1, 2, 3, 4], j=[5, 6, 7, 8]))
    labels = pd.DataFrame(dict(station=["PHX", "TUS", "XXX", "A39", "PHX"], valid=t0, label=0))
    st = station_targets(labels, stations, t0, [15, 30, 60]).merge(
        stations[["sid", "lon", "lat", "i", "j"]], left_on="station", right_on="sid")
    assert add21.station_order(labels, stations) == st.station.tolist()


def test_v15b_trains_end_to_end(tmp_path):
    cfg = copy.deepcopy(load_config(Path(__file__).parents[1] / "configs" / "safe_gate_v15b.yaml"))
    cfg["paths"].update(samples=str(tmp_path / "samples"), runs=str(tmp_path / "runs"), static=str(tmp_path / "static"))
    cfg["train"].update(batch_size=2, num_workers=0, crop=[48, 64], amp=False, patience=3)
    cfg["model"].update(base_channels=8, depth=2)
    make_synthetic_dataset(cfg["paths"]["samples"], Grid.from_config(cfg), n=10, small=True)
    from dustahead.train.dataset import DustDataset
    ds = DustDataset(cfg["paths"]["samples"], "train", (48, 64), cfg=cfg)
    b = {k: (v[None] if torch.is_tensor(v) else v) for k, v in ds[0].items()}
    model = build_model(cfg, "gated_unet")
    p, z, h = model(b["x"], return_parts=True)
    _, parts = DustLoss(cfg)(p, b, z=z, h=h, gate_floor=model.gate_floor)
    assert "sky" in parts and np.isfinite(parts["sky"])
    run_dir = train(cfg, "gated_unet", tmp_path / "runs" / "v15b", max_epochs=1, device="cpu")
    m, _ = load_checkpoint(run_dir / "best.pt")
    assert m.gate_floor == 0.02                          # the floor travels with the checkpoint's config
