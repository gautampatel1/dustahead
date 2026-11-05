import numpy as np
import pytest
import torch

from dustahead import load_config
from dustahead.data.sampler import N_CH, N_CH_IN
from dustahead.models import GatedUNet, build_model
from dustahead.models.baselines import advected, hrrr_rule, persistence
from dustahead.train.losses import DustLoss, focal_loss, fss, station_bce

CFG = load_config()


@pytest.mark.parametrize("name", ["gated_unet", "unet_all", "unet_dust_only", "convlstm"])
def test_models_forward(name):
    m = build_model(CFG, name)
    x = torch.randn(2, N_CH_IN, 48, 56)       # not a multiple of 16: tests padding
    p, z, h = m(x, return_parts=True)
    assert p.shape == (2, 3, 48, 56) and (p >= 0).all() and (p <= 1).all()


def test_gate_starts_open_and_factorises():
    m = GatedUNet(gate_bias=2.2)
    x = torch.randn(1, N_CH_IN, 32, 32)
    p, z, h = m(x, return_parts=True)
    g = torch.sigmoid(h)
    assert abs(g.mean().item() - 0.90) < 0.01            # sigmoid(2.2) = 0.900
    assert torch.allclose(p, torch.sigmoid(z) * g)
    # the gate sees only ground channels: perturbing satellite channels must not change h
    x2 = x.clone(); x2[:, :47] += 5; x2[:, 62] += 5
    assert torch.allclose(m(x2, return_parts=True)[2], h)


def test_losses_and_grad():
    torch.manual_seed(0)
    B, L, H, W = 2, 3, 24, 24
    p = torch.rand(B, L, H, W, requires_grad=True)
    y = (torch.rand(B, L, H, W) > 0.9).float()
    w = torch.ones_like(y)
    assert fss(y, y, w).item() == pytest.approx(1.0)
    assert focal_loss(y, y, w).item() < focal_loss(1 - y, y, w).item()
    st_ij = torch.tensor([[[1, 1], [5, 5]], [[2, 2], [0, 0]]])
    st_y = torch.tensor([[[1, 1, -1], [0, 0, 0]], [[1, -1, -1], [-1, -1, -1]]])
    batch = dict(y=y, w=w, st_ij=st_ij, st_y=st_y)
    total, parts = DustLoss(CFG)(p, batch)
    total.backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()
    assert set(parts) == {"focal", "fss", "station"}
    assert station_bce(torch.full((1, 3, 4, 4), 0.99), torch.tensor([[[0, 0]]]), torch.tensor([[[1, 1, 1]]])).item() < 0.02


def test_baselines_shapes():
    x = np.random.rand(N_CH_IN, 20, 30).astype(np.float32)
    for f in (persistence, advected, hrrr_rule):
        out = f(x)
        assert out.shape == (3, 20, 30)


def test_genesis_weight_and_config_overlay():
    from dustahead import load_config
    from dustahead.data.sampler import genesis_per_station
    # station 0 is dusty now -> station 1 (3 cells away) is not genesis, station 2 (far) is
    gen = genesis_per_station(np.array([[10, 10], [12, 12], [60, 60]]), np.array([1, 0, -1]), radius_cells=12)
    assert gen.tolist() == [False, False, True]
    p = torch.full((1, 3, 70, 70), 0.3)
    ij = torch.tensor([[[10, 10], [12, 12], [60, 60]]]); y = torch.tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
    g = torch.tensor([[False, False, True]])
    base = station_bce(p, ij, y, 10.0)
    assert station_bce(p, ij, y, 10.0, g, 0.0).item() == pytest.approx(base.item())
    assert station_bce(p, ij, y, 10.0, g, 3.0).item() > base.item()
    cfg = load_config("configs/genesis_w3.yaml")
    assert cfg["loss"]["station_genesis_weight"] == 3.0 and cfg["model"]["base_channels"] == 32   # overlay keeps defaults
    assert DustLoss(cfg).gen_w == 3.0


def test_warm_start_with_forecast_channels_reproduces_base_model():
    """load_partial: a v8-style checkpoint (63 inputs) widened to 69 (HRRR forecast block) must compute the same
    output while the new input weights are zero, whatever the forecast channels contain."""
    import copy
    from dustahead import load_config
    from dustahead.models import build_model
    from dustahead.train.train import load_partial
    from dustahead.data.sampler import N_CH_IN, N_CH_IN_FCST
    cfg = copy.deepcopy(load_config())
    cfg["model"].update(base_channels=8, depth=2, gate_channels=["static", "physics"])
    base = build_model(cfg, "gated_unet").eval()
    cfg2 = copy.deepcopy(cfg); cfg2["channels"]["hrrr_fcst"] = 6
    wide = build_model(cfg2, "gated_unet").eval()
    n = load_partial(wide, base.state_dict())
    assert n == 1                                   # only the first U-Net convolution is widened
    x = torch.randn(2, N_CH_IN_FCST, 32, 32)
    with torch.no_grad():
        assert torch.allclose(base(x[:, :N_CH_IN]), wide(x), atol=1e-6)


def test_pretrain_and_transfer(tmp_path):
    """Masked-reconstruction pre-training runs on synthetic data and its U-Net weights load into the gated model."""
    import copy
    from dustahead import load_config
    from dustahead.data.synthetic import make_synthetic_dataset
    from dustahead.grid import Grid
    from dustahead.models import build_model
    from dustahead.train.pretrain import pretrain, load_pretrained_unet
    cfg = copy.deepcopy(load_config())
    cfg["paths"].update(samples=str(tmp_path / "samples"), runs=str(tmp_path / "runs"))
    cfg["train"].update(batch_size=2, num_workers=0, crop=[48, 64], amp=False)
    cfg["model"].update(base_channels=8, depth=2, gate_channels=["static", "physics"])
    make_synthetic_dataset(cfg["paths"]["samples"], Grid.from_config(cfg), n=8, small=True)
    out = pretrain(cfg, tmp_path / "pre", epochs=1, patch=8, ratio=0.5, device="cpu")
    ck = torch.load(out / "unet.pt", weights_only=False)
    assert not any(k.startswith("head.") for k in ck["unet"]) and ck["masked_mse"] > 0
    m = build_model(cfg, "gated_unet")
    before = m.unet.enc[0].net[0].weight.clone()
    n = load_pretrained_unet(m, out / "unet.pt")
    assert n > 10 and not torch.equal(before, m.unet.enc[0].net[0].weight)


def test_radar_block_and_aux_head():
    """channels.radar adds three sky-branch inputs at the fixed radar slot; loss.w_aux adds a dense auxiliary head
    whose logits reach the loss; a checkpoint without either is still a channel prefix for a warm start."""
    import copy
    from dustahead import load_config
    from dustahead.models import build_model
    from dustahead.train.losses import DustLoss
    from dustahead.data.sampler import CH, N_CH_IN_FULL, N_FCST, N_RADAR
    assert CH["radar"].stop == N_CH_IN_FULL and CH["radar"].start == CH["hrrr_fcst"].stop
    cfg = copy.deepcopy(load_config())
    cfg["model"].update(base_channels=8, depth=2, gate_channels=["static", "physics"])
    cfg["channels"].update(hrrr_fcst=6, radar=3); cfg["loss"]["w_aux"] = 1.0
    m = build_model(cfg, "gated_unet")
    assert int(m.dyn_idx.max()) == N_CH_IN_FULL - 1 and m.unet.aux_head is not None
    x = torch.randn(2, N_CH_IN_FULL, 32, 32)
    p = m(x)
    assert m.unet.last_aux.shape == p.shape
    batch = dict(y=torch.zeros(2, 3, 32, 32), w=torch.ones(2, 3, 32, 32), st_ij=torch.zeros(2, 4, 2, dtype=torch.long),
                 st_y=torch.zeros(2, 4, 3, dtype=torch.long))
    total, parts = DustLoss(cfg)(p, batch, m.unet.last_aux)
    assert "aux" in parts and torch.isfinite(total)
    # radar without the forecast block still sits at the same slot (forecast slot filled with zeros by input_stack)
    cfg2 = copy.deepcopy(cfg); cfg2["channels"]["hrrr_fcst"] = 0; cfg2["loss"]["w_aux"] = 0
    m2 = build_model(cfg2, "gated_unet")
    assert int(m2.dyn_idx.max()) == N_CH_IN_FULL - 1 and m2.unet.aux_head is None and (N_CH_IN_FULL - 63) == N_FCST + N_RADAR
