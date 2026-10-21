import numpy as np
import pytest
import torch

from dustahead import load_config
from dustahead.data.sampler import N_CH
from dustahead.models import GatedUNet, build_model
from dustahead.models.baselines import advected, hrrr_rule, persistence
from dustahead.train.losses import DustLoss, focal_loss, fss, station_bce

CFG = load_config()


@pytest.mark.parametrize("name", ["gated_unet", "unet_all", "unet_dust_only", "convlstm"])
def test_models_forward(name):
    m = build_model(CFG, name)
    x = torch.randn(2, N_CH, 48, 56)       # not a multiple of 16: tests padding
    p, z, h = m(x, return_parts=True)
    assert p.shape == (2, 3, 48, 56) and (p >= 0).all() and (p <= 1).all()


def test_gate_starts_open_and_factorises():
    m = GatedUNet(gate_bias=2.2)
    x = torch.randn(1, N_CH, 32, 32)
    p, z, h = m(x, return_parts=True)
    g = torch.sigmoid(h)
    assert abs(g.mean().item() - 0.90) < 0.01            # sigmoid(2.2) = 0.900
    assert torch.allclose(p, torch.sigmoid(z) * g)
    # the gate sees only ground channels: perturbing satellite channels must not change h
    x2 = x.clone(); x2[:, :47] += 5
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
    x = np.random.rand(N_CH, 20, 30).astype(np.float32)
    for f in (persistence, advected, hrrr_rule):
        out = f(x)
        assert out.shape == (3, 20, 30)
