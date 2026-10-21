"""Gated U-Net: dust probability = (what the sky is doing) x (whether the ground can emit).

    P_tau(r) = sigmoid( z_tau(r) ) * sigmoid( h(r) )

  z_tau : U-Net on the DYNAMIC channels (satellite frames, lightning, IR change) -> one logit map
          per lead time tau in {15, 30, 60} min. "Is a storm outflow about to reach r?"
  h     : small dilated CNN on the GROUND channels (HRRR wind/moisture, terrain, soil, physics
          prior) -> one shared gate logit. "Can this ground be lofted now?"

How this differs from Attention U-Net (Oktay et al. 2018): there, attention gates sit on the
SKIP CONNECTIONS and are learned from the same image. Here the gate is a separate branch on
different physical inputs and multiplies the OUTPUT, so it is a factorised model
P(dust) = P(outflow) * P(erodible) that can be inspected and ablated. The gate bias is
initialised to +2.2 (sigmoid 0.90) so training starts as a plain U-Net and the gate only
closes where the ground data says it should.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.sampler import CH

DYNAMIC = ("satellite", "lightning", "ir_change")
GROUND = ("hrrr", "static", "physics")


def channel_index(groups) -> list[int]:
    idx = []
    for g in groups:
        s = CH[g]
        idx.extend(range(s.start, s.stop))
    return idx


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU())

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Plain encoder-decoder with skip connections. Output: (B, n_out, H, W) logits."""

    def __init__(self, cin: int, n_out: int = 3, base: int = 32, depth: int = 4):
        super().__init__()
        chs = [base * 2 ** k for k in range(depth + 1)]
        self.enc = nn.ModuleList([ConvBlock(cin if k == 0 else chs[k - 1], chs[k]) for k in range(depth + 1)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[k + 1], chs[k], 2, stride=2) for k in range(depth)])
        self.dec = nn.ModuleList([ConvBlock(chs[k] * 2, chs[k]) for k in range(depth)])
        self.head = nn.Conv2d(chs[0], n_out, 1)
        self.depth = depth

    def forward(self, x):
        H, W = x.shape[-2:]
        m = 2 ** self.depth
        pad = (0, (m - W % m) % m, 0, (m - H % m) % m)
        x = F.pad(x, pad)
        skips = []
        for k, blk in enumerate(self.enc):
            x = blk(x)
            if k < self.depth:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for k in reversed(range(self.depth)):
            x = self.up[k](x)
            x = self.dec[k](torch.cat([x, skips[k]], 1))
        return self.head(x)[..., :H, :W]


class GroundGate(nn.Module):
    """Dilated CNN on ground-state channels -> one gate logit map (B, 1, H, W)."""

    def __init__(self, cin: int, width: int = 32, bias_init: float = 2.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=4, dilation=4), nn.GELU(),
            nn.Conv2d(width, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, bias_init)

    def forward(self, g):
        return self.net(g)


class GatedUNet(nn.Module):
    def __init__(self, n_leads: int = 3, base: int = 32, depth: int = 4, gate_bias: float = 2.2,
                 dynamic=DYNAMIC, ground=GROUND):
        super().__init__()
        self.register_buffer("dyn_idx", torch.tensor(channel_index(dynamic)))
        self.register_buffer("gnd_idx", torch.tensor(channel_index(ground)))
        self.unet = UNet(len(self.dyn_idx), n_leads, base, depth)
        self.gate = GroundGate(len(self.gnd_idx), base, gate_bias)

    def forward(self, x, return_parts: bool = False):
        z = self.unet(x[:, self.dyn_idx])          # (B, L, H, W) outflow logits
        h = self.gate(x[:, self.gnd_idx])          # (B, 1, H, W) gate logit
        p = torch.sigmoid(z) * torch.sigmoid(h)
        return (p, z, h) if return_parts else p


class PlainUNet(nn.Module):
    """Ablation: everything into one U-Net, no gate (channels chosen by `groups`)."""

    def __init__(self, n_leads=3, base=32, depth=4, groups=DYNAMIC + GROUND):
        super().__init__()
        self.register_buffer("idx", torch.tensor(channel_index(groups)))
        self.unet = UNet(len(self.idx), n_leads, base, depth)

    def forward(self, x, return_parts=False):
        z = self.unet(x[:, self.idx])
        p = torch.sigmoid(z)
        return (p, z, None) if return_parts else p


class ConvLSTMCell(nn.Module):
    def __init__(self, cin, hid):
        super().__init__()
        self.conv = nn.Conv2d(cin + hid, 4 * hid, 3, padding=1)
        self.hid = hid

    def forward(self, x, state):
        h, c = state
        i, f, o, g = self.conv(torch.cat([x, h], 1)).chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTMNet(nn.Module):
    """Baseline: 6 satellite+lightning frames through a ConvLSTM, ground channels concatenated at the end."""

    def __init__(self, n_leads=3, hid=32):
        super().__init__()
        self.cell = ConvLSTMCell(7, hid)            # 6 bands + 1 lightning per frame
        self.register_buffer("gnd_idx", torch.tensor(channel_index(GROUND)))
        self.head = nn.Sequential(nn.Conv2d(hid + len(self.gnd_idx), hid, 3, padding=1), nn.GELU(), nn.Conv2d(hid, n_leads, 1))
        self.hid = hid

    def forward(self, x, return_parts=False):
        B, _, H, W = x.shape
        h = x.new_zeros(B, self.hid, H, W)
        c = x.new_zeros(B, self.hid, H, W)
        for t in range(6):
            frame = torch.cat([x[:, 6 * t:6 * t + 6], x[:, 36 + t:37 + t]], 1)
            h, c = self.cell(frame, (h, c))
        z = self.head(torch.cat([h, x[:, self.gnd_idx]], 1))
        p = torch.sigmoid(z)
        return (p, z, None) if return_parts else p


def build_model(cfg: dict, name: str | None = None) -> nn.Module:
    m = cfg["model"]
    name = name or m["name"]
    n_leads = len(cfg["time"]["leads_min"])
    if name == "gated_unet":
        return GatedUNet(n_leads, m["base_channels"], m["depth"], m["gate_bias_init"])
    if name == "unet_all":
        return PlainUNet(n_leads, m["base_channels"], m["depth"])
    if name == "unet_dust_only":
        return PlainUNet(n_leads, m["base_channels"], m["depth"], groups=("satellite",))
    if name == "convlstm":
        return ConvLSTMNet(n_leads, m["base_channels"])
    raise ValueError(name)
