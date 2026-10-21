"""Loss = w_focal * focal(p, y; w) + w_fss * (1 - FSS_n(p, y)) + w_station * BCE(p at stations).

Focal loss (Lin et al. 2017) handles the ~1:500 pixel imbalance:  FL = -alpha_t (1-p_t)^gamma log p_t
FSS loss (Fractions Skill Score, Roberts & Lean 2008) rewards being right at the neighbourhood scale:
    FSS_n = 1 - sum (P_n - O_n)^2 / (sum P_n^2 + sum O_n^2), P_n/O_n = n x n box means of p / y.
Station BCE is the only term on true ground observations; the pixel terms use painted targets.
All operate on probabilities p (product of two sigmoids has no single logit).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-6


def focal_loss(p, y, w, alpha=0.25, gamma=2.0):
    p = p.clamp(EPS, 1 - EPS)
    y = y.float()
    pt = torch.where(y > 0.5, p, 1 - p)
    at = torch.where(y > 0.5, torch.full_like(p, alpha), torch.full_like(p, 1 - alpha))
    fl = -at * (1 - pt) ** gamma * torch.log(pt)
    return (fl * w).sum() / w.sum().clamp_min(1.0)


def fss(p, y, w, n=5):
    """Differentiable FSS over the known (w>0) region, averaged over batch and leads."""
    k = torch.ones(1, 1, n, n, device=p.device) / (n * n)
    B, L, H, W = p.shape
    pn = F.conv2d((p * (w > 0)).reshape(B * L, 1, H, W), k, padding=n // 2)
    on = F.conv2d((y.float() * (w > 0)).reshape(B * L, 1, H, W), k, padding=n // 2)
    num = ((pn - on) ** 2).sum((1, 2, 3))
    den = (pn ** 2).sum((1, 2, 3)) + (on ** 2).sum((1, 2, 3))
    return (1 - num / den.clamp_min(EPS)).mean()


def station_bce(p, st_ij, st_y):
    """p: (B, L, H, W); st_ij: (B, S, 2); st_y: (B, S, L) with -1 = no report (masked)."""
    B, L, H, W = p.shape
    i = st_ij[..., 0].long().clamp(0, H - 1)
    j = st_ij[..., 1].long().clamp(0, W - 1)
    b = torch.arange(B, device=p.device)[:, None].expand_as(i)
    ps = p[b, :, i, j]                     # (B, S, L)
    y = st_y.float()
    m = (y >= 0).float()
    ps = ps.clamp(EPS, 1 - EPS)
    bce = -(y.clamp_min(0) * torch.log(ps) + (1 - y.clamp_min(0)) * torch.log(1 - ps))
    return (bce * m).sum() / m.sum().clamp_min(1.0)


class DustLoss(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        L = cfg["loss"]
        self.a, self.g, self.n = L["focal_alpha"], L["focal_gamma"], L["fss_window"]
        self.wf, self.ws, self.wst = L["w_focal"], L["w_fss"], L["w_station"]

    def forward(self, p, batch):
        y, w = batch["y"], batch["w"]
        lf = focal_loss(p, y, w, self.a, self.g)
        ls = 1 - fss(p, y, w, self.n)
        lst = station_bce(p, batch["st_ij"], batch["st_y"])
        total = self.wf * lf + self.ws * ls + self.wst * lst
        return total, dict(focal=lf.item(), fss=ls.item(), station=lst.item())
