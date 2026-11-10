"""Loss = w_focal * focal(p, y; w) + w_fss * (1 - FSS_n(p, y)) + w_station * BCE(p at stations).

Focal loss (Lin et al. 2017) handles the ~1:500 pixel imbalance:  FL = -alpha_t (1-p_t)^gamma log p_t
FSS loss (Fractions Skill Score, Roberts & Lean 2008) rewards being right at the neighbourhood scale:
    FSS_n = 1 - sum (P_n - O_n)^2 / (sum P_n^2 + sum O_n^2), P_n/O_n = n x n box means of p / y.
Station BCE is the only term on true ground observations; the pixel terms use painted targets.
By default focal and station BCE operate on probabilities p clamped to [1e-6, 1 - 1e-6] (the product of two
sigmoids has no single logit). With loss.log_space they take (log P, log(1 - P)) from models.log_probs instead:
no clamp, so a closed gate keeps its gradient (the clamp returns exactly zero once P < 1e-6). FSS stays on p.
Sky term (loss.w_sky, gated model only): BCE of the sky branch ALONE, sigmoid(z), against "outflow reached the
station" (st_wind: wind or gust >= label.outflow_kt at t0 + lead). It gives sigmoid(z) its own meaning,
P(outflow arrives), instead of letting the gate do all the discriminating, and every gusty afternoon is a positive.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..models.gated_unet import log_probs

EPS = 1e-6


def focal_loss(p, y, w, alpha=0.25, gamma=2.0):
    p = p.clamp(EPS, 1 - EPS)
    y = y.float()
    pt = torch.where(y > 0.5, p, 1 - p)
    at = torch.where(y > 0.5, torch.full_like(p, alpha), torch.full_like(p, 1 - alpha))
    fl = -at * (1 - pt) ** gamma * torch.log(pt)
    return (fl * w).sum() / w.sum().clamp_min(1.0)


def focal_loss_log(lp, l1p, y, w, alpha=0.25, gamma=2.0):
    """focal_loss from log P and log(1 - P): same value where p is not clamped, finite gradient everywhere."""
    pos = y.float() > 0.5
    lpt = torch.where(pos, lp, l1p)
    at = torch.where(pos, torch.full_like(lp, alpha), torch.full_like(lp, 1 - alpha))
    fl = -at * (-torch.expm1(lpt)) ** gamma * lpt          # (1 - p_t)^gamma, with 1 - p_t = -expm1(log p_t)
    return (fl * w).sum() / w.sum().clamp_min(1.0)


def _at_stations(t, st_ij):
    """(B, L, H, W) map -> (B, S, L) values at the station cells."""
    B, L, H, W = t.shape
    i = st_ij[..., 0].long().clamp(0, H - 1)
    j = st_ij[..., 1].long().clamp(0, W - 1)
    b = torch.arange(B, device=t.device)[:, None].expand_as(i)
    return t[b, :, i, j]


def station_bce_log(lp, l1p, st_ij, st_y, pos_weight: float = 1.0, st_gen=None, genesis_weight: float = 0.0):
    """station_bce from log P and log(1 - P) maps (same masking and weights)."""
    ls, l1s = _at_stations(lp, st_ij), _at_stations(l1p, st_ij)
    y = st_y.float()
    m = (y >= 0).float()
    yp = y.clamp_min(0)
    pw = torch.full_like(ls, float(pos_weight))
    if st_gen is not None and genesis_weight:
        pw = pw * (1.0 + genesis_weight * st_gen.float()[..., None])
    bce = -(pw * yp * ls + (1 - yp) * l1s)
    return (bce * m).sum() / m.sum().clamp_min(1.0)


def sky_bce(z, st_ij, st_wind, pos_weight: float = 1.0):
    """BCE of sigmoid(z) at the stations against the outflow target st_wind (B, S, L), -1 = unknown (masked)."""
    zs = _at_stations(z, st_ij)
    y = st_wind.float()
    m = (y >= 0).float()
    bce = F.binary_cross_entropy_with_logits(zs, y.clamp_min(0), reduction="none",
                                             pos_weight=torch.tensor(float(pos_weight), device=z.device))
    return (bce * m).sum() / m.sum().clamp_min(1.0)


def fss(p, y, w, n=5):
    """Differentiable FSS over the known (w>0) region, averaged over batch and leads."""
    k = torch.ones(1, 1, n, n, device=p.device) / (n * n)
    B, L, H, W = p.shape
    pn = F.conv2d((p * (w > 0)).reshape(B * L, 1, H, W), k, padding=n // 2)
    on = F.conv2d((y.float() * (w > 0)).reshape(B * L, 1, H, W), k, padding=n // 2)
    num = ((pn - on) ** 2).sum((1, 2, 3))
    den = (pn ** 2).sum((1, 2, 3)) + (on ** 2).sum((1, 2, 3))
    return (1 - num / den.clamp_min(EPS)).mean()


def station_bce(p, st_ij, st_y, pos_weight: float = 1.0, st_gen=None, genesis_weight: float = 0.0):
    """p: (B, L, H, W); st_ij: (B, S, 2); st_y: (B, S, L) with -1 = no report (masked). pos_weight scales positives;
    positives at genesis stations (st_gen: (B, S) bool) are scaled by a further (1 + genesis_weight)."""
    B, L, H, W = p.shape
    i = st_ij[..., 0].long().clamp(0, H - 1)
    j = st_ij[..., 1].long().clamp(0, W - 1)
    b = torch.arange(B, device=p.device)[:, None].expand_as(i)
    ps = p[b, :, i, j]                     # (B, S, L)
    y = st_y.float()
    m = (y >= 0).float()
    ps = ps.clamp(EPS, 1 - EPS)
    yp = y.clamp_min(0)
    pw = torch.full_like(ps, float(pos_weight))
    if st_gen is not None and genesis_weight:
        pw = pw * (1.0 + genesis_weight * st_gen.float()[..., None])
    bce = -(pw * yp * torch.log(ps) + (1 - yp) * torch.log(1 - ps))
    return (bce * m).sum() / m.sum().clamp_min(1.0)


class DustLoss(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        L = cfg["loss"]
        self.a, self.g, self.n = L["focal_alpha"], L["focal_gamma"], L["fss_window"]
        self.wf, self.ws, self.wst = L["w_focal"], L["w_fss"], L["w_station"]
        self.pos_w = L.get("station_pos_weight", 1.0)
        self.gen_w = L.get("station_genesis_weight", 0.0)
        # auxiliary dense head (model.unet.last_aux): focal + FSS against the same painted targets but with EVERY
        # labelled pixel at full weight (ADP satellite pixels included, w > 0), no station term.
        self.w_aux = float(L.get("w_aux", 0.0))
        self.log_space = bool(L.get("log_space", False))
        self.w_sky = float(L.get("w_sky", 0.0))
        self.sky_pos_w = float(L.get("sky_pos_weight", 1.0))

    def forward(self, p, batch, aux_logits=None, z=None, h=None, gate_floor: float = 0.0):
        """p: model probabilities. z, h: the logits from model(x, return_parts=True) (h None without a gate); the
        log-space and sky terms need them and fall back to the probability path when they are not given."""
        y, w = batch["y"], batch["w"]
        if self.log_space and z is not None:
            lp, l1p = log_probs(z.float(), None if h is None else h.float(), gate_floor)
            p = lp.exp()
            lf = focal_loss_log(lp, l1p, y, w, self.a, self.g)
            lst = station_bce_log(lp, l1p, batch["st_ij"], batch["st_y"], self.pos_w, batch.get("st_gen"), self.gen_w)
        else:
            lf = focal_loss(p, y, w, self.a, self.g)
            lst = station_bce(p, batch["st_ij"], batch["st_y"], self.pos_w, batch.get("st_gen"), self.gen_w)
        ls = 1 - fss(p, y, w, self.n)
        total = self.wf * lf + self.ws * ls + self.wst * lst
        parts = dict(focal=lf.item(), fss=ls.item(), station=lst.item())
        if self.w_sky and z is not None and h is not None and "st_wind" in batch:
            lsk = sky_bce(z.float(), batch["st_ij"], batch["st_wind"], self.sky_pos_w)
            total = total + self.w_sky * lsk
            parts["sky"] = lsk.item()
        if self.w_aux and aux_logits is not None:
            pa = torch.sigmoid(aux_logits.float())
            wa = (w > 0).float()
            la = focal_loss(pa, y, wa, self.a, self.g) + (1 - fss(pa, y, wa, self.n))
            total = total + self.w_aux * la
            parts["aux"] = la.item()
        return total, parts
