"""Self-supervised PRE-TRAINING of the sky branch (the U-Net) by masked-patch reconstruction.

Why: every supervised run saturates after one or two passes over ~130-400 onset positives; the network never sees
enough examples to learn what sky structure precedes a dust storm. The inputs themselves are plentiful: every stored
sample carries 47 satellite / lightning / infrared-change channels, with no label needed. Masked reconstruction
(He et al. 2022, Masked Autoencoders) hides random patches of the input and asks the network to paint them back
from context, which forces it to learn cloud edges, outflow arcs, convective texture and their motion across the six
frames - exactly the features the dust head then needs. The pre-trained encoder + decoder weights initialise the
U-Net of a supervised run (train.init_unet_from); only the 1x1 output head is new.

Design choices:
  * The SAME U-Net class, channel set and crop size as the supervised model, so every weight transfers 1:1.
    The input is the dynamic channel set of the gated model (satellite, lightning, IR change, station, HRRR, and
    the forecast block if configured); the reconstruction target is the same stack.
  * Patch masking: square patches of `patch` cells, a fraction `ratio` of them zeroed in EVERY channel (so the
    network cannot copy a channel from its neighbour at the same pixel). Loss = mean squared error on the masked
    patches only (the visible ones are trivial).
  * Training data: the training split only (the 2025 validation half and 2026 are never touched, even without labels).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models import build_model
from .dataset import DustDataset
from .train import seed_all


def patch_mask(shape: tuple[int, int, int, int], patch: int, ratio: float, device) -> torch.Tensor:
    """(B, 1, H, W) float mask: 1 = hidden. Patches of `patch` x `patch` cells, a fraction `ratio` hidden."""
    B, _, H, W = shape
    gh, gw = -(-H // patch), -(-W // patch)
    m = (torch.rand(B, 1, gh, gw, device=device) < ratio).float()
    m = m.repeat_interleave(patch, 2).repeat_interleave(patch, 3)[:, :, :H, :W]
    return m


def pretrain(cfg: dict, run_dir: str | Path, epochs: int = 15, patch: int = 16, ratio: float = 0.5,
             device: str | None = None, limit: int | None = None) -> Path:
    T = cfg["train"]
    seed_all(T["seed"])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ds = DustDataset(cfg["paths"]["samples"], "train", tuple(T["crop"]), T["seed"], limit=limit, cfg=cfg)
    dl = DataLoader(ds, T["batch_size"], shuffle=True, num_workers=T["num_workers"], drop_last=True, pin_memory=device == "cuda")
    print(f"pretraining on {len(ds)} training samples, patch {patch}, mask ratio {ratio}, {epochs} epochs", flush=True)

    gm = build_model(cfg, "gated_unet")                 # same U-Net, same dynamic channel set as the supervised model
    unet = gm.unet
    cin = int(gm.dyn_idx.numel())
    unet.head = nn.Conv2d(unet.head.in_channels, cin, 1)   # reconstruct the input stack
    dyn_idx = gm.dyn_idx.clone()
    unet = unet.to(device)
    opt = torch.optim.AdamW(unet.parameters(), lr=T["lr"], weight_decay=T["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    use_amp = bool(T["amp"]) and device == "cuda" and amp_dtype == torch.bfloat16
    log = []
    for ep in range(epochs):
        t = time.time()
        unet.train()
        tot, n = 0.0, 0
        for b in dl:
            x = b["x"][:, dyn_idx].to(device)
            m = patch_mask(x.shape, patch, ratio, device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                pred = unet(x * (1 - m))
            loss = (((pred.float() - x) ** 2) * m).sum() / (m.sum() * x.shape[1] + 1e-6)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            opt.step()
            tot += loss.item() * x.shape[0]
            n += x.shape[0]
        sched.step()
        row = dict(epoch=ep, masked_mse=tot / max(n, 1), lr=sched.get_last_lr()[0], seconds=round(time.time() - t))
        log.append(row)
        print(f"ep {ep:03d} masked MSE {row['masked_mse']:.5f}  ({row['seconds']}s)", flush=True)
        sd = {k: v for k, v in unet.state_dict().items() if not k.startswith("head.")}
        torch.save(dict(unet=sd, cfg=cfg, cin=cin, patch=patch, ratio=ratio, epoch=ep, masked_mse=row["masked_mse"]), run_dir / "unet.pt")
        (run_dir / "log.json").write_text(json.dumps(log, indent=1))
    return run_dir


@torch.no_grad()
def load_pretrained_unet(model: nn.Module, path: str | Path, device: str = "cpu") -> int:
    """Copy every pre-trained U-Net tensor whose name and shape match into model.unet (the head stays new).
    Returns the number of tensors copied; raises if nothing matched (wrong channel set)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    own = model.unet.state_dict()
    n = 0
    for k, v in ck["unet"].items():
        if k in own and own[k].shape == v.shape:
            own[k].copy_(v)
            n += 1
    if n == 0:
        raise ValueError(f"no tensor of {path} matches model.unet (pre-trained cin {ck.get('cin')})")
    model.unet.load_state_dict(own)
    return n
