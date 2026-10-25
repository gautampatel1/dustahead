"""Training loop: AdamW + cosine schedule, mixed precision, early stopping on validation station CSI."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..eval.metrics import station_scores
from ..models import build_model
from .dataset import DustDataset
from .losses import DustLoss


def seed_all(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    ps, ys = [], []
    for b in loader:
        x = b["x"].to(device)
        p = model(x)
        B, L, H, W = p.shape
        i, j = b["st_ij"][..., 0].to(device), b["st_ij"][..., 1].to(device)
        bb = torch.arange(B, device=device)[:, None].expand_as(i)
        ps.append(p[bb, :, i, j].cpu().numpy())
        ys.append(b["st_y"].numpy())
    p, y = np.concatenate(ps).reshape(-1, ps[0].shape[-1]), np.concatenate(ys).reshape(-1, ys[0].shape[-1])
    out = {}
    for l in range(p.shape[1]):
        r = station_scores(p[:, l], y[:, l], threshold)
        # model selection uses the best threshold (rare positives sit below 0.5 early on); 05_evaluate reports both
        best = max(((station_scores(p[:, l], y[:, l], t)["csi"], t) for t in np.arange(0.05, 0.96, 0.05)),
                   key=lambda z: (-1 if np.isnan(z[0]) else z[0]))
        r["csi_best"], r["thr_best"] = float(best[0]), float(best[1])
        out[f"lead{l}"] = r
    return out


def train(cfg: dict, model_name: str | None = None, run_dir: str | Path | None = None, max_epochs: int | None = None,
          device: str | None = None) -> Path:
    T = cfg["train"]
    seed_all(T["seed"])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    name = model_name or cfg["model"]["name"]
    run_dir = Path(run_dir or Path(cfg["paths"]["runs"]) / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)

    tr = DustDataset(cfg["paths"]["samples"], "train", tuple(T["crop"]), T["seed"], cfg=cfg)
    va = DustDataset(cfg["paths"]["samples"], "val", limit=T.get("val_max_samples"), cfg=cfg)   # per-epoch subset; 05_evaluate uses all
    sw = tr.sampler_weights(T.get("pos_oversample", 8.0))
    sampler = None if sw is None else WeightedRandomSampler(sw, num_samples=len(tr), replacement=True,
                                                            generator=torch.Generator().manual_seed(T["seed"]))
    print(f"train {len(tr)} samples ({'positive-aware sampling' if sampler is not None else 'uniform'}), val {len(va)} samples", flush=True)
    dl_tr = DataLoader(tr, T["batch_size"], shuffle=sampler is None, sampler=sampler, num_workers=T["num_workers"],
                       drop_last=False, pin_memory=device == "cuda")
    dl_va = DataLoader(va, T["batch_size"], num_workers=T["num_workers"])

    model = build_model(cfg, name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=T["lr"], weight_decay=T["weight_decay"])
    epochs = max_epochs or T["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    # bf16 has fp32's exponent range: no overflow -> no inf/NaN from the fp16 path, and no GradScaler needed
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    use_scaler = T["amp"] and device == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    crit = DustLoss(cfg)
    n_skipped = 0

    best, bad, log = -1.0, 0, []
    for ep in range(epochs):
        t_ep = time.time()
        model.train()
        tot, n = 0.0, 0
        for b in dl_tr:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=amp_dtype, enabled=(T["amp"] and device == "cuda")):
                p = model(b["x"])
            loss, parts = crit(p.float(), b)
            if not torch.isfinite(loss):
                n_skipped += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gn):
                n_skipped += 1
                opt.zero_grad(set_to_none=True)
                continue
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * b["x"].shape[0]
            n += b["x"].shape[0]
        sched.step()
        if n_skipped:
            print(f"  skipped {n_skipped} non-finite batches so far", flush=True)
        if not all(torch.isfinite(q).all() for q in model.parameters()):
            print("  non-finite weights: restoring best checkpoint and halving the learning rate", flush=True)
            if (run_dir / "best.pt").exists():
                model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=False)["model"])
            for g in opt.param_groups:
                g["lr"] *= 0.5
        val = evaluate(model, dl_va, device, cfg["eval"]["threshold"])
        score = float(np.nanmean([v["csi_best"] for v in val.values()]))
        row = dict(epoch=ep, train_loss=tot / max(n, 1), val=val, val_csi=score, lr=sched.get_last_lr()[0])
        log.append(row)
        print(f"ep {ep:03d} loss {row['train_loss']:.4f} val bestCSI {score:.3f} "
              + " ".join(f"{k}:{v['csi_best']:.2f}@{v['thr_best']:.2f}" for k, v in val.items())
              + f" | CSI@0.5 {np.nanmean([v['csi'] for v in val.values()]):.2f} Brier {np.nanmean([v['brier'] for v in val.values()]):.4f}"
              + f"  ({time.time() - t_ep:.0f}s)", flush=True)
        if score > best:
            best, bad = score, 0
            torch.save(dict(model=model.state_dict(), cfg=cfg, name=name, epoch=ep, val=val), run_dir / "best.pt")
        else:
            bad += 1
            if bad >= T["patience"]:
                print("early stop")
                break
        (run_dir / "log.json").write_text(json.dumps(log, indent=1, default=float))
    return run_dir


def load_checkpoint(path: str | Path, device: str = "cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["cfg"], ck["name"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck
