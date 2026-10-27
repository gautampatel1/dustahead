"""PyTorch dataset over the .npz samples written by data/sampler.py.

Positives are rare (about 1 station label in 1,000 is dust), so two things matter:
  * which files form train/val: by year (default) or, with splits.val_from_2025, half of the 2025
    EVENTS join the training set (event-level split; the 2026 test year is never touched);
  * how batches are drawn: `sampler_weights()` up-weights samples that contain a positive so roughly
    half of every batch has dust somewhere (WeightedRandomSampler in train.py).
Both use data/samples/index.csv from scripts/06_index_samples.py.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..data.sampler import genesis_per_station, load_sample, paint_station_now

MAX_STATIONS = 96


def _event_hash(event_id: str) -> float:
    return int(hashlib.md5(event_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def split_files(root: str | Path, split: str, cfg: dict | None = None) -> tuple[list[Path], pd.DataFrame | None]:
    """Files for a split, plus their index rows (None if index.csv is missing)."""
    root = Path(root)
    idx_path = root / "index.csv"
    idx = pd.read_csv(idx_path) if idx_path.exists() else None
    frac = (cfg or {}).get("splits", {}).get("val_from_2025")
    if idx is None or not frac or split == "test":
        files = sorted((root / split).glob("*.npz"))
        rows = idx[idx.split == split].set_index("file").loc[[f.name for f in files]].reset_index() if idx is not None and files else None
        return files, rows
    idx = idx.copy()
    idx["h"] = idx.event.map(_event_hash)
    if split == "train":
        sel = idx[(idx.split == "train") | ((idx.split == "val") & (idx.h >= frac))]
    else:
        sel = idx[(idx.split == "val") & (idx.h < frac)]
    files = [root / r.split / r.file for r in sel.itertuples()]
    return files, sel.reset_index(drop=True)


class DustDataset(Dataset):
    def __init__(self, root: str | Path, split: str, crop: tuple[int, int] | None = None, seed: int = 0,
                 limit: int | None = None, cfg: dict | None = None):
        self.files, self.rows = split_files(root, split, cfg)
        if not self.files:
            raise FileNotFoundError(f"no samples for split {split} under {root}")
        if limit and len(self.files) > limit:
            if self.rows is not None:       # keep every sample with a station positive, thin the rest evenly
                pos = (self.rows[["pos15", "pos30", "pos60"]].sum(axis=1) > 0).values
                keep = np.flatnonzero(pos).tolist()
                neg = np.flatnonzero(~pos)
                n_neg = max(limit - len(keep), 0)
                if n_neg and len(neg):
                    keep += neg[np.linspace(0, len(neg) - 1, min(n_neg, len(neg))).astype(int)].tolist()
                keep = sorted(set(keep))
            else:
                step = len(self.files) / limit
                keep = [int(k * step) for k in range(limit)]
            self.files = [self.files[k] for k in keep]
            self.rows = self.rows.iloc[keep].reset_index(drop=True) if self.rows is not None else None
        self.crop = crop
        self.rng = np.random.default_rng(seed)
        # station-dropout augmentation (training only): hide a fraction of the current station reports so the model
        # cannot rely on the station channel alone and keeps using satellite/HRRR cues (robustness to missing reports)
        self.station_dropout = float((cfg or {}).get("train", {}).get("station_dropout", 0.0)) if split == "train" else 0.0

    def sampler_weights(self, pos_weight: float = 8.0) -> torch.Tensor | None:
        """Per-sample weights for WeightedRandomSampler: samples with any positive count pos_weight x."""
        if self.rows is None:
            return None
        st_pos = (self.rows[["pos15", "pos30", "pos60"]].sum(axis=1) > 0).values
        pix_pos = (self.rows.pix_pos > 0).values
        w = np.where(st_pos, pos_weight, np.where(pix_pos, 1.0 + 0.25 * (pos_weight - 1), 1.0))
        return torch.tensor(w, dtype=torch.double)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, k):
        s = load_sample(self.files[k])
        x = s["x"].astype(np.float32)
        st_now = s["st_now"].copy()
        if self.station_dropout > 0:
            st_now[self.rng.random(len(st_now)) < self.station_dropout] = -1       # -1 = no report
        x = np.concatenate([x, paint_station_now(s["st_ij"], st_now, x.shape[1:])[None]], 0)   # channel 62
        y = s["y"].astype(np.float32)
        w = s["w"].astype(np.float32)
        ij = s["st_ij"].astype(np.int64)
        sy = s["st_y"].astype(np.int64)
        gen = genesis_per_station(ij, s["st_now"])          # per-station genesis flag (no dust within 50 km at t0)
        H, W = x.shape[1:]
        if self.crop and (self.crop[0] < H or self.crop[1] < W):
            ch, cw = min(self.crop[0], H), min(self.crop[1], W)
            # crop around a positive station when there is one (keeps the rare signal inside the window)
            pos_idx = np.flatnonzero((sy == 1).any(1))
            gen_idx = pos_idx[gen[pos_idx]]
            if len(pos_idx) and self.rng.random() < 0.8:
                # prefer a genesis positive (the rare onset signal) when the sample has one
                pi, pj = ij[self.rng.choice(gen_idx if len(gen_idx) and self.rng.random() < 0.5 else pos_idx)]
                i0 = int(np.clip(pi - self.rng.integers(0, ch), 0, H - ch))
                j0 = int(np.clip(pj - self.rng.integers(0, cw), 0, W - cw))
            else:
                i0 = int(self.rng.integers(0, H - ch + 1))
                j0 = int(self.rng.integers(0, W - cw + 1))
            x, y, w = x[:, i0:i0 + ch, j0:j0 + cw], y[:, i0:i0 + ch, j0:j0 + cw], w[:, i0:i0 + ch, j0:j0 + cw]
            ij = ij - np.array([i0, j0])
            keep = (ij[:, 0] >= 0) & (ij[:, 0] < ch) & (ij[:, 1] >= 0) & (ij[:, 1] < cw)
            sy = np.where(keep[:, None], sy, -1)
            ij = np.clip(ij, 0, [ch - 1, cw - 1])
        S = ij.shape[0]
        ij_p = np.zeros((MAX_STATIONS, 2), np.int64)
        sy_p = np.full((MAX_STATIONS, sy.shape[1]), -1, np.int64)
        gen_p = np.zeros(MAX_STATIONS, bool)
        ij_p[:S] = ij[:MAX_STATIONS]
        sy_p[:S] = sy[:MAX_STATIONS]
        gen_p[:S] = gen[:MAX_STATIONS]
        return dict(x=torch.from_numpy(x), y=torch.from_numpy(y), w=torch.from_numpy(w),
                    st_ij=torch.from_numpy(ij_p), st_y=torch.from_numpy(sy_p), st_gen=torch.from_numpy(gen_p),
                    genesis=torch.tensor(bool(s["genesis"])), idx=k)
