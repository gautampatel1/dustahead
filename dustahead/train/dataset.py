"""PyTorch dataset over the .npz samples written by data/sampler.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.sampler import load_sample

MAX_STATIONS = 64


class DustDataset(Dataset):
    def __init__(self, root: str | Path, split: str, crop: tuple[int, int] | None = None, seed: int = 0):
        self.files = sorted(Path(root, split).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no samples in {Path(root, split)}")
        self.crop = crop
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, k):
        s = load_sample(self.files[k])
        x = s["x"].astype(np.float32)
        y = s["y"].astype(np.float32)
        w = s["w"].astype(np.float32)
        ij = s["st_ij"].astype(np.int64)
        sy = s["st_y"].astype(np.int64)
        H, W = x.shape[1:]
        if self.crop and (self.crop[0] < H or self.crop[1] < W):
            ch, cw = min(self.crop[0], H), min(self.crop[1], W)
            i0 = self.rng.integers(0, H - ch + 1)
            j0 = self.rng.integers(0, W - cw + 1)
            x, y, w = x[:, i0:i0 + ch, j0:j0 + cw], y[:, i0:i0 + ch, j0:j0 + cw], w[:, i0:i0 + ch, j0:j0 + cw]
            ij = ij - np.array([i0, j0])
            keep = (ij[:, 0] >= 0) & (ij[:, 0] < ch) & (ij[:, 1] >= 0) & (ij[:, 1] < cw)
            sy = np.where(keep[:, None], sy, -1)
            ij = np.clip(ij, 0, [ch - 1, cw - 1])
        # pad stations to a fixed count so batches stack
        S = ij.shape[0]
        ij_p = np.zeros((MAX_STATIONS, 2), np.int64)
        sy_p = np.full((MAX_STATIONS, sy.shape[1]), -1, np.int64)
        ij_p[:S] = ij[:MAX_STATIONS]
        sy_p[:S] = sy[:MAX_STATIONS]
        return dict(x=torch.from_numpy(x), y=torch.from_numpy(y), w=torch.from_numpy(w),
                    st_ij=torch.from_numpy(ij_p), st_y=torch.from_numpy(sy_p),
                    genesis=torch.tensor(bool(s["genesis"])), idx=k)
