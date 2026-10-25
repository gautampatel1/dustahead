"""Index every sample once: positives per lead at stations and pixels -> data/samples/index.csv.
Used by the dataset for positive-aware sampling and for a validation subset that keeps every positive.
    python scripts/06_index_samples.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dustahead import load_config


def main():
    cfg = load_config()
    root = Path(cfg["paths"]["samples"])
    rows = []
    for split in ("train", "val", "test"):
        for f in sorted((root / split).glob("*.npz")):
            with np.load(f) as z:
                st_y, y, w = z["st_y"], z["y"], z["w"]
                rows.append(dict(split=split, file=f.name, event=f.stem.rsplit("_", 1)[0], kind=f.stem.split("_")[0],
                                 genesis=bool(z["genesis"]), n_st=int((st_y >= 0).any(1).sum()),
                                 pos15=int((st_y[:, 0] == 1).sum()), pos30=int((st_y[:, 1] == 1).sum()), pos60=int((st_y[:, 2] == 1).sum()),
                                 pix_pos=int(y.sum()), pix_known=int((w > 0).sum())))
    df = pd.DataFrame(rows)
    df.to_csv(root / "index.csv", index=False)
    df["any_pos"] = (df[["pos15", "pos30", "pos60"]].sum(1) > 0)
    print(df.groupby("split").agg(samples=("file", "size"), with_station_pos=("any_pos", "sum"),
                                  st_pos_total=("pos30", "sum"), pix_pos_mean=("pix_pos", "mean"), genesis=("genesis", "mean")).to_string())
    print("station reports per sample (mean):", df.n_st.mean().round(1))


if __name__ == "__main__":
    main()
