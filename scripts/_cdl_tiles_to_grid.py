"""Resumable helper: warp each CDL tile onto the grid and save a partial .npz; then combine. Called by 02_build_static."""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dustahead import load_config
from dustahead.grid import Grid
from dustahead.data import static as S

budget = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9
cfg = load_config(); g = Grid.from_config(cfg); d = Path(cfg["paths"]["static"])
t0 = time.time(); todo = 0
for tif in sorted((d / "tif/cdl").glob("cdl_*_*_*.tif")):
    out = tif.with_suffix(".grid2.npz")
    if out.exists():
        continue
    if time.time() - t0 > budget:
        todo += 1; continue
    cls, bare = S.mosaic_classes_to_grid([tif], g, S.CDL_BARE)
    np.savez_compressed(out, cls=cls, bare=bare); print("warped", tif.name, round(time.time() - t0), "s", flush=True)
if todo:
    print(f"{todo} tiles left; run again"); sys.exit(2)
cls = np.full(g.shape, np.nan, np.float32); bare = np.full(g.shape, np.nan, np.float32)
for npz in sorted((d / "tif/cdl").glob("*.grid2.npz")):
    with np.load(npz) as z:
        m = np.isfinite(z["cls"]); cls[m] = z["cls"][m]
        m = np.isfinite(z["bare"]); bare[m] = z["bare"][m]
print("class nan%", np.isnan(cls).mean().round(3), "bare nan%", np.isnan(bare).mean().round(3), "bare mean", np.nanmean(bare).round(3))
print("Phoenix", bare[g.to_ij(-112.01, 33.43)], "Casa Grande farmland", bare[g.to_ij(-111.75, 32.9)], "Flagstaff forest", bare[g.to_ij(-111.65, 35.2)])
np.save(d / "bare.npy", np.nan_to_num(bare, nan=S.CDL_NODATA_FILL).astype(np.float32)); np.save(d / "landcover.npy", (np.nan_to_num(cls) / 255.0).astype(np.float32))
print("saved bare.npy, landcover.npy")
