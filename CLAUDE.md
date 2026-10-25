# DustAhead — project memory for Claude Code

Short-range (15/30/60 min) dust-storm nowcasting for Interstate 10 in Arizona and New Mexico.
CS 7150 Deep Learning term project (Northeastern, Prof. Ilmi Yoon) that continues as a research
programme. Team: Gautam Patel (data), Sanidhya Chimurkar (labels + evaluation), Aayush Katoch (product).
Prototype deadline: **30 Nov 2026**. Target venue after that: workshop paper, then AIES journal.

## What this repo is
- `dustahead/grid.py` — 250×200 lat/lon grid (~4 km) over lon −115…−103, lat 30.5…36.5. Row 0 = south.
- `dustahead/events/` — NWS warnings from Iowa Environmental Mesonet (IEM) → event list; ASOS reports → station labels.
- `dustahead/data/` — GOES-18 ABI (6 bands × 6 frames), GLM lightning, HRRR surface fields, static ground
  layers, physics prior (friction velocity vs threshold), and `sampler.py` that stacks the **62 channels**.
- `dustahead/models/gated_unet.py` — **P(dust) = σ(z_τ) · σ(h)**: U-Net on sky channels × gate CNN on ground channels.
  Ablations `unet_all`, `unet_dust_only`; baseline `convlstm`; non-learned baselines in `baselines.py`.
- `dustahead/train/` — focal + FSS + station-BCE loss; AdamW, cosine, AMP, early stop on val station CSI.
- `dustahead/eval/` — CSI/POD/FAR/HSS, Brier/BSS/ECE, FSS, temperature scaling, event bootstrap, Holm.
- `dustahead/app/` — `predict.py` (live inference + I-10 road alerts), `dashboard.py` (Streamlit admin/user).
- `scripts/00…05` — the pipeline in order. `tests/` — unit + synthetic end-to-end.

## Hard rules (do not break)
1. **Labels are never hand-drawn.** Station label = vis ≤ 1 mi AND a dust code (DS/DU/BLDU…) AND inside a DS.W
   warning window/polygon or an NCEI Storm Events dust entry. Rain codes are recorded, never exclude (the Phoenix
   2025 haboob reads `-TSRA +DS`). ADP pixels are weak labels (weight 0.3). See `events/stations.py`, `DATA_SOURCES.md`.
2. **Splits by year:** train 2023–24, val 2025, **test 2026 is run once at the end** (`05_evaluate.py --split test`).
   Never tune on it. GOES-17 (2019–22) may be added to train only.
3. **Only data available at forecast time** goes into a sample (HRRR analysis ≤ t0, GOES frames ≤ t0).
4. **Primary metric:** genesis-stratified station CSI at +30 and +60 min (genesis = no dust within 50 km at t0).
   Report event-bootstrap CIs (B=1000) and Holm-corrected p-values vs every baseline.
5. Traffic (AADT) is for impact scoring only — never a model input.
6. No horizontal flips in augmentation (terrain and prevailing outflow are not symmetric).
7. Keep `configs/default.yaml` the single source of truth; scripts take `--config` for experiments.

## Commands
```
conda env create -f environment.yml && conda activate dustahead      # or: pip install -r requirements.txt
pytest -q                                                             # unit + synthetic end-to-end (~1 min CPU)
python scripts/00_check_sources.py                                    # PREFLIGHT: every source, package, GPU, speed
python scripts/00_make_synthetic.py --small && python scripts/04_train.py --epochs 2   # smoke run
python scripts/01_build_event_list.py                                 # events.csv  (needs internet)
python scripts/02_build_static.py                                     # static layers (once)
python scripts/03_build_samples.py --limit 3                          # first real samples
python scripts/04_train.py --model gated_unet
python scripts/05_evaluate.py --ckpt runs/latest/best.pt --split val
streamlit run dustahead/app/dashboard.py -- --ckpt runs/latest/best.pt --demo
```

## State on 2026-10-24 (Claude, from the linked PC)
- Preflight passes for every data source. Satellite = GOES-East (16 -> 19 on 2025-04-07); see DATA_SOURCES.md.
- `data/labels/events.csv`: 152 dust events + 268 null days (2023-26). `data/static/*.npy`: all six layers built.
- First real samples in `data/samples/val/` (Phoenix 2025-08-25): labels 23 dust-positive reports, 8 severe.
- Full build running natively on Windows since 2026-10-24 02:08 via bootstrap_windows.bat (5.7 s/sample, ~17 h). Then training (04_train.py) in the same .venv.
- Claude's Linux shell on this PC is limited to 3-minute commands: use it for checks and fixes, never for the bulk build.

## Known gaps / TODO (Phase 1)
- `hrrr.dcape_proxy` is a proxy (HRRR has no DCAPE). Say so in the paper.
- `static.build_all` fetches 3DEP, SoilGrids, CDL; NLCD must be downloaded manually. Verify each endpoint on first run.
- `app/predict.I10` waypoints are coarse; replace with OSM geometry + FHWA HPMS AADT in Phase 2.
- GOES file timing: CONUS scans every 5 min; `nearest_file` tolerates 6 min. Flag any sample with a gap.
- GOES/GLM are read remotely (fsspec + h5netcdf) and only regridded frames are cached; HRRR via Herbie needs
  eccodes/cfgrib (on Windows install them from conda-forge).
- Data-build order: 00_check_sources → 01 → 02 → 03 --limit 2 (inspect) → 03 full (--workers 4–6, overnight).
- `make_real_samples.py` (Dust RGB PNGs for the slides) lives in `scripts/`; untested end to end.

## Style
- Python 3.11, numpy/pandas/xarray/torch. Type hints, docstrings that explain the physics, not just the code.
- Every abbreviation spelled out once per module docstring.
- Commit messages: imperative, one line + why. Never commit `data/` or `runs/` (see .gitignore).
