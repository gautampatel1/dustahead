# DustAhead — running the data build and training on the Windows PC

Everything below runs in a normal Windows terminal (PowerShell) in `C:\DustAhead`, NOT inside the remote
Linux shell: that shell is limited to 3-minute commands and cannot see the GPU. The logs are read
from this folder and fixes anything that fails.

## Setup + data build in one go: double-click `bootstrap_windows.bat`
It finds Python 3.12, creates `.venv`, installs CUDA 12.8 PyTorch + all packages (pip-only; eccodes ships
binary wheels now, so no conda needed), runs the tests and the preflight, then runs the full sample build.
Everything is logged to `logs\bootstrap.log` and `logs\build_samples.log`. Resumable: if it stops, double-click again.
(launched this way on 2026-10-24 02:08; GPU = RTX 5060 Ti 16 GB, torch 2.11+cu128.)
`scripts\setup_windows.ps1` is the conda alternative if pip wheels ever fail.

## Current state (28 Oct 2026)
Data set complete (train 13,730 / val 5,514 / test 3,029 samples, 2018-2026). Models v1-v8 trained; product =
max(unet_all_v4, gated_unet_v4, gated_unet_v8). Results in `results\val_2025_eventsplit\README.md`.

## The job runner (how everything runs on this PC)
Double-click `runner2.bat` once and leave its window open. It activates `.venv`, keeps the PC awake and runs every
`.bat` dropped into `jobs\q2\` in name order, logging to `logs\jobs\<name>.log` and moving the job to `jobs\q2_done\`.
`logs\runner2.log` is the queue history. A job is one or more commands, e.g.

```
@echo off
python -u scripts\04_train.py --model gated_unet --config configs\gate_static.yaml --run-dir runs\gated_unet_v8
python -u scripts\05_evaluate.py --ckpt runs\gated_unet_v8\best.pt --split val
```

Jobs run one at a time; a long job blocks the queue. Only `.venv` Python is on the path.

## Data build
Done. To rebuild or extend (resumable - rerun after any interruption):
```
python scripts\01_build_event_list.py --years 2018 2019 2020 2021 2022 2023 2024 2025 2026
python scripts\03_build_samples.py --workers 6
python scripts\06_index_samples.py
```
Measured 5-7 s per sample with 6 workers (network-bound). The HRRR archive has gaps in 2018-2020 (about 1.4 % of
samples are skipped and logged). `pyproj` is blocked by Windows Application Control on this PC; the code no longer needs it.

## Training and evaluation
```
python scripts\04_train.py --model gated_unet --config configs\gate_static.yaml --run-dir runs\gated_unet_v8
python scripts\05_evaluate.py --ckpt runs\gated_unet_v8\best.pt --split val
```
About 150 s per epoch on the RTX 5060 Ti; a run early-stops after 10-27 epochs. Evaluation on val takes ~7 min and also
writes `eval_val_preds.npz` for offline analysis. Overlays: `configs\genesis_w3.yaml`, `genesis_w1.yaml`,
`station_dropout.yaml`, `gate_static.yaml`. The 2026 test split is evaluated ONCE at the very end (`--split test`).

## Figures, live test, dashboard
```
python scripts\07_demo_figures.py --event dust_20250825_2236 --split val      # 2x3 panels + alerts.csv
python scripts\09_slide_media.py                                            # replay strips, gate maps, threshold curves
python scripts\10_live_smoke.py                                             # live GOES/GLM/HRRR/ASOS -> prediction
python scripts\launch_dashboard.py --demo --samples data\samples\val         # detached Streamlit on http://localhost:8501
```
The dashboard PID is written to `logs\dashboard.pid`; stop it with `taskkill /PID <pid> /T /F`.
