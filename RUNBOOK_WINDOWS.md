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

## Live forecasts for the apps (scheduled task, every 10 minutes)
The web and phone apps read `latest.json` and three dust maps from a Cloudflare R2 bucket. On this PC a Windows
scheduled task runs `scripts\windows\publish_live.bat` at :02, :12, :22 and so on. Each run does one live forecast
(`scripts\35_publish.py --live --once --upload`): GOES/GLM/HRRR/ASOS -> v8 -> alert tiers + AlertTracker ->
bundle -> R2 -> Healthchecks ping. It runs beside the job runner, not through it, so a long training job can't
hold up the forecasts.

One-time setup:
1. **Secrets.** Copy `configs\publish.env.example` to `.secrets\publish.env` and fill in the four R2 values and the
   Healthchecks URL. `.secrets\` is gitignored; never commit the filled-in file.
   - R2: create a bucket and an API token with "Object Read & Write" on that bucket only. Turn on its public URL
     and add a CORS rule allowing `GET` from the website's address.
   - Healthchecks.io: create a check with period 10 min and grace 15 min, and add the team's emails. It pages when
     runs stop or fail, which is the same moment the apps show "No new forecast".
2. **Test without uploading.** In this folder run `scripts\windows\publish_live.bat local` and read
   `logs\publish\<today>.log`. It should end with `[publish] t0 ...Z alerts N uploaded 0 ...s` and `exit 0`, and
   `runs\publish\latest.json` should exist. Note the run time: it must stay well under 9 minutes.
3. **Install the task.** Double-click `scripts\windows\install_publish_task.bat`. It registers "DustAhead publish",
   starts a first run and prints the next run time.
   - **As SYSTEM** (no console window, runs with nobody logged on): open Command Prompt as administrator, go to
     `C:\DustAhead`, and run `scripts\windows\install_publish_task.bat system`.
   - **Remove:** `install_publish_task.bat remove`.

Day to day:
- Logs: `logs\publish\YYYY-MM-DD.log`, one per day. Each run is framed by `==== ... start` and `==== ... exit N`.
  - Exit 0 = published.
  - Exit 1 = the forecast failed, with a traceback in the log; Healthchecks was told.
  - Exit 2 = setup problem (no `.venv`, or `--upload` without R2 secrets).
  - Exit 3 = the run hit its 9-minute limit and was stopped.
- Alert hysteresis (30-minute minimum hold, release below 0.50) survives between runs in `runs\publish\tracker.json`.
  A state older than three runs is dropped, so after an outage alerts restart from the thresholds.
- The PC must stay awake. Sleep stops the forecasts: set Power > Sleep to Never, or keep `keep_awake.bat` running.
- The task uses the GPU next to any training job. v8 inference is small, but a training run that fills the 16 GB can
  make a forecast fail with out-of-memory; the log and Healthchecks will show it.
- To see what the apps see, open the site with `?src=<R2 public URL>/`, or locally run `cd web` and
  `npm run dev`, then open `/?src=data/` after copying `runs\publish\*` into `web\public\data\`.

## Figures, live test, dashboard
```
python scripts\07_demo_figures.py --event dust_20250825_2236 --split val      # 2x3 panels + alerts.csv
python scripts\09_slide_media.py                                            # replay strips, gate maps, threshold curves
python scripts\10_live_smoke.py                                             # live GOES/GLM/HRRR/ASOS -> prediction
python scripts\launch_dashboard.py --demo --samples data\samples\val         # detached Streamlit on http://localhost:8501
```
The dashboard PID is written to `logs\dashboard.pid`; stop it with `taskkill /PID <pid> /T /F`.
