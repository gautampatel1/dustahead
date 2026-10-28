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

## Data build (prepared on 2026-10-24)
Done: `data\labels\events.csv` (152 dust events + 268 null days, 2023-2026), `data\static\*.npy` (six
ground layers), first 3 samples of the Phoenix 2025-08-25 event in `data\samples\val\`.

Run the full build in the background (resumable - rerun the same command after any interruption):
```
conda activate dustahead
python scripts\03_build_samples.py --workers 6 > logs\build_samples.log 2>&1
```
~11,000 samples; measured 5.7 s each with 6 worker processes (network-bound, ~80 Mbit/s) -> about 17 h. Check progress any time with
`Get-Content logs\build_samples.log -Tail 20` or by counting files in `data\samples\*`.

## Training (after the build)
```
python scripts\04_train.py --model gated_unet        > logs\train_gated.log 2>&1
python scripts\05_evaluate.py --ckpt runs\latest\best.pt --split val
```
Then the ablations: `--model unet_all`, `--model unet_dust_only`, `--model convlstm`.
The 2026 test split is evaluated ONCE at the very end (`--split test`).

## Dashboard
```
streamlit run dustahead\app\dashboard.py -- --ckpt runs\latest\best.pt --demo
```
