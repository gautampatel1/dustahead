# DustAhead — running the data build and training on the Windows PC

Everything below runs in a normal Windows terminal (PowerShell) in `C:\DustAhead`, NOT inside Claude's
Linux shell: that shell is limited to 3-minute commands and cannot see the GPU. Claude reads the logs
from this folder and fixes anything that fails.

## One-time setup (10-20 min)
1. Install Miniconda if `conda --version` fails: https://docs.conda.io/en/latest/miniconda.html, then
   open a new PowerShell and run `conda init powershell`, close and reopen PowerShell.
2. `cd C:\DustAhead`
3. `powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1`
   Creates the `dustahead` conda environment (eccodes/cfgrib for HRRR come from conda-forge), installs
   PyTorch for CUDA 12.8, runs the tests and the preflight. Expect every line `OK`.

## Data build (already prepared by Claude on 2026-10-24)
Done: `data\labels\events.csv` (152 dust events + 268 null days, 2023-2026), `data\static\*.npy` (six
ground layers), first 3 samples of the Phoenix 2025-08-25 event in `data\samples\val\`.

Run the full build in the background (resumable - rerun the same command after any interruption):
```
conda activate dustahead
python scripts\03_build_samples.py --workers 6 > logs\build_samples.log 2>&1
```
~11,000 samples; roughly 5 s each with 6 workers -> about 15 h. Check progress any time with
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
