# DustAhead one-time setup on Windows (run in PowerShell from C:\DustAhead):
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
# Needs Miniconda (https://docs.conda.io/en/latest/miniconda.html) - conda-forge is the only clean
# source of eccodes/cfgrib on Windows, which Herbie needs for HRRR.
$ErrorActionPreference = "Stop"
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
  Write-Host "conda not found. Install Miniconda, reopen PowerShell, run 'conda init powershell', then rerun this script."
  exit 1
}
conda env create -f environment.yml -n dustahead 2>$null; if ($LASTEXITCODE -ne 0) { conda env update -f environment.yml -n dustahead }
conda activate dustahead
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python -m pytest -q
python scripts\00_check_sources.py
