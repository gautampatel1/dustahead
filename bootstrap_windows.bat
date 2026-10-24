@echo off
REM DustAhead bootstrap: creates a Python environment with GPU PyTorch, runs tests + preflight, then builds the data set.
REM Double-click this file (or run it in PowerShell) from C:\DustAhead. Everything is logged to logs\.
setlocal EnableDelayedExpansion
cd /d %~dp0
if not exist logs mkdir logs
set LOG=logs\bootstrap.log
echo ==== DustAhead bootstrap %date% %time% ==== > %LOG%

set PY=
for %%v in (3.12 3.11 3.10 3.13) do (
  if "!PY!"=="" ( py -%%v -c "import sys" >nul 2>&1 && set PY=py -%%v )
)
if "!PY!"=="" ( python -c "import sys; assert sys.version_info[:2] >= (3,10)" >nul 2>&1 && set PY=python )
if "!PY!"=="" (
  echo No Python 3.10-3.13 found. Install Python 3.12 from https://www.python.org/downloads/windows/ ^(tick "Add python.exe to PATH" and "py launcher"^) and run this again. >> %LOG%
  type %LOG%
  pause
  exit /b 1
)
echo Using !PY! >> %LOG%
!PY! --version >> %LOG% 2>&1

if not exist .venv (
  !PY! -m venv .venv >> %LOG% 2>&1 || (echo venv creation failed >> %LOG% & type %LOG% & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip wheel >> %LOG% 2>&1
echo [1/6] PyTorch for CUDA 12.8 (RTX 50-series) ... >> %LOG%
pip install torch --index-url https://download.pytorch.org/whl/cu128 >> %LOG% 2>&1
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')" >> %LOG% 2>&1
echo [2/6] project packages ... >> %LOG%
pip install -r requirements.txt >> %LOG% 2>&1
pip install eccodes cfgrib >> %LOG% 2>&1
python -c "import cfgrib, eccodes; print('eccodes', eccodes.__version__)" >> %LOG% 2>&1
echo [3/6] unit tests ... >> %LOG%
python -m pytest -q -p no:cacheprovider >> %LOG% 2>&1
echo [4/6] preflight ... >> %LOG%
set DUSTAHEAD_TMP=%TEMP%
set HYRIVER_CACHE_DISABLE=true
python scripts\00_check_sources.py --NLCD >> %LOG% 2>&1
echo [5/6] sanity: 2 samples of the Phoenix event ... >> %LOG%
python -u scripts\03_build_samples.py --event dust_20250825_2236 --max-samples 6 --workers 4 >> %LOG% 2>&1
echo [6/6] FULL BUILD (hours; resumable - rerun this file to continue) ... >> %LOG%
echo Build started %date% %time% >> %LOG%
python -u scripts\03_build_samples.py --workers 6 >> logs\build_samples.log 2>&1
echo Build finished %date% %time% >> %LOG%
type %LOG%
pause
