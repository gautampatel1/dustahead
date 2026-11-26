@echo off
REM One live forecast for the web and phone apps: predict -> alerts -> bundle -> upload to R2 -> heartbeat.
REM Run every 10 minutes by the "DustAhead publish" scheduled task (install_publish_task.bat).
REM   publish_live.bat          publish and upload
REM   publish_live.bat local    publish to runs\publish only (no upload), for a first test without R2
REM Log: logs\publish\YYYY-MM-DD.log, one file per day. Secrets: .secrets\publish.env (see configs\publish.env.example).
setlocal
cd /d "%~dp0..\.."
set PY=.venv\Scripts\python.exe
if not exist "%PY%" (
  echo .venv not found in %CD%. Run bootstrap_windows.bat first.
  exit /b 2
)
if not exist logs\publish mkdir logs\publish
for /f %%d in ('"%PY%" -c "import datetime; print(datetime.date.today())"') do set DAY=%%d
set LOG=logs\publish\%DAY%.log
REM same environment as the job runner (runner2.bat)
set DUSTAHEAD_TMP=%TEMP%
set HYRIVER_CACHE_DISABLE=true
set UPLOAD=--upload
if /i "%~1"=="local" set UPLOAD=
echo ==== %DATE% %TIME% start %UPLOAD% >> "%LOG%"
"%PY%" -u scripts\35_publish.py --live --once %UPLOAD% >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%
echo ==== %DATE% %TIME% exit %RC% >> "%LOG%"
exit /b %RC%
