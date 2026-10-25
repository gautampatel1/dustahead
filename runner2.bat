@echo off
REM DustAhead job runner (Python). Leave this window open. Drop .bat files into jobs\q2\ to run them.
cd /d %~dp0
call .venv\Scripts\activate.bat
set DUSTAHEAD_TMP=%TEMP%
set HYRIVER_CACHE_DISABLE=true
title DustAhead job runner - leave this window open
python -u dustahead\runner.py
pause
