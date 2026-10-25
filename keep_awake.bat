@echo off
REM Keeps the PC awake while this window is open (no power-setting changes). Close the window to stop.
cd /d %~dp0
call .venv\Scripts\activate.bat
title DustAhead keep-awake - leave this window open
python dustahead\keep_awake.py
pause
