@echo off
REM Registers the Windows scheduled task "DustAhead publish": scripts\windows\publish_live.bat every 10 minutes,
REM at :02, :12, :22 ... (two minutes after each 10-minute forecast time, for GOES latency).
REM   install_publish_task.bat           install or replace the task for the logged-on user, then run it once now
REM   install_publish_task.bat system    same, but as SYSTEM: runs with nobody logged on and shows no window
REM                                      (right-click > Run as administrator)
REM   install_publish_task.bat remove    delete the task
REM Overlapping runs cannot pile up: the task does not start a new run while one is going, and every run stops itself
REM after 9 minutes (35_publish.py --max-seconds).
setlocal
set TASK=DustAhead publish
set BAT=%~dp0publish_live.bat

if /i "%~1"=="remove" (
  schtasks /Delete /TN "%TASK%" /F
  pause
  exit /b
)

set RUNAS=
if /i "%~1"=="system" set RUNAS=/RU SYSTEM

schtasks /Create /TN "%TASK%" /TR "\"%BAT%\"" /SC MINUTE /MO 10 /ST 00:02 /RL LIMITED %RUNAS% /F
if errorlevel 1 (
  echo.
  echo Could not create the task. For "system", run this file as administrator.
  pause
  exit /b 1
)
echo.
echo Installed "%TASK%". Starting the first run now...
schtasks /Run /TN "%TASK%"
echo.
schtasks /Query /TN "%TASK%" /V /FO LIST | findstr /i /c:"Next Run Time" /c:"Status" /c:"Run As User" /c:"Task To Run"
echo.
echo Watch logs\publish\ for today's log. A run takes a few minutes (satellite, HRRR and station downloads).
pause
