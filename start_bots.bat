@echo off
REM Double-click to start BOTH bots with the dashboard supervisor.
REM The bots run with no windows of their own; use watch_m5.bat /
REM watch_m1.bat to see what they are doing.
REM Ctrl+C here (or closing this window) stops both bots.
cd /d "%~dp0"
title XAU/TISM bots - supervisor (close to stop both bots)
git pull --ff-only
python -m bot.supervisor
:again
if %errorlevel%==3 (
  echo The supervisor was updated, starting the new one...
  python -m bot.supervisor --resume
  goto again
)
pause
