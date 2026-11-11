@echo off
REM Double-click to open the bot supervisor. Both bots start STOPPED:
REM press Start on each bot in the phone dashboard to run it.
REM Keep this window open: it is what carries out the dashboard buttons.
REM watch_m5.bat / watch_m1.bat show what the bots are doing.
REM Ctrl+C here (or closing this window) stops both bots.
cd /d "%~dp0"
title XAU/TISM bots - supervisor (keep open, close to stop both bots)
git pull --ff-only
python -m bot.supervisor --stopped
:again
if %errorlevel%==3 (
  echo The supervisor was updated, starting the new one...
  python -m bot.supervisor --resume
  goto again
)
pause
