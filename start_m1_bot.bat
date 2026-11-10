@echo off
REM Double-click to start the M1 bot (uses .env.m1)
cd /d "%~dp0"
git pull
python -m bot.main --bot m1
pause
