@echo off
REM Double-click to start the M5 bot (uses .env)
cd /d "%~dp0"
git pull
python -m bot.main --bot m5
pause
