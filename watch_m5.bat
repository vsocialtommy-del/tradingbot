@echo off
REM Shows the M5 bots log live (the bot itself runs under start_bots.bat).
cd /d "%~dp0"
title M5 bot log
powershell -NoProfile -Command "Get-Content logs\m5.log -Wait -Tail 40"
pause
