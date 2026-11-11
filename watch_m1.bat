@echo off
REM Shows the M1 bots log live (the bot itself runs under start_bots.bat).
cd /d "%~dp0"
title M1 bot log
powershell -NoProfile -Command "Get-Content logs\m1.log -Wait -Tail 40"
pause
