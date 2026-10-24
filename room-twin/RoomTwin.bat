@echo off
rem Room Twin: double-click to start (or: RoomTwin.bat --colab / stop / status)
cd /d "%~dp0"
where py >nul 2>nul && (py -3 launch.py %*) || (python launch.py %*)
pause
