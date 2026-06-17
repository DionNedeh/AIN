@echo off
setlocal

where powershell.exe >nul 2>nul
if errorlevel 1 (
    echo PowerShell was not found on this system.
    pause
    exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0ActivityFeed.ps1" %*

echo.
echo Terminal Activity Feed stopped.
pause
