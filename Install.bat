@echo off
title Myna - Installer
echo.
echo   Installing Myna. This takes a few minutes and downloads
echo   roughly 2 GB of packages the first time. Please leave this window open.
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
echo.
if errorlevel 1 (
  echo   Installation failed. The error is shown above.
) else (
  echo   Done. Look for "Myna" on your Desktop and Start Menu.
)
echo.
pause
