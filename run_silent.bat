@echo off
:: run_silent.bat - Thin wrapper around tools\stop_servers.ps1
:: This is called by run_silent.vbs with window style 0 (hidden).
:: Delegates all shutdown logic to tools\stop_servers.ps1 (single implementation).

set "PROJECT_DIR=%~1"
if "%PROJECT_DIR%"=="" set "PROJECT_DIR=%cd%"
cd /d "%PROJECT_DIR%"

:: Delegate to tools\stop_servers.ps1 - no duplicate kill logic here
powershell -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_DIR%\tools\stop_servers.ps1"
