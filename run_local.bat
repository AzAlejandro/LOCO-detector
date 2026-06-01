@echo off
title LOCO Detector
setlocal enabledelayedexpansion

echo ============================================
echo   LOCO Detector v1.0.0
echo   Automated circle detection for TEM images
echo ============================================
echo.

:: --- Read .env configuration ---
set "BACKEND_PORT=8011"
set "BACKEND_HOST=127.0.0.1"
set "FRONTEND_PORT=5173"
set "FRONTEND_HOST=localhost"
set "VITE_API_BASE=http://127.0.0.1:8011"

if exist ".env" (
    for /f "usebackq delims=" %%a in (".env") do (
        set "line=%%a"
        if not "!line!"=="" if not "!line:~0,1!"=="#" (
            for /f "tokens=1,* delims==" %%b in ("!line!") do (
                set "key=%%b"
                set "val=%%c"
                if "!key!"=="BACKEND_PORT" set "BACKEND_PORT=!val!"
                if "!key!"=="BACKEND_HOST" set "BACKEND_HOST=!val!"
                if "!key!"=="FRONTEND_PORT" set "FRONTEND_PORT=!val!"
                if "!key!"=="FRONTEND_HOST" set "FRONTEND_HOST=!val!"
                if "!key!"=="VITE_API_BASE" set "VITE_API_BASE=!val!"
            )
        )
    )
)

echo   Backend port : %BACKEND_PORT%
echo   Frontend port: %FRONTEND_PORT%
echo.

:: --- Sync VITE_API_BASE into frontend/.env ---
echo VITE_API_BASE=%VITE_API_BASE% > frontend\.env

:: --- Check Python ---
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b 1
)

:: --- Check Node ---
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not installed or not in PATH.
    pause
    exit /b 1
)

:: --- Stop any leftover LOCO processes before starting ---
echo [INFO] Stopping any leftover LOCO processes...
powershell -ExecutionPolicy Bypass -File .\stop_servers.ps1
echo.

:: --- Install Python dependencies if needed ---
if not exist "venv\" (
    echo [INFO] Creating Python virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

:: --- Install frontend dependencies if needed ---
if not exist "frontend\node_modules\" (
    echo [INFO] Installing frontend dependencies...
    cd frontend
    npm install
    cd ..
)

:: --- Start backend ---
echo [INFO] Starting backend on http://%BACKEND_HOST%:%BACKEND_PORT%
start "LOCO-Backend" cmd /c "cd /d %~dp0 && venv\Scripts\python.exe app.py"

:: --- Wait for backend to start (poll health endpoint up to 30 seconds) ---
echo [INFO] Waiting for backend to be ready...
set "BACKEND_READY="
for /l %%i in (1,1,30) do (
    timeout /t 1 /nobreak >nul
    curl -s http://%BACKEND_HOST%:%BACKEND_PORT%/api/health >nul 2>&1
    if not errorlevel 1 (
        set "BACKEND_READY=1"
        goto backend_ready
    )
)
:backend_ready
if defined BACKEND_READY (
    echo [INFO] Backend is ready.
) else (
    echo [WARNING] Backend did not respond within 30 seconds. Starting frontend anyway...
)

:: --- Start frontend ---
echo [INFO] Starting frontend on http://%FRONTEND_HOST%:%FRONTEND_PORT%
start "LOCO-Frontend" cmd /c "cd frontend && npm run dev -- --port %FRONTEND_PORT% --host %FRONTEND_HOST%"

echo.
echo ============================================
echo   LOCO Detector is running!
echo   Frontend: http://%FRONTEND_HOST%:%FRONTEND_PORT%
echo   Backend:  http://%BACKEND_HOST%:%BACKEND_PORT%
echo ============================================
echo.
echo   Close this window to keep servers running.
echo   Use stop_servers.ps1 to stop all processes.
echo.

endlocal
