@echo off
title LOCO Detector
echo ============================================
echo   LOCO Detector v1.0.0
echo   Automated circle detection for TEM images
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b 1
)

:: Check Node
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not installed or not in PATH.
    pause
    exit /b 1
)

:: Install Python dependencies if needed
if not exist "venv\" (
    echo [INFO] Creating Python virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

:: Install frontend dependencies if needed
if not exist "frontend\node_modules\" (
    echo [INFO] Installing frontend dependencies...
    cd frontend
    npm install
    cd ..
)

:: Start backend
echo [INFO] Starting backend on http://127.0.0.1:8011
start "LOCO-Backend" cmd /c "call venv\Scripts\activate.bat && python app.py"

:: Wait for backend to start (poll health endpoint up to 30 seconds)
echo [INFO] Waiting for backend to be ready...
set "BACKEND_READY="
for /l %%i in (1,1,30) do (
    timeout /t 1 /nobreak >nul
    curl -s http://127.0.0.1:8011/api/health >nul 2>&1
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

:: Start frontend
echo [INFO] Starting frontend on http://localhost:5173
start "LOCO-Frontend" cmd /c "cd frontend && npm run dev"

echo.
echo ============================================
echo   LOCO Detector is running!
echo   Frontend: http://localhost:5173
echo   Backend:  http://127.0.0.1:8011
echo ============================================
echo.
echo   Close this window to keep servers running.
echo   Use stop_servers.ps1 to stop all processes.
echo.
