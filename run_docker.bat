@echo off
setlocal
cd /d "%~dp0"

if not defined DOCKER_BACKEND_PORT set "DOCKER_BACKEND_PORT=8011"
if not defined DOCKER_FRONTEND_PORT set "DOCKER_FRONTEND_PORT=5178"
if not defined DOCKER_VITE_API_BASE set "DOCKER_VITE_API_BASE=http://localhost:%DOCKER_BACKEND_PORT%"

where docker >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Docker no esta instalado o no esta disponible en PATH.
  echo Instala Docker Desktop y vuelve a ejecutar este archivo.
  exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Docker Desktop no esta iniciado.
  echo Abre Docker Desktop, espera que termine de iniciar y vuelve a ejecutar este archivo.
  exit /b 1
)

docker compose version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Docker Compose no esta disponible.
  echo Actualiza Docker Desktop y vuelve a ejecutar este archivo.
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File ".\check_docker_ports.ps1" -BackendPort %DOCKER_BACKEND_PORT% -FrontendPort %DOCKER_FRONTEND_PORT%
if errorlevel 1 (
  echo [ERROR] No se iniciara Docker mientras existan servidores locales en conflicto.
  exit /b 1
)

echo [INFO] Construyendo y levantando LOCO Detector...
docker compose up --build -d
if errorlevel 1 goto :failed

echo [INFO] Esperando backend en http://127.0.0.1:%DOCKER_BACKEND_PORT%/api/health ...
set /a attempts=0

:wait_backend
set /a attempts+=1
curl --silent --fail "http://127.0.0.1:%DOCKER_BACKEND_PORT%/api/health" >nul 2>&1
if not errorlevel 1 goto :ready
if %attempts% GEQ 60 goto :failed
timeout /t 2 /nobreak >nul
goto :wait_backend

:ready
echo [INFO] LOCO Detector listo.
echo [INFO] Frontend: http://localhost:%DOCKER_FRONTEND_PORT%
echo [INFO] Para restaurar trabajo, entra a Configuracion e importa el ZIP exportado.
docker compose ps
start "" "http://localhost:%DOCKER_FRONTEND_PORT%"
exit /b 0

:failed
echo [ERROR] No se pudo levantar LOCO Detector.
docker compose ps
docker compose logs --tail 80
exit /b 1
