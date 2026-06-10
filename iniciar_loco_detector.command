@echo off
title Iniciar LOCO Detector
setlocal
cd /d "%~dp0"

set "IMAGE_ARCHIVE=loco-detector-v1.0.0-docker-images.tar.gz"
set "IMAGE_TAR=loco-detector-v1.0.0-docker-images.tar"

echo ==========================================
echo Iniciando LOCO Detector
echo ==========================================
echo.

where docker >nul 2>&1
if errorlevel 1 (
  echo ERROR: Docker no esta instalado o no esta disponible en PATH.
  echo Instala Docker Desktop y vuelve a ejecutar este archivo.
  pause
  exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
  echo ERROR: Docker Desktop no esta iniciado.
  echo Abre Docker Desktop, espera que termine de iniciar y vuelve a ejecutar este archivo.
  pause
  exit /b 1
)

docker compose version >nul 2>&1
if errorlevel 1 (
  echo ERROR: Docker Compose no esta disponible.
  echo Actualiza Docker Desktop y vuelve a ejecutar este archivo.
  pause
  exit /b 1
)

if not exist "%IMAGE_ARCHIVE%" (
  echo ERROR: No se encontro %IMAGE_ARCHIVE%.
  echo Este archivo debe estar en la misma carpeta que este .bat.
  pause
  exit /b 1
)

if not exist "docker-compose.release.yml" (
  echo ERROR: No se encontro docker-compose.release.yml.
  echo Este archivo debe estar en la misma carpeta que este .bat.
  pause
  exit /b 1
)

if not exist "%IMAGE_TAR%" (
  echo Descomprimiendo imagenes Docker...
  tar -xzf "%IMAGE_ARCHIVE%"
)

if not exist "%IMAGE_TAR%" (
  echo ERROR: No se pudo descomprimir %IMAGE_ARCHIVE%.
  pause
  exit /b 1
)

echo Cargando imagenes Docker...
docker load -i "%IMAGE_TAR%"
if errorlevel 1 (
  echo ERROR: Docker no pudo cargar las imagenes.
  pause
  exit /b 1
)

echo Deteniendo contenedores anteriores si existen...
docker compose -f docker-compose.release.yml down

echo Levantando LOCO Detector...
docker compose -f docker-compose.release.yml up -d
if errorlevel 1 (
  echo ERROR: No se pudo iniciar LOCO Detector.
  echo Puede que los puertos 8011 o 5178 esten ocupados.
  echo Cierra otras versiones de LOCO Detector o reinicia el equipo.
  pause
  exit /b 1
)

echo Esperando backend...
set /a attempts=0

:wait_backend
set /a attempts+=1
curl --silent --fail "http://127.0.0.1:8011/api/health" >nul 2>&1
if not errorlevel 1 goto :ready
if %attempts% GEQ 60 goto :failed
timeout /t 2 /nobreak >nul
goto :wait_backend

:ready
echo LOCO Detector listo.
echo Frontend: http://localhost:5178
start "" "http://localhost:5178"
pause
exit /b 0

:failed
echo ERROR: El backend no respondio a tiempo.
docker compose -f docker-compose.release.yml logs --tail 80
pause
exit /b 1