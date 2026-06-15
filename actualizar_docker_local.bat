@echo off
title Actualizar LOCO Detector en Docker local
setlocal EnableExtensions

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"

echo ==========================================
echo Actualizar LOCO Detector en Docker local
echo ==========================================
echo.
echo Este script reconstruye backend y frontend desde el codigo actual,
echo reemplaza las imagenes locales :latest y levanta la app.
echo.
echo No borra el volumen de datos loco_outputs.
echo.

cd /d "%ROOT%" || (
  echo [ERROR] No se pudo entrar a la raiz del repo.
  pause
  exit /b 1
)

where docker >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Docker no esta instalado o no esta disponible en PATH.
  echo Instala Docker Desktop y vuelve a ejecutar este archivo.
  pause
  exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Docker Desktop no esta iniciado.
  echo Abre Docker Desktop, espera que termine de iniciar y vuelve a ejecutar este archivo.
  pause
  exit /b 1
)

docker compose version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Docker Compose no esta disponible.
  echo Actualiza Docker Desktop y vuelve a ejecutar este archivo.
  pause
  exit /b 1
)

if not exist "%ROOT%\docker-compose.yml" (
  echo [ERROR] Falta docker-compose.yml.
  pause
  exit /b 1
)

if not exist "%ROOT%\docker-compose.release.yml" (
  echo [ERROR] Falta docker-compose.release.yml.
  pause
  exit /b 1
)

echo [INFO] Deteniendo contenedores anteriores si existen...
docker compose -f docker-compose.release.yml down
docker compose down

echo [INFO] Eliminando contenedores LOCO con nombres fijos si quedaron huerfanos...
docker rm -f loco-detector-backend >nul 2>&1
docker rm -f loco-detector-frontend >nul 2>&1

echo.
echo [INFO] Reconstruyendo imagenes desde el codigo actual...
docker compose build backend frontend
if errorlevel 1 (
  echo [ERROR] Fallo docker compose build.
  pause
  exit /b 1
)

echo.
echo [INFO] Verificando imagen backend...
docker image inspect loco-detector-backend:latest >nul 2>&1
if errorlevel 1 (
  echo [ERROR] No existe loco-detector-backend:latest despues del build.
  pause
  exit /b 1
)

echo [INFO] Verificando imagen frontend...
docker image inspect loco-detector-frontend:latest >nul 2>&1
if errorlevel 1 (
  echo [ERROR] No existe loco-detector-frontend:latest despues del build.
  pause
  exit /b 1
)

echo.
echo [INFO] Levantando contenedores con las imagenes actualizadas...
docker compose -f docker-compose.release.yml up -d --force-recreate
if errorlevel 1 (
  echo [ERROR] No se pudo iniciar LOCO Detector.
  echo Puede que los puertos 8011 o 5178 esten ocupados.
  echo Revisa con: docker ps
  pause
  exit /b 1
)

echo.
echo [INFO] Esperando backend...
set /a attempts=0

:wait_backend
set /a attempts+=1
curl --silent --fail "http://127.0.0.1:8011/api/health" >nul 2>&1
if not errorlevel 1 goto :ready
if %attempts% GEQ 60 goto :failed
timeout /t 2 /nobreak >nul
goto :wait_backend

:ready
echo.
echo [OK] LOCO Detector actualizado y listo.
echo Frontend: http://localhost:5178
echo Backend:  http://localhost:8011/api/health
echo.
echo Para cerrar luego usa:
echo   detener_loco_detector.bat
echo.
start "" "http://localhost:5178"
pause
exit /b 0

:failed
echo.
echo [ERROR] El backend no respondio a tiempo.
echo [INFO] Ultimos logs:
docker compose -f docker-compose.release.yml logs --tail 80
pause
exit /b 1
