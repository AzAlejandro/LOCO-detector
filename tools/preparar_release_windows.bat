@echo off
setlocal EnableExtensions EnableDelayedExpansion

if /I "%~1"=="--help" goto :usage
if /I "%~1"=="-h" goto :usage
if /I "%~1"=="/?" goto :usage

set "VERSION=%~1"
if "%VERSION%"=="" set "VERSION=v1.0.0"

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "ASSETS=%ROOT%\release-assets"
set "RELEASE_DIR=%ASSETS%\LOCO-detector-%VERSION%-release"
set "IMAGE_TAR=%ASSETS%\loco-detector-%VERSION%-docker-images.tar"
set "IMAGE_TGZ=%ASSETS%\loco-detector-%VERSION%-docker-images.tar.gz"
set "RELEASE_ZIP=%ASSETS%\LOCO-detector-%VERSION%-release.zip"

echo [INFO] Preparando release Windows %VERSION%
echo [INFO] Repo: %ROOT%

cd /d "%ROOT%" || (
  echo [ERROR] No se pudo entrar a la raiz del repo.
  exit /b 1
)

for %%F in (
  "docker-compose.release.yml"
  "docs\README_USUARIO.md"
  "iniciar_loco_detector.bat"
  "detener_loco_detector.bat"
  "iniciar_loco_detector.command"
  "detener_loco_detector.command"
) do (
  if not exist "%ROOT%\%%~F" (
    echo [ERROR] Falta %%~F. No se puede armar la release.
    exit /b 1
  )
)

docker info >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Docker Desktop no esta disponible. Abre Docker Desktop y espera a que termine de iniciar.
  exit /b 1
)

echo [INFO] Limpiando carpeta release-assets...
if exist "%ASSETS%" rmdir /s /q "%ASSETS%"
mkdir "%ASSETS%" || (
  echo [ERROR] No se pudo crear release-assets.
  exit /b 1
)

echo [INFO] Construyendo imagenes Docker...
docker compose build
if errorlevel 1 (
  echo [ERROR] Fallo docker compose build.
  exit /b 1
)

echo [INFO] Verificando imagen backend...
docker image inspect loco-detector-backend:latest >nul 2>&1
if errorlevel 1 (
  echo [ERROR] No existe loco-detector-backend:latest despues del build.
  exit /b 1
)

echo [INFO] Verificando imagen frontend...
docker image inspect loco-detector-frontend:latest >nul 2>&1
if errorlevel 1 (
  echo [ERROR] No existe loco-detector-frontend:latest despues del build.
  exit /b 1
)

echo [INFO] Exportando backend y frontend en un solo TAR...
docker save -o "%IMAGE_TAR%" loco-detector-backend:latest loco-detector-frontend:latest
if errorlevel 1 (
  echo [ERROR] Fallo docker save.
  exit /b 1
)

echo [INFO] Comprimiendo imagenes Docker...
tar -czf "%IMAGE_TGZ%" -C "%ASSETS%" "loco-detector-%VERSION%-docker-images.tar"
if errorlevel 1 (
  echo [ERROR] Fallo compresion tar.gz.
  exit /b 1
)

echo [INFO] Armando carpeta final...
mkdir "%RELEASE_DIR%" || (
  echo [ERROR] No se pudo crear la carpeta final.
  exit /b 1
)

copy "%IMAGE_TGZ%" "%RELEASE_DIR%\" >nul || exit /b 1
copy "%ROOT%\docker-compose.release.yml" "%RELEASE_DIR%\" >nul || exit /b 1
copy "%ROOT%\docs\README_USUARIO.md" "%RELEASE_DIR%\README_USUARIO.md" >nul || exit /b 1
copy "%ROOT%\iniciar_loco_detector.bat" "%RELEASE_DIR%\" >nul || exit /b 1
copy "%ROOT%\detener_loco_detector.bat" "%RELEASE_DIR%\" >nul || exit /b 1
copy "%ROOT%\iniciar_loco_detector.command" "%RELEASE_DIR%\" >nul || exit /b 1
copy "%ROOT%\detener_loco_detector.command" "%RELEASE_DIR%\" >nul || exit /b 1

echo [INFO] Comprimiendo carpeta final...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Compress-Archive -LiteralPath '%RELEASE_DIR%' -DestinationPath '%RELEASE_ZIP%' -Force"
if errorlevel 1 (
  echo [ERROR] Fallo Compress-Archive.
  exit /b 1
)

echo.
echo [OK] Release creada:
echo %RELEASE_ZIP%
echo.
echo [INFO] Esta carpeta esta ignorada por Git. Sube el ZIP como asset en GitHub Releases.

endlocal
exit /b 0

:usage
echo Uso:
echo   tools\preparar_release_windows.bat [version]
echo.
echo Ejemplos:
echo   tools\preparar_release_windows.bat
echo   tools\preparar_release_windows.bat v1.0.0
echo.
echo El script limpia release-assets, construye backend/frontend con Docker,
echo exporta ambas imagenes y genera el ZIP final de release.
exit /b 0
