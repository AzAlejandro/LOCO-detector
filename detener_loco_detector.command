@echo off
title Detener LOCO Detector
setlocal
cd /d "%~dp0"

echo Deteniendo LOCO Detector...
docker compose -f docker-compose.release.yml down

echo.
echo LOCO Detector fue detenido.
echo Los datos importados se mantienen en el volumen Docker.
echo.
echo Para borrar tambien los datos importados se usaria docker compose down -v,
echo pero no se recomienda para usuarios normales.
pause