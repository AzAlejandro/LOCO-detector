@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "CHECKPOINT_REMOTE=origin"
set "CHECKPOINT_SHA=%~1"
set "TEMP_PATHS=%TEMP%\loco_detector_ui_paths_%RANDOM%_%RANDOM%.txt"

echo ============================================================
echo LOCO Detector - Restaurar etapa visual
echo ============================================================
echo.

where git >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Git no esta instalado o no esta disponible en PATH.
  exit /b 1
)

if not exist ".git\" (
  echo [ERROR] Ejecuta este archivo desde una copia Git del repositorio.
  exit /b 1
)

if "%CHECKPOINT_SHA%"=="" (
  echo [ERROR] Falta el SHA aprobado que deseas restaurar.
  echo Uso: restore_ui_stage.bat ^<sha-aprobado^>
  exit /b 1
)

git diff --cached --quiet
if errorlevel 2 (
  echo [ERROR] Git no pudo revisar el indice local.
  exit /b 1
)
if errorlevel 1 (
  echo [ERROR] Hay cambios preparados para commit ^(staged^).
  echo Haz commit o quita esos cambios del stage antes de restaurar.
  exit /b 1
)

echo SHA solicitado: %CHECKPOINT_SHA%
echo.
echo Este proceso:
echo   - Descarga el SHA exacto desde GitHub.
echo   - Restaura solamente archivos versionados dentro de frontend.
echo   - Conserva backend, outputs, imagenes, .env y archivos adicionales.
echo   - No elimina archivos, no cambia de rama y no ejecuta git reset.
echo.
echo [ADVERTENCIA] Los cambios locales sin commit dentro de frontend se sobrescribiran.
set /p "CONFIRM=Escribe SI para restaurar esta etapa visual: "
if /I not "%CONFIRM%"=="SI" (
  echo [INFO] Restauracion cancelada.
  exit /b 0
)

echo [INFO] Descargando SHA desde %CHECKPOINT_REMOTE%...
git fetch --no-tags "%CHECKPOINT_REMOTE%" "%CHECKPOINT_SHA%"
if errorlevel 1 goto :failed

set "FETCHED_SHA="
for /f "delims=" %%I in ('git rev-parse FETCH_HEAD') do set "FETCHED_SHA=%%I"
if /I not "%FETCHED_SHA%"=="%CHECKPOINT_SHA%" (
  echo [ERROR] GitHub devolvio un commit distinto al SHA esperado.
  goto :failed
)

echo [INFO] Preparando lista explicita de archivos frontend...
git ls-tree -rz --name-only "%CHECKPOINT_SHA%" -- frontend > "%TEMP_PATHS%"
if errorlevel 1 goto :failed

for %%I in ("%TEMP_PATHS%") do if %%~zI EQU 0 (
  echo [ERROR] El SHA no contiene archivos frontend versionados.
  goto :failed
)

echo [INFO] Restaurando frontend sin borrar contenido local adicional...
git restore --source="%CHECKPOINT_SHA%" --worktree --pathspec-from-file="%TEMP_PATHS%" --pathspec-file-nul
if errorlevel 1 goto :failed

call :cleanup
echo.
echo [OK] Etapa visual restaurada de forma conservadora.
echo [INFO] Revisa los cambios locales con: git status --short
git status --short
exit /b 0

:failed
call :cleanup
echo.
echo [ERROR] No se pudo restaurar la etapa visual.
exit /b 1

:cleanup
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference = 'Stop';" ^
  "$tempRoot = [IO.Path]::GetFullPath($env:TEMP).TrimEnd('\') + '\';" ^
  "$candidate = '%TEMP_PATHS%';" ^
  "if (-not [string]::IsNullOrWhiteSpace($candidate)) {" ^
  "  $full = [IO.Path]::GetFullPath($candidate);" ^
  "  if (-not $full.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) { throw 'Ruta temporal insegura: ' + $full };" ^
  "  if (Test-Path -LiteralPath $full) { Remove-Item -LiteralPath $full -Force };" ^
  "}"
exit /b 0
