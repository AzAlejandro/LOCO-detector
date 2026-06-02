@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "CHECKPOINT_REMOTE=origin"
set "CHECKPOINT_SHA=7c0c9bdad809e6d4e28a3e305dfa480587898bdb"
set "TEMP_PATHS=%TEMP%\loco_detector_checkpoint_paths_%RANDOM%_%RANDOM%.txt"

echo ============================================================
echo LOCO Detector - Restaurar checkpoint conservador
echo ============================================================
echo Checkpoint: %CHECKPOINT_SHA%
echo.
echo Este proceso:
echo   - Descarga el checkpoint exacto desde GitHub.
echo   - Restaura solo archivos versionados presentes en ese checkpoint.
echo   - Conserva outputs, imagenes, .env, datos locales y archivos adicionales.
echo   - No elimina archivos, no cambia de rama y no ejecuta git reset.
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

git diff --cached --quiet
if errorlevel 2 (
  echo [ERROR] Git no pudo revisar el indice local.
  exit /b 1
)
if errorlevel 1 (
  echo [ERROR] Hay cambios preparados para commit ^(staged^).
  echo Haz commit o quita esos cambios del stage antes de restaurar.
  echo El script se detiene para no dejar el indice Git en un estado ambiguo.
  exit /b 1
)

echo [ADVERTENCIA] Los cambios locales sin commit en archivos versionados se sobrescribiran.
set /p "CONFIRM=Escribe SI para restaurar este checkpoint: "
if /I not "%CONFIRM%"=="SI" (
  echo [INFO] Restauracion cancelada.
  exit /b 0
)

echo [INFO] Descargando checkpoint desde %CHECKPOINT_REMOTE%...
git fetch --no-tags "%CHECKPOINT_REMOTE%" "%CHECKPOINT_SHA%"
if errorlevel 1 goto :failed

set "FETCHED_SHA="
for /f "delims=" %%I in ('git rev-parse FETCH_HEAD') do set "FETCHED_SHA=%%I"
if /I not "%FETCHED_SHA%"=="%CHECKPOINT_SHA%" (
  echo [ERROR] GitHub devolvio un commit distinto al checkpoint esperado.
  goto :failed
)

echo [INFO] Preparando lista explicita de archivos versionados...
git ls-tree -rz --name-only "%CHECKPOINT_SHA%" > "%TEMP_PATHS%"
if errorlevel 1 goto :failed

echo [INFO] Restaurando archivos sin borrar contenido local adicional...
git restore --source="%CHECKPOINT_SHA%" --worktree --pathspec-from-file="%TEMP_PATHS%" --pathspec-file-nul
if errorlevel 1 goto :failed

set "CURRENT_HEAD="
for /f "delims=" %%I in ('git rev-parse HEAD') do set "CURRENT_HEAD=%%I"
if /I "%CURRENT_HEAD%"=="%CHECKPOINT_SHA%" (
  rem Reconcile Windows CRLF stat metadata without staging a rollback.
  git add -u --pathspec-from-file="%TEMP_PATHS%" --pathspec-file-nul
  if errorlevel 1 goto :failed
  git diff --cached --quiet
  if errorlevel 1 (
    echo [ERROR] La reconciliacion local genero cambios staged inesperados.
    goto :failed
  )
)

call :cleanup
echo.
echo [OK] Checkpoint restaurado de forma conservadora.
echo [INFO] Archivos locales adicionales y datos pesados fueron conservados.
echo [INFO] Revisa los cambios locales con: git status --short
git status --short
exit /b 0

:failed
call :cleanup
echo.
echo [ERROR] No se pudo restaurar el checkpoint.
exit /b 1

:cleanup
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference = 'Stop';" ^
  "$tempRoot = [IO.Path]::GetFullPath($env:TEMP).TrimEnd('\') + '\';" ^
  "foreach ($candidate in @('%TEMP_PATHS%')) {" ^
  "  if ([string]::IsNullOrWhiteSpace($candidate)) { continue };" ^
  "  $full = [IO.Path]::GetFullPath($candidate);" ^
  "  if (-not $full.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) { throw 'Ruta temporal insegura: ' + $full };" ^
  "  if (Test-Path -LiteralPath $full) { Remove-Item -LiteralPath $full -Recurse -Force };" ^
  "}"
exit /b 0
