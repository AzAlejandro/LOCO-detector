# Guia de GitHub Releases para LOCO Detector

Esta guia explica como preparar y publicar una version estable de LOCO Detector usando GitHub Releases, y como usar esa version en Windows o macOS sin instalar Python, Node ni dependencias manuales.

El objetivo es una release offline simple para usuario final. La persona que recibe la release deberia poder instalar Docker Desktop, descomprimir un ZIP, iniciar la aplicacion y luego importar su proyecto desde `Configuracion`.

El punto tecnico mas importante es este: el `docker-compose.yml` normal del repositorio usa `build:` y sirve para desarrollo. Para una release offline no basta entregar ese archivo junto con las imagenes Docker, porque intentaria reconstruir backend y frontend desde codigo fuente. La release final debe usar un archivo separado, `docker-compose.release.yml`, que use `image:` y no `build:`.

## 1. Que contiene una release correcta

LOCO Detector corre como dos servicios Docker:

```text
backend  -> API y procesamiento Python
frontend -> interfaz web servida por Nginx
```

Por eso la release debe incluir las dos imagenes Docker:

```text
loco-detector-backend:latest
loco-detector-frontend:latest
```

Tambien debe incluir un compose especial de release:

```text
docker-compose.release.yml
```

Ese compose no debe construir nada. Solo debe levantar las imagenes que el usuario ya cargo con `docker load`.

La carpeta final recomendada de una release es:

```text
LOCO-detector-v1.0.0-release/
|
|-- loco-detector-v1.0.0-docker-images.tar.gz
|-- docker-compose.release.yml
|-- README_USUARIO.md
|-- iniciar_loco_detector.bat
|-- detener_loco_detector.bat
|-- iniciar_loco_detector.command
|-- detener_loco_detector.command
`-- opcionalmente:
    `-- proyecto-exportado-desde-configuracion.zip
```

Los archivos `.bat` son para Windows. Los archivos `.command` son para macOS.

El ZIP de proyecto es opcional. Si se incluye, sirve para restaurar imagenes, scribbles, datasets, modelos y otros datos exportados desde la pestana `Configuracion`.

## 2. Archivos que existen hoy y archivos que hay que crear

El repositorio ya tiene estos archivos utiles para desarrollo:

```text
docker-compose.yml
Dockerfile.backend
Dockerfile.frontend
run_docker.bat
check_docker_ports.ps1
README.md
```

El archivo `run_docker.bat` actual sirve para trabajar desde el repositorio, porque ejecuta `docker compose up --build -d`. Eso esta bien para desarrollo, pero no es el lanzador ideal para una carpeta offline de usuario final.

Para una release final, hay que crear una vez estos archivos nuevos:

```text
docker-compose.release.yml
iniciar_loco_detector.bat
detener_loco_detector.bat
iniciar_loco_detector.command
detener_loco_detector.command
README_USUARIO.md
```

Esta guia muestra exactamente que debe contener cada uno.

## 3. Crear `docker-compose.release.yml`

Crea este archivo en la raiz del repositorio:

```yaml
services:
  backend:
    image: loco-detector-backend:latest
    container_name: loco-detector-backend
    environment:
      BACKEND_HOST: 0.0.0.0
      BACKEND_PORT: 8011
      DEV_RELOAD: "false"
    ports:
      - "8011:8011"
    volumes:
      - loco_outputs:/app/outputs
    healthcheck:
      test: ["CMD", "curl", "--fail", "--silent", "--show-error", "http://127.0.0.1:8011/api/health"]
      interval: 15s
      timeout: 5s
      start_period: 20s
      retries: 5
    restart: unless-stopped

  frontend:
    image: loco-detector-frontend:latest
    container_name: loco-detector-frontend
    ports:
      - "5178:80"
    depends_on:
      backend:
        condition: service_healthy
    restart: unless-stopped

volumes:
  loco_outputs:
```

La diferencia importante con `docker-compose.yml` es que aqui se usa `image:`. No aparece `build:`. Esto permite que el usuario final ejecute la aplicacion sin tener el codigo fuente.

## 4. Crear lanzador para Windows

Crea `iniciar_loco_detector.bat` en la raiz del repositorio con este contenido:

```bat
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
```

Crea tambien `detener_loco_detector.bat`:

```bat
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
```

## 5. Crear lanzador para macOS

Crea `iniciar_loco_detector.command` en la raiz del repositorio con este contenido:

```bash
#!/bin/bash
set -e

cd "$(dirname "$0")"

IMAGE_ARCHIVE="loco-detector-v1.0.0-docker-images.tar.gz"
IMAGE_TAR="loco-detector-v1.0.0-docker-images.tar"

echo "=========================================="
echo "Iniciando LOCO Detector"
echo "=========================================="
echo

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: Docker no esta instalado o no esta disponible."
  echo "Instala Docker Desktop y vuelve a ejecutar este archivo."
  read -n 1 -s -r -p "Presiona cualquier tecla para cerrar..."
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker Desktop no esta iniciado."
  echo "Abre Docker Desktop, espera que termine de iniciar y vuelve a ejecutar este archivo."
  read -n 1 -s -r -p "Presiona cualquier tecla para cerrar..."
  exit 1
fi

if [ ! -f "$IMAGE_ARCHIVE" ]; then
  echo "ERROR: No se encontro $IMAGE_ARCHIVE."
  echo "Este archivo debe estar en la misma carpeta que este .command."
  read -n 1 -s -r -p "Presiona cualquier tecla para cerrar..."
  exit 1
fi

if [ ! -f "docker-compose.release.yml" ]; then
  echo "ERROR: No se encontro docker-compose.release.yml."
  echo "Este archivo debe estar en la misma carpeta que este .command."
  read -n 1 -s -r -p "Presiona cualquier tecla para cerrar..."
  exit 1
fi

if [ ! -f "$IMAGE_TAR" ]; then
  echo "Descomprimiendo imagenes Docker..."
  tar -xzf "$IMAGE_ARCHIVE"
fi

echo "Cargando imagenes Docker..."
docker load -i "$IMAGE_TAR"

echo "Deteniendo contenedores anteriores si existen..."
docker compose -f docker-compose.release.yml down

echo "Levantando LOCO Detector..."
docker compose -f docker-compose.release.yml up -d

echo "Esperando backend..."
for i in {1..60}; do
  if curl --silent --fail "http://127.0.0.1:8011/api/health" >/dev/null 2>&1; then
    echo "LOCO Detector listo."
    echo "Frontend: http://localhost:5178"
    open "http://localhost:5178"
    read -n 1 -s -r -p "Presiona cualquier tecla para cerrar esta ventana..."
    exit 0
  fi
  sleep 2
done

echo "ERROR: El backend no respondio a tiempo."
docker compose -f docker-compose.release.yml logs --tail 80
read -n 1 -s -r -p "Presiona cualquier tecla para cerrar..."
exit 1
```

Crea tambien `detener_loco_detector.command`:

```bash
#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "Deteniendo LOCO Detector..."
docker compose -f docker-compose.release.yml down

echo
echo "LOCO Detector fue detenido."
echo "Los datos importados se mantienen en el volumen Docker."
read -n 1 -s -r -p "Presiona cualquier tecla para cerrar..."
```

Antes de distribuir en macOS, da permisos de ejecucion:

```bash
chmod +x iniciar_loco_detector.command
chmod +x detener_loco_detector.command
```

## 6. Crear `README_USUARIO.md`

Crea `README_USUARIO.md` en la raiz del repositorio con este contenido:

```markdown
# LOCO Detector - Uso rapido

Esta version permite ejecutar LOCO Detector sin instalar Python, Node ni dependencias manualmente.

## Requisito

Instala Docker Desktop:

https://www.docker.com/products/docker-desktop/

Despues de instalarlo, abre Docker Desktop y espera a que termine de iniciar.

## Windows

1. Descomprime `LOCO-detector-v1.0.0-release.zip`.
2. Entra a la carpeta descomprimida.
3. Haz doble click en `iniciar_loco_detector.bat`.
4. Espera a que se abra el navegador.
5. Si no se abre automaticamente, entra a `http://localhost:5178`.

Para cerrar LOCO Detector, haz doble click en `detener_loco_detector.bat`.

## macOS

1. Descomprime `LOCO-detector-v1.0.0-release.zip`.
2. Entra a la carpeta descomprimida.
3. Si macOS no permite abrir el archivo `.command`, abre Terminal en esa carpeta y ejecuta:

```bash
chmod +x iniciar_loco_detector.command
chmod +x detener_loco_detector.command
```

4. Haz doble click en `iniciar_loco_detector.command`.
5. Espera a que se abra el navegador.
6. Si no se abre automaticamente, entra a `http://localhost:5178`.

Para cerrar LOCO Detector, haz doble click en `detener_loco_detector.command`.

## Como cargar un proyecto

Si recibiste un archivo como `proyecto-exportado-desde-configuracion.zip`, abre LOCO Detector y entra a:

`Configuracion` -> `Exportar e importar`

Luego usa la opcion de importar proyecto.

## Si la interfaz aparece vacia

No significa que Docker haya fallado. Significa que la aplicacion abrio correctamente, pero todavia no has importado un proyecto.
```

## 7. Preparar una release nueva

Abre PowerShell en Windows o Terminal en macOS y entra a la raiz del repositorio.

En Windows:

```powershell
cd "C:\Users\alejo\Documents\GitHub\LOCO-detector"
```

En macOS, usa la ruta donde tengas clonado el repo:

```bash
cd ~/Documents/GitHub/LOCO-detector
```

Verifica que Docker Desktop este abierto:

```bash
docker info
```

Prueba que la app funciona desde el compose de desarrollo:

```bash
docker compose up --build -d
```

Abre:

```text
http://localhost:5178
```

Verifica que la interfaz abre y que puedes entrar a `Configuracion`. Luego detiene:

```bash
docker compose down
```

Construye las imagenes limpias:

```bash
docker compose build
```

Confirma que existen:

```bash
docker images
```

Debes ver algo equivalente a:

```text
loco-detector-backend    latest
loco-detector-frontend   latest
```

## 8. Exportar backend y frontend en un solo archivo

Crea la carpeta de salida:

En Windows:

```powershell
mkdir release-assets
```

En macOS:

```bash
mkdir -p release-assets
```

Exporta ambas imagenes:

```bash
docker save -o release-assets/loco-detector-v1.0.0-docker-images.tar loco-detector-backend:latest loco-detector-frontend:latest
```

Comprime el `.tar`:

```bash
tar -czf release-assets/loco-detector-v1.0.0-docker-images.tar.gz -C release-assets loco-detector-v1.0.0-docker-images.tar
```

El archivo importante sera:

```text
release-assets/loco-detector-v1.0.0-docker-images.tar.gz
```

## 9. Armar la carpeta final de release

En Windows:

```powershell
mkdir release-assets\LOCO-detector-v1.0.0-release
copy release-assets\loco-detector-v1.0.0-docker-images.tar.gz release-assets\LOCO-detector-v1.0.0-release\
copy docker-compose.release.yml release-assets\LOCO-detector-v1.0.0-release\
copy README_USUARIO.md release-assets\LOCO-detector-v1.0.0-release\
copy iniciar_loco_detector.bat release-assets\LOCO-detector-v1.0.0-release\
copy detener_loco_detector.bat release-assets\LOCO-detector-v1.0.0-release\
copy iniciar_loco_detector.command release-assets\LOCO-detector-v1.0.0-release\
copy detener_loco_detector.command release-assets\LOCO-detector-v1.0.0-release\
```

En macOS:

```bash
mkdir -p release-assets/LOCO-detector-v1.0.0-release
cp release-assets/loco-detector-v1.0.0-docker-images.tar.gz release-assets/LOCO-detector-v1.0.0-release/
cp docker-compose.release.yml release-assets/LOCO-detector-v1.0.0-release/
cp README_USUARIO.md release-assets/LOCO-detector-v1.0.0-release/
cp iniciar_loco_detector.bat release-assets/LOCO-detector-v1.0.0-release/
cp detener_loco_detector.bat release-assets/LOCO-detector-v1.0.0-release/
cp iniciar_loco_detector.command release-assets/LOCO-detector-v1.0.0-release/
cp detener_loco_detector.command release-assets/LOCO-detector-v1.0.0-release/
```

Si quieres incluir un proyecto exportado desde `Configuracion`, copialo tambien dentro de esa carpeta.

Comprime la carpeta final:

En Windows:

```powershell
Compress-Archive -Path release-assets\LOCO-detector-v1.0.0-release -DestinationPath release-assets\LOCO-detector-v1.0.0-release.zip -Force
```

En macOS:

```bash
cd release-assets
zip -r LOCO-detector-v1.0.0-release.zip LOCO-detector-v1.0.0-release
cd ..
```

El archivo que se sube a GitHub Releases es:

```text
release-assets/LOCO-detector-v1.0.0-release.zip
```

## 10. Probar la release antes de publicarla

Antes de subirla, prueba el ZIP como si fueras usuario final.

En Windows:

```powershell
mkdir C:\LOCO-release-test
copy release-assets\LOCO-detector-v1.0.0-release.zip C:\LOCO-release-test\
```

Si `C:\LOCO-release-test` ya existe, PowerShell puede mostrar un aviso. No es un problema. Lo importante es que el ZIP quede dentro de esa carpeta.

Luego entra a la carpeta de prueba:

```powershell
cd C:\LOCO-release-test
```

Descomprime el ZIP:

```powershell
Expand-Archive -Path .\LOCO-detector-v1.0.0-release.zip -DestinationPath . -Force
```

Entra a la carpeta descomprimida:

```powershell
cd .\LOCO-detector-v1.0.0-release
```

Ejecuta el iniciador. En PowerShell debes anteponer `.\` para ejecutar un archivo de la carpeta actual:

```text
.\iniciar_loco_detector.bat
```

En macOS, crea una carpeta de prueba:

```bash
mkdir -p ~/LOCO-release-test
cp release-assets/LOCO-detector-v1.0.0-release.zip ~/LOCO-release-test/
cd ~/LOCO-release-test
```

Descomprime el ZIP:

```bash
unzip -o LOCO-detector-v1.0.0-release.zip
```

Entra a la carpeta descomprimida:

```bash
cd LOCO-detector-v1.0.0-release
```

Da permisos de ejecucion al iniciador y ejecutalo:

```bash
chmod +x iniciar_loco_detector.command
./iniciar_loco_detector.command
```

La prueba es correcta si ocurre esto:

```text
1. Docker carga las imagenes.
2. Docker levanta backend y frontend.
3. Se abre http://localhost:5178.
4. La app permite entrar a Configuracion.
5. Puedes importar un ZIP de proyecto si tienes uno.
```

Para cerrar:

En Windows:

```powershell
.\detener_loco_detector.bat
```

En macOS:

```bash
./detener_loco_detector.command
```

## 11. Crear tag y publicar en GitHub Releases

Cuando la release ya este probada:

Antes de ejecutar `git add .`, revisa que los artefactos pesados de release no entren al repositorio. Los archivos como estos no deben versionarse con Git:

```text
release-assets/
LOCO-detector-v1.0.0-release.zip
loco-detector-v1.0.0-docker-images.tar
loco-detector-v1.0.0-docker-images.tar.gz
```

Esos archivos se suben despues como assets de GitHub Releases, no como archivos del repositorio.

El `.gitignore` del proyecto debe incluir reglas especificas como estas:

```gitignore
release-assets/
LOCO-detector-v*-release.zip
loco-detector-v*-docker-images.tar
loco-detector-v*-docker-images.tar.gz
```

Usa reglas especificas para no bloquear cualquier `.zip` que en el futuro si pudiera ser parte valida de la documentacion o de tests.

```bash
git add .
git commit -m "Prepare release v1.0.0"
git tag v1.0.0
git push origin main
git push origin v1.0.0
```

Si ya hiciste commit antes, no repitas commits innecesarios. Lo importante es que el tag apunte al estado correcto.

Luego en GitHub:

1. Entra al repositorio.
2. Ve a `Releases`.
3. Haz click en `Draft a new release`.
4. Selecciona el tag `v1.0.0`.
5. Usa el titulo `LOCO Detector v1.0.0`.
6. Adjunta `LOCO-detector-v1.0.0-release.zip`.
7. Publica la release.

Notas recomendadas:

```text
LOCO Detector v1.0.0

Version estable para ejecucion local con Docker Desktop.

Incluye:
- Backend de LOCO Detector
- Frontend de LOCO Detector
- Imagenes Docker exportadas
- Lanzador para Windows
- Lanzador para macOS
- Guia rapida de usuario

Requisitos:
- Docker Desktop instalado e iniciado

Uso rapido:
1. Descargar LOCO-detector-v1.0.0-release.zip
2. Descomprimir el ZIP
3. Abrir Docker Desktop
4. En Windows, ejecutar iniciar_loco_detector.bat
5. En macOS, ejecutar iniciar_loco_detector.command
6. Abrir http://localhost:5178 si no se abre automaticamente
7. Importar el proyecto ZIP desde Configuracion si corresponde
```

## 12. Que no necesita el usuario final

El usuario final no necesita:

```text
Dockerfile.backend
Dockerfile.frontend
backend/
frontend/
requirements.txt
package.json
node_modules/
venv/
.git/
docker-compose.yml de desarrollo
```

Tampoco necesita ejecutar:

```text
git clone
pip install
npm install
docker compose build
```

## 13. Problemas comunes

Si Docker no esta instalado, el lanzador mostrara un error indicando que Docker no esta disponible. La solucion es instalar Docker Desktop, abrirlo y volver a ejecutar el lanzador.

Si Docker Desktop esta cerrado, el lanzador indicara que Docker no esta iniciado. La solucion es abrir Docker Desktop, esperar a que termine de iniciar y volver a ejecutar el lanzador.

Si los puertos `8011` o `5178` estan ocupados, cierra otras versiones de LOCO Detector. Si no sabes que proceso ocupa los puertos, reinicia el computador, abre Docker Desktop y ejecuta el lanzador nuevamente.

Si la interfaz aparece vacia, no significa que la aplicacion fallo. Significa que todavia no se ha importado un proyecto. Entra a `Configuracion` -> `Exportar e importar` e importa el ZIP del proyecto.

## 14. Checklist antes de publicar

Antes de subir la release, confirma:

```text
[ ] Docker Desktop esta abierto.
[ ] docker compose build termina sin errores.
[ ] Existen loco-detector-backend:latest y loco-detector-frontend:latest.
[ ] docker save exporta ambas imagenes.
[ ] El archivo .tar.gz se genera correctamente.
[ ] docker-compose.release.yml no usa build:.
[ ] iniciar_loco_detector.bat funciona en Windows.
[ ] iniciar_loco_detector.command funciona en macOS o fue validado con Terminal.
[ ] http://localhost:5178 abre.
[ ] detener_loco_detector.bat o detener_loco_detector.command detiene la app.
[ ] El ZIP final contiene todos los archivos necesarios.
[ ] El usuario final no necesita codigo fuente.
```

## 15. Resumen del metodo correcto

El metodo correcto para usuarios finales es una release binaria offline con Docker:

```text
LOCO-detector-v1.0.0-release.zip
```

Ese ZIP debe contener:

```text
LOCO-detector-v1.0.0-release/
|
|-- loco-detector-v1.0.0-docker-images.tar.gz
|-- docker-compose.release.yml
|-- README_USUARIO.md
|-- iniciar_loco_detector.bat
|-- detener_loco_detector.bat
|-- iniciar_loco_detector.command
|-- detener_loco_detector.command
`-- opcionalmente:
    `-- proyecto-exportado-desde-configuracion.zip
```

Ese ZIP es el archivo principal que se sube a GitHub Releases.
