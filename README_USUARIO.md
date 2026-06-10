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