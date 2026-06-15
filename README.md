<div align="center">

# LOCO Detector

**Detección automática de círculos y medición de diámetros en imágenes TEM de nanofibras**

![Version](https://img.shields.io/badge/version-v1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightblue)

</div>

## 📸 Demo

<!--
Coloca aqui una captura real cuando exista:
docs/assets/demo.png

Ejemplo futuro, cuando exista la captura:
usar una imagen Markdown apuntando a docs/assets/demo.png
-->

LOCO Detector es una herramienta de análisis científico para imágenes TEM de nanofibras. Permite segmentar regiones de soporte con scribbles, detectar secciones circulares, medir diámetros en px/nm/um y revisar distribuciones estadísticas. El flujo combina modelos ML como Random Forest, ExtraTrees, XGBoost, CatBoost y LightGBM con una interfaz local orientada a investigación.

## 📋 Tabla de Contenidos

- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [🐳 Docker](#-docker)
- [⚙️ Instalación local](#️-instalación-local)
- [🖥️ Uso e interfaz](#️-uso-e-interfaz)
- [🔬 Pipeline técnico](#-pipeline-técnico)
- [📡 API Reference](#-api-reference)
- [🧪 Tests](#-tests)
- [🗂️ Estructura de salida](#️-estructura-de-salida)
- [🔒 Seguridad](#-seguridad)
- [📝 Notas de mantenimiento](#-notas-de-mantenimiento)
- [📄 Licencia](#-licencia)

## ✨ Features

| Área | Funcionalidad |
|---|---|
| 🖌️ Scribbles | Segmentación interactiva foreground / halo / background con editor visual, borrador persistente y filtros visuales no destructivos. |
| 🔵 Detección ML | Entrenamiento y uso de modelos Random Forest, ExtraTrees, XGBoost, CatBoost y LightGBM para candidatos circulares. |
| 📏 Diámetros | Medición automática por metodologías geométricas y basadas en máscara, con puntos, líneas y círculos de medición. |
| 📐 Calibración | Conversión px -> nm/um por imagen mediante línea de escala guardada. |
| 📊 Histogramas | Métricas por imagen y globales, histograma configurable, exportación CSV/JSON/SVG. |
| 🗂️ Datasets | Generación de dataset LOCO, augmentation, entrenamiento, batch y gestión de modelos. |
| ✅ Validación | Revisión manual de resultados, tiers, decisiones y exportación de reportes. |
| 🧪 LOCO Lab | Flujo experimental proposal -> measure -> evaluate para investigación avanzada. |
| 🏷️ Proyectos y tags | Organización por proyecto, tags estructurados, filtros por proyecto/tags y exportación selectiva. |
| 🐳 Distribución | Stack Docker CPU-only para Windows, macOS y Linux, con restauración de trabajo mediante ZIP. |

## 🚀 Quick Start

### Opción A — Docker, recomendado para usuarios

En Windows, desde la raíz del repositorio:

```bat
run_docker.bat
```

En cualquier sistema con Docker Compose:

```bash
docker compose up --build -d
```

Abre la aplicación en:

```text
http://localhost:5178
```

Si la aplicación se ve vacía en una máquina nueva, es normal. El código y el entorno viajan en Docker; el trabajo pesado se restaura desde `Configuracion > Exportar e importar` usando el ZIP exportado previamente.

### Opción B — Local, para desarrollo

```bash
git clone https://github.com/AzAlejandro/LOCO-detector.git
cd LOCO-detector
```

En Windows:

```bat
run_local.bat
```

Inicio manual:

```bash
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt

cd frontend
npm install
npm run dev
```

En otra terminal, desde la raíz:

```bash
venv/Scripts/python app.py
```

Por defecto, local usa frontend `http://localhost:5173` y backend `http://127.0.0.1:8011`.

## 🐳 Docker

### Requisitos

| Requisito | Detalle |
|---|---|
| Docker Desktop | Debe estar instalado e iniciado antes de ejecutar Compose. |
| Docker Compose | Incluido en Docker Desktop moderno. |
| Puertos libres | Frontend `5178`, backend `8011` por defecto. |
| ZIP de proyecto | Opcional, pero necesario para restaurar imágenes, modelos, runs y análisis existentes. |

### Iniciar

```bash
docker compose up --build -d
```

### Detener sin borrar trabajo importado

```bash
docker compose down
```

### Borrar también el volumen persistente

```bash
docker compose down -v
```

Usa `-v` solo si quieres empezar desde cero. El volumen `loco_outputs` guarda los datos importados dentro del entorno Docker.

### Restaurar trabajo desde ZIP

1. Abre `http://localhost:5178`.
2. Entra a `Configuracion > Exportar e importar`.
3. Selecciona el ZIP exportado desde otra instalación.
4. Toca `Revisar contenido`.
5. Importa conservando existentes o sobrescribiendo conflictos.
6. Espera el progreso y recarga la interfaz cuando el modal lo indique.

Para publicar y consumir versiones con GitHub Releases en Windows y macOS, usa la guía completa: [docs/GITHUB_RELEASES_WINDOWS_MAC.md](docs/GITHUB_RELEASES_WINDOWS_MAC.md).

## ⚙️ Instalación local

### Requisitos

| Herramienta | Versión mínima |
|---|---|
| Python | 3.10+ |
| Node.js | 18+ |
| npm | 9+ |

### Instalación paso a paso

```bash
git clone https://github.com/AzAlejandro/LOCO-detector.git
cd LOCO-detector
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
cd frontend
npm install
cd ..
```

### Arranque local en Windows

```bat
run_local.bat
```

### Arranque local manual

Terminal 1:

```bash
venv/Scripts/python app.py
```

Terminal 2:

```bash
cd frontend
npm run dev -- --port 5173 --host localhost
```

### Configuración de puertos

Los puertos se leen desde `.env` en la raíz del repo. No subas `.env` a Git.

```env
BACKEND_PORT=8011
BACKEND_HOST=127.0.0.1
FRONTEND_PORT=5173
FRONTEND_HOST=localhost
DEV_RELOAD=true
VITE_API_BASE=http://127.0.0.1:8011
```

Para diagnosticar puertos ocupados:

```powershell
.\tools\diagnose_ports.ps1
```

Para detener procesos locales del proyecto:

```powershell
.\tools\stop_servers.ps1
```

## 🖥️ Uso e interfaz

| Grupo | Sub-tabs principales | Función |
|---|---|---|
| **Proyecto** | Seleccion proyecto, Imagenes y tags, Manejo de tags | Define proyecto activo, tags, rutas informativas y organización de biblioteca. |
| **Entrenamiento / Scribble** | Scribbles y Experimentos, Revision de Resultados, Modelos de Asistencia, Gestion de Modelos | Carga imágenes, dibuja scribbles, ejecuta experimentos y administra modelos de asistencia. |
| **Entrenamiento / LOCO** | Generar Dataset, Aumentacion, Entrenamiento, Test de Modelo, Gestion de Modelos | Construye datasets, genera aumentaciones, entrena y gestiona modelos LOCO. |
| **Produccion / Deteccion** | Detector LOCO, Medicion de Diametros | Detecta candidatos, filtra círculos, envía aceptados a medición y calibra escala. |
| **Produccion / Analisis** | Seleccion de imagenes, Histograma | Agrupa mediciones internas, filtra por tags/proyecto y exporta análisis. |
| **Otros** | Configuracion, Tutorial | Exporta/importa proyectos y lanza recorridos guiados con Driver.js. |

### Flujo recomendado

1. Crea o activa un proyecto en `Proyecto`.
2. Carga imágenes desde navegador en `Scribbles y Experimentos`.
3. Dibuja scribbles y ejecuta experimentos para generar máscaras.
4. Revisa resultados y entrena modelos de asistencia si corresponde.
5. Genera dataset LOCO, aumenta datos y entrena modelos LOCO.
6. Usa `Detector LOCO` sobre imágenes nuevas.
7. Mide diámetros, calibra escala y guarda mediciones internas.
8. Analiza histogramas por proyecto/tags.
9. Exporta el proyecto a ZIP para transportarlo.

## 🔬 Pipeline técnico

```text
Image -> Support Region -> Candidate Grid + Radii -> Feature Extraction
    -> Binary Model -> Multiclass Model -> Combined Filter
    -> Circle-NMS -> Spatial Filter -> Accepted Circles
    -> Diameter Measurement -> Calibration -> Distribution
```

| Paso | Descripción |
|---|---|
| Image | Imagen TEM cargada desde navegador o biblioteca. |
| Support Region | Región de soporte derivada desde máscara/scribbles aprobados. |
| Candidate Grid + Radii | Muestreo de centros y radios candidatos. |
| Feature Extraction | Extracción de patches, intensidad, gradientes y variables geométricas. |
| Binary Model | Clasificación inicial círculo válido / no válido. |
| Multiclass Model | Clasificación valid / crossing / other. |
| Combined Filter | Fusión de umbrales y criterios de aceptación. |
| Circle-NMS | Supresión de círculos redundantes mediante IoU/distancia. |
| Spatial Filter | Filtro espacial posterior para reducir falsos positivos. |
| Accepted Circles | Candidatos aceptados para revisión o medición. |
| Diameter Measurement | Medición por círculo, línea o máscara según método. |
| Calibration | Conversión px -> nm/um con escala por imagen. |
| Distribution | Métricas e histogramas exportables. |

## 📡 API Reference

La API corre por defecto en `http://127.0.0.1:8011`. En Docker se publica en el puerto configurado por `DOCKER_BACKEND_PORT`, normalmente `8011`.

### Session & Image

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/session/new` | Create new session |
| POST | `/api/image/load` | Load image into session |
| GET | `/api/library/images` | List saved images |
| POST | `/api/library/load` | Load saved image |
| POST | `/api/library/delete` | Delete saved image |
| POST | `/api/local-images/upload-browser` | Upload browser-selected image |

### Projects & Tags

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/projects/list` | List projects and active project |
| POST | `/api/projects/upsert` | Create or update project |
| POST | `/api/projects/activate` | Activate project |
| POST | `/api/projects/delete` | Delete project metadata |
| GET | `/api/projects/images` | List images by project view |
| GET | `/api/projects/tag-catalog` | List structured tag catalog |
| POST | `/api/projects/image/update` | Update image project/tags |

### Scribbles

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/scribble/draft/save` | Save scribble draft |
| GET | `/api/scribble/draft/load` | Load scribble draft |
| POST | `/api/scribble/draft/clear` | Clear scribble draft |

### Experiments & Review

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/experiments/catalog` | List available experiments |
| POST | `/api/experiments/run` | Run single experiment |
| POST | `/api/experiments/run-batch` | Run batch experiments |
| GET | `/api/results/list` | List experiment results |
| GET | `/api/results/get` | Get result details |
| GET | `/api/results/mask-thumb` | Get mask thumbnail |
| POST | `/api/review/mark` | Mark run decision/tier |
| GET | `/api/review/list` | List review metadata |
| GET | `/api/reports/export` | Export reports |

### Assist Models

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/assist-models/dataset/images` | List trainable images |
| GET | `/api/assist-models/list` | List assist models |
| POST | `/api/assist-models/train` | Train assist model |
| POST | `/api/assist-models/set-default` | Set default assist model |
| POST | `/api/assist-models/update-meta` | Update model metadata |
| POST | `/api/assist-models/delete` | Delete assist model |
| POST | `/api/assist-models/predict` | Predict class probabilities |
| POST | `/api/assist-models/predict-mask` | Predict mask |

### Diameter Research

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/diameter-research/points/update` | Update diameter points |
| POST | `/api/diameter-research/points/save` | Save diameter points |
| GET | `/api/diameter-research/points/load` | Load diameter points |
| GET | `/api/diameter-research/points/list` | List point files |
| POST | `/api/diameter-research/run` | Run diameter measurement |
| GET | `/api/diameter-research/results/list` | List diameter results |
| GET | `/api/diameter-research/results/get` | Get diameter result details |
| GET | `/api/diameter-research/reports/export` | Export report |

### LOCO Dataset, Training & Detector

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/diameter-research/loco-dataset/circles/load` | Load LOCO dataset circles |
| POST | `/api/diameter-research/loco-dataset/circles/sync` | Sync LOCO dataset circles |
| POST | `/api/diameter-research/loco-dataset/features` | Extract dataset features |
| POST | `/api/diameter-research/loco-dataset/save` | Save LOCO dataset |
| POST | `/api/diameter-research/loco-dataset/augment/preview` | Preview augmentation |
| POST | `/api/diameter-research/loco-dataset/augment/apply` | Apply augmentation |
| POST | `/api/diameter-research/loco-training/train` | Train LOCO models |
| POST | `/api/diameter-research/loco-training/tune` | Tune LOCO model |
| GET | `/api/diameter-research/loco-training/saved-models` | List saved LOCO models |
| POST | `/api/diameter-research/loco-training/save-model` | Save LOCO model |
| POST | `/api/diameter-research/loco-training/test-circles` | Test circles with model |
| POST | `/api/diameter-research/loco-models/detect-base` | Run base detector |
| POST | `/api/diameter-research/loco-models/apply-threshold` | Apply threshold |
| POST | `/api/diameter-research/loco-models/apply-nms` | Apply NMS |
| POST | `/api/diameter-research/loco-models/apply-spatial` | Apply spatial filter |
| POST | `/api/diameter-research/loco-models/measure-accepted` | Measure accepted circles |

### Calibration & Analysis

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/diameter-research/calibration/save` | Save scale calibration |
| GET | `/api/diameter-research/calibration/load` | Load scale calibration |
| GET | `/api/diameter-research/measurements/summary-by-image` | Measurement summary per image |
| POST | `/api/diameter-research/measurements/save-from-run` | Save internal measurements |
| GET | `/api/diameter-research/measurements/query` | Query measurements |
| GET | `/api/diameter-research/analysis/list` | List saved analyses |
| POST | `/api/diameter-research/analysis/save` | Save analysis |
| GET | `/api/diameter-research/analysis/export/download` | Download analysis export |

### Project Transfer

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/project-transfer/catalog` | Inspect exportable categories |
| POST | `/api/project-transfer/export/prepare` | Prepare export ZIP |
| GET | `/api/project-transfer/export/download` | Download export ZIP |
| POST | `/api/project-transfer/import/inspect` | Inspect import ZIP |
| POST | `/api/project-transfer/import/start` | Start import job |
| GET | `/api/project-transfer/import/progress` | Poll import progress |
| POST | `/api/project-transfer/import/apply` | Legacy synchronous import |

### Validation

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/diameter-research/validation/case/upsert` | Create/update validation case |
| GET | `/api/diameter-research/validation/cases` | List validation cases |
| POST | `/api/diameter-research/validation/run-case` | Run validation case |
| GET | `/api/diameter-research/validation/export` | Export validation results |

## 🧪 Tests

```bash
pytest tests/ -v
```

| Test file | Tests | Description |
|---|---:|---|
| `test_api.py` | 1 | Full API flow: session -> image -> run -> review -> export. |
| `test_calibration.py` | 4 | Calibration save/load/delete and safe image IDs. |
| `test_core.py` | 3 | Experiment registry, scribble normalization and draft persistence. |
| `test_diameter_research.py` | 17 | Diameter pipelines, LOCO probes, persistence and API flow. |
| `test_distribution.py` | 18 | Statistics, binning, CSV export and integration. |
| `test_loco_to_diameter.py` | 4 | LOCO accepted circles converted into measurement points. |
| `test_navigation.py` | 10 | Navigation mapping and group structure. |
| `test_project_transfer.py` | 3 | Export/import conflicts, unsafe paths and manifest validation. |

## 🗂️ Estructura de salida

La distribución Docker persiste el trabajo en `outputs/`. Algunos flujos históricos y tests también usan `data/`.

```text
data/
├── calibration/          # Scale calibration files (*.json)
├── images/               # Loaded images
├── runs/                 # Diameter measurement runs
├── loco_dataset/         # LOCO training dataset
├── loco_training/        # Trained models
└── reports/              # Exported reports
```

```text
outputs/scribble_research/
├── library/              # Copias registradas de imágenes y metadata
├── projects/             # Proyectos, tags y proyecto activo
├── drafts/               # Scribbles guardados
├── runs/                 # Experimentos Scribble completos
├── assist_models/        # Modelos de asistencia Scribble
└── diameter_research/    # Dataset LOCO, modelos LOCO, mediciones y análisis
```

## 🔒 Seguridad

- **Localhost-only binding**: la API local usa `127.0.0.1:8011` por defecto; en Docker se publica solo en los puertos configurados.
- **Input validation**: los endpoints validan campos requeridos y devuelven errores controlados.
- **Path sanitization**: las rutas de importación/exportación se sanitizan para evitar traversal.
- **Session isolation**: cada sesión mantiene estado aislado.
- **CORS**: el frontend autorizado se limita al origen configurado.
- **Dependency scanning**: se recomienda revisar dependencias con `pip audit` y `npm audit`.
- **Datos pesados fuera de Git**: `outputs/`, `img/`, artefactos de release y `.env*` están ignorados.

Ver [SECURITY.md](SECURITY.md) para detalles.

## 📝 Notas de mantenimiento

Los archivos de texto del repositorio deben mantenerse en `UTF-8` sin BOM. Ese estándar soporta `ñ`, tildes y símbolos técnicos sin romper Python, JavaScript, Markdown, Docker ni scripts.

Para auditar encodings:

```bash
python scripts/encoding_audit.py
```

Este proyecto usa versionado semántico. Ver [CHANGELOG.md](CHANGELOG.md) para historial de cambios.

## 📄 Licencia

Este proyecto está bajo la licencia **MIT**. Ver el archivo [LICENSE](LICENSE) para más detalles.
