# LOCO Detector

**Automated circle detection and diameter measurement for TEM nanofiber images.**

LOCO Detector is a standalone, installable program that detects circular cross-sections of nanofibers in Transmission Electron Microscopy (TEM) images using machine learning. It provides a complete pipeline from image loading → scribble-based segmentation → circle detection → diameter measurement → scale calibration → statistical distribution.

---

## Features

- **Scribble-based segmentation** — Interactive foreground/background labeling with real-time feedback
- **LOCO Circle Detection** — ML-powered circle detection using Random Forest, ExtraTrees, XGBoost, CatBoost, and LightGBM
- **Diameter measurement** — Automatic diameter measurement along detected circles using multiple methodologies (v1, v2, v3)
- **Scale calibration** — Convert pixels to nanometers/micrometers with persistent calibration per image
- **Statistical distribution** — Interactive histogram with mean, median, std, min, max, and CSV export
- **Hierarchical UI** — 3 logical groups with sub-tabs for organized workflow
- **Validation system** — Manual review and validation of detected circles
- **Dataset management** — Build, augment, and manage training datasets for custom models
- **Experimental LOCO Lab** — Advanced proposal/measure/evaluate pipeline for research

---

## Quick Start

## Text Encoding

Repository text files should use `UTF-8` without BOM. This supports `ñ` and
accented characters while staying compatible with Python, JavaScript, Markdown,
Docker, and shell tooling.

### Docker distribution

For sharing the application across machines, Docker is the recommended
distribution path. The Docker images contain source code and CPU-only
dependencies, but never preload the heavy project work stored under `outputs/`.

For a complete step-by-step publication and usage guide with GitHub Releases on
Windows and macOS, see [docs/GITHUB_RELEASES_WINDOWS_MAC.md](docs/GITHUB_RELEASES_WINDOWS_MAC.md).

Requirements:

- Docker Desktop with Docker Compose
- the project ZIP exported from `Configuracion` when existing training work
  must be restored

On Windows, run:

```bat
run_docker.bat
```

Or use Docker Compose directly:

```bash
docker compose up --build -d
```

Open [http://localhost:5178](http://localhost:5178). On a new machine, enter
`Configuracion`, inspect the exported ZIP and import it. Docker keeps imported
files in a local named volume so they survive container restarts; the ZIP
remains the official mechanism for moving work between machines.

Stop the containers without deleting imported local work:

```bash
docker compose down
```

The Windows-only native folder picker is not available from the Linux
container. Use browser uploads and `Configuracion` project ZIP import when
running the Docker distribution.

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm 9+

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/AzAlejandro/LOCO-detector.git
cd LOCO-detector

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install frontend dependencies
cd frontend
npm install
cd ..
```

### Run locally

```bash
# Option A: Using the batch script (Windows)
run_local.bat

# Option B: Manual startup
# Terminal 1 — Backend (port 8011)
python app.py

# Terminal 2 — Frontend (port 5173)
cd frontend && npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

### Stop servers

```powershell
# Windows PowerShell
.\stop_servers.ps1
```

### Port configuration

Ports are configured in the root `.env` file:

```
BACKEND_PORT=8011
BACKEND_HOST=127.0.0.1
FRONTEND_PORT=5173
FRONTEND_HOST=localhost
DEV_RELOAD=true
VITE_API_BASE=http://127.0.0.1:8011
```

To change the backend port, edit `BACKEND_PORT` and `VITE_API_BASE` in `.env`. The frontend reads `VITE_API_BASE` from `frontend/.env`, which is automatically synced by `run_local.bat` and `run_silent.vbs`.

### Diagnose ports

```powershell
.\diagnose_ports.ps1
```

This read-only script shows all processes using the configured backend and frontend ports. It never kills any process.

---

## UI Navigation (3 hierarchical groups)

| Level 1 (Group) | Level 2 (Sub-tabs) | Purpose |
|---|---|---|
| **Grupo 1: Scribbles → Modelo** | Scribbles, Run, Review | Segmentation + diameter measurement |
| **Grupo 2: Dataset → LOCO Model** | Dataset, Training, Test | Build datasets, train ML models, test |
| **Grupo 3: LOCO Detector → Diameter** | Detector, Diameter, Validation | Circle detection + measurement + validation |
| **LOCO Lab** *(experimental)* | — | Advanced proposal/measure/evaluate pipeline |

---

## Pipeline: LOCO Detector → Diameter Measurement

```
Image → Support Region → Candidate Grid + Radii → Feature Extraction
    → Binary Model → Multiclass Model → Combined Filter
    → Circle-NMS → Spatial Filter → Accepted Circles
    → Diameter Measurement → Calibration → Distribution
```

1. **Support region** — Extracted from scribble-based segmentation
2. **Candidate generation** — Grid-based centers × multiple radii
3. **Feature extraction** — Pixel patches + tabular features (intensity, gradient, geometry)
4. **Binary classification** — Valid circle vs. non-circle
5. **Multiclass classification** — Valid / Crossing / Other
6. **Combined decision** — Threshold-based final filtering
7. **Circle-NMS** — Non-maximum suppression by IoU
8. **Spatial filter** — Post-NMS spatial overlap removal
9. **Diameter measurement** — Automatic measurement along accepted circles
10. **Scale calibration** — px → nm/μm conversion
11. **Statistical distribution** — Histogram + CSV export

---

## Output structure

```
data/
├── calibration/          # Scale calibration files (*.json)
├── images/               # Loaded images
├── runs/                 # Diameter measurement runs
├── loco_dataset/         # LOCO training dataset
├── loco_training/        # Trained models
└── reports/              # Exported reports
```

---

## API Endpoints

### Session & Image
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/session/new` | Create new session |
| POST | `/api/image/load` | Load image into session |
| GET | `/api/library/images` | List saved images |
| POST | `/api/library/load` | Load saved image |
| POST | `/api/library/delete` | Delete saved image |

### Scribbles
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/scribble/draft/save` | Save scribble draft |
| GET | `/api/scribble/draft/load` | Load scribble draft |
| POST | `/api/scribble/draft/clear` | Clear scribble draft |

### Experiments (Diameter Research)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/experiments/run` | Run single experiment |
| POST | `/api/experiments/run-batch` | Run batch experiments |
| GET | `/api/experiments/catalog` | List available experiments |

### Diameter Research
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/diameter-research/points/update` | Update diameter points |
| POST | `/api/diameter-research/points/save` | Save diameter points |
| GET | `/api/diameter-research/points/load` | Load diameter points |
| POST | `/api/diameter-research/run` | Run diameter measurement |
| POST | `/api/diameter-research/loco/preview` | Preview LOCO candidates |
| GET | `/api/diameter-research/results/list` | List results |
| GET | `/api/diameter-research/results/get` | Get result details |
| GET | `/api/diameter-research/reports/export` | Export report |

### LOCO Detector
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/diameter-research/loco-models/detect-circles` | Detect circles using trained model |
| POST | `/api/diameter-research/loco-models/measure-accepted` | Measure accepted circles |
| POST | `/api/diameter-research/loco-training/train` | Train LOCO model |
| POST | `/api/diameter-research/loco-training/test-circles` | Test circles with trained model |
| POST | `/api/diameter-research/loco-dataset/features` | Extract dataset features |
| POST | `/api/diameter-research/loco-dataset/save` | Save dataset |
| POST | `/api/diameter-research/loco-dataset/augment/preview` | Preview augmentation |
| POST | `/api/diameter-research/loco-dataset/augment/apply` | Apply augmentation |
| POST | `/api/diameter-research/loco-dataset/augment/clear` | Clear augmentation |

### Calibration
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/diameter-research/calibration/save` | Save scale calibration |
| GET | `/api/diameter-research/calibration/load` | Load scale calibration |
| POST | `/api/diameter-research/calibration/delete` | Delete scale calibration |

### LOCO Lab (experimental)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/diameter-research/loco/proposals` | Generate proposals |
| POST | `/api/diameter-research/loco/filter` | Filter proposals |
| POST | `/api/diameter-research/loco/measure` | Measure proposals |
| POST | `/api/diameter-research/loco/evaluate` | Evaluate proposals |
| POST | `/api/diameter-research/loco/run` | Full LOCO Lab pipeline |

### Validation
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/diameter-research/validation/case/upsert` | Create/update validation case |
| GET | `/api/diameter-research/validation/cases` | List validation cases |
| POST | `/api/diameter-research/validation/run-case` | Run validation case |
| GET | `/api/diameter-research/validation/export` | Export validation results |

---

## Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_calibration.py -v
pytest tests/test_navigation.py -v
pytest tests/test_distribution.py -v
pytest tests/test_loco_to_diameter.py -v
```

| Test file | Tests | Description |
|-----------|-------|-------------|
| `test_api.py` | 1 | Full API flow (session → image → run → review → export) |
| `test_core.py` | 3 | Registry, scribble normalization, draft persistence |
| `test_diameter_research.py` | 14 | Pipeline v1/v2/v3, orientation, geometry guard, persistence |
| `test_navigation.py` | 10 | Legacy-to-group mapping, round-trip, group structure |
| `test_distribution.py` | 14 | Statistics, binning, CSV export, integration |
| `test_loco_to_diameter.py` | 4 | Points replace action, circle conversion |
| `test_calibration.py` | 4 | Calibration save/load/delete, safe ID sanitization |

---

## Security

- **Localhost-only binding** — API binds to `127.0.0.1:8011` by default, not exposed to network
- **Input validation** — All API endpoints validate required fields; invalid requests return clear errors
- **Path sanitization** — File paths are sanitized to prevent directory traversal
- **Session isolation** — Each session has isolated state; images are scoped to sessions
- **CORS** — Frontend origin restriction; only `http://localhost:5173` is allowed
- **Dependency scanning** — Regularly update dependencies with `pip audit` or `npm audit`

See [`SECURITY.md`](SECURITY.md) for full details.

---

## Versioning

This project uses [Semantic Versioning](https://semver.org/). See [`CHANGELOG.md`](CHANGELOG.md) for version history.

Current version: **v1.0.0**

---

## License

MIT
