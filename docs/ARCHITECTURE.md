# LOCO Detector — Architecture

## Overview

LOCO Detector is a full-stack application with a Python/FastAPI backend and a React frontend. It is designed for automated circle detection and diameter measurement in TEM nanofiber images.

```
┌─────────────────────────────────────────────────────────────┐
│                    LOCO Detector                             │
│                                                             │
│  ┌──────────┐          ┌──────────────────────────────┐    │
│  │ Frontend │  HTTP    │         Backend               │    │
│  │ React    │◄────────►│  FastAPI + Uvicorn            │    │
│  │ Vite     │ :5173    │  :8011                        │    │
│  │          │          │                               │    │
│  │ App.jsx  │          │  main.py (session, image,     │    │
│  │ api.js   │          │    scribbles, experiments)    │    │
│  │ styles.css│         │                               │    │
│  │          │          │  diameter_research/           │    │
│  │ Navigation.jsx      │    api.py (LOCO endpoints)    │    │
│  │ Histogram.jsx       │    pipeline.py (v1)           │    │
│  │ CalibrationPanel.jsx│    pipeline_v2.py             │    │
│  └──────────┘          │    v3/ (multiscale pipeline)  │    │
│                         │    persistence.py            │    │
│                         │    validation.py             │    │
│                         └──────────────────────────────┘    │
│                                                             │
│  ┌──────────────────────────────────────────────────┐       │
│  │  Configuration (.env)                            │       │
│  │  ├─ BACKEND_PORT / BACKEND_HOST                  │       │
│  │  ├─ FRONTEND_PORT / FRONTEND_HOST                │       │
│  │  ├─ DEV_RELOAD (true/false)                      │       │
│  │  └─ VITE_API_BASE (synced to frontend/.env)      │       │
│  └──────────────────────────────────────────────────┘       │
│                                                             │
│  ┌──────────────────────────────────────────────────┐       │
│  │  Scripts                                          │       │
│  │  ├─ run_local.bat     — reads .env, starts both  │       │
│  │  ├─ run_silent.vbs    — silent launcher          │       │
│  │  ├─ run_silent.bat    — delegates to stop_servers│       │
│  │  ├─ stop_servers.ps1  — safe shutdown (reads .env)│      │
│  │  └─ diagnose_ports.ps1— read-only diagnostics    │       │
│  └──────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
LOCO-detector/
├── .env                            # Port and host configuration
├── app.py                          # Backend entry point (reads .env)
├── requirements.txt                # Python dependencies
├── package.json                    # Root package metadata
├── README.md                       # This file
├── CHANGELOG.md                    # Version history
├── SECURITY.md                     # Security policy
├── .gitignore                      # Git ignore rules
├── run_local.bat                   # Windows startup script (reads .env)
├── run_silent.bat                  # Thin wrapper — delegates to stop_servers.ps1
├── run_silent.vbs                  # Silent launcher (reads .env, syncs frontend/.env)
├── stop_servers.ps1                # Windows stop script (reads .env, safety guards)
├── diagnose_ports.ps1              # Read-only port diagnostics
│
├── backend/                        # Python backend
│   ├── __init__.py
│   ├── main.py                     # FastAPI app, session/image/scribble endpoints
│   ├── image_codec.py              # Image encoding/decoding utilities
│   ├── library_store.py            # Saved image library management
│   ├── persistence.py              # Run persistence (CSV/PNG)
│   ├── runner.py                   # Experiment runner
│   ├── scribble.py                 # Scribble processing
│   └── session_store.py            # In-memory session store
│   │
│   └── diameter_research/          # LOCO + Diameter module
│       ├── __init__.py
│       ├── api.py                  # All LOCO/diameter/calibration endpoints
│       ├── persistence.py          # Points/runs persistence
│       ├── pipeline.py             # Diameter measurement v1
│       ├── pipeline_v2.py          # Diameter measurement v2
│       ├── orientation.py          # Local orientation estimation
│       ├── profiles.py             # Intensity profile extraction
│       ├── support_region.py       # Support region extraction
│       ├── report.py               # Report generation
│       ├── validation.py           # Validation case management
│       │
│       └── v3/                     # v3 multiscale pipeline
│           ├── __init__.py
│           ├── pipeline.py         # v3 pipeline orchestrator
│           ├── diagnostics.py      # Diagnostics and metrics
│           ├── edge_pairs.py       # Edge pair detection
│           ├── fallback.py         # Fallback strategies
│           ├── geometry_guard.py   # Geometry validation
│           ├── local_orientation.py# Local orientation (v3)
│           ├── local_preprocess.py # Local preprocessing
│           ├── methodologies.py    # Measurement methodologies
│           ├── multiscale.py       # Multiscale analysis
│           ├── recenter.py         # Center refinement
│           └── support_roi.py      # Support ROI refinement
│
├── frontend/                       # React frontend
│   ├── index.html                  # HTML entry point
│   ├── package.json                # Frontend dependencies
│   ├── vite.config.js              # Vite configuration
│   │
│   └── src/
│       ├── main.jsx                # React entry point
│       ├── App.jsx                 # Main application component (~9400 lines)
│       ├── api.js                  # API client (fetch wrapper)
│       ├── styles.css              # Application styles
│       │
│       └── components/
│           ├── Navigation.jsx      # Hierarchical navigation (3 groups + LOCO Lab)
│           ├── Histogram.jsx       # SVG histogram + statistics
│           └── CalibrationPanel.jsx# Scale calibration UI
│
├── tests/                          # Test suite
│   ├── test_api.py                 # Full API flow test
│   ├── test_core.py                # Core functionality tests
│   ├── test_diameter_research.py   # Diameter pipeline tests
│   ├── test_navigation.py          # Navigation mapping tests
│   ├── test_distribution.py        # Histogram/statistics tests
│   ├── test_loco_to_diameter.py    # Pipeline connection tests
│   └── test_calibration.py         # Calibration endpoint tests
│
└── docs/
    └── LOCO_DETECTOR_PARAMETROS.md # Parameter documentation (Spanish)
```

---

## UI Navigation Architecture

The UI is organized into 3 hierarchical groups plus LOCO Lab (experimental):

```
Level 1 (Group tabs)          Level 2 (Sub-tabs)
─────────────────────         ─────────────────────
Grupo 1: Scribbles → Modelo   Scribbles | Run | Review
Grupo 2: Dataset → LOCO Model Dataset | Training | Test
Grupo 3: LOCO Detector → Diam Detector | Diameter | Validation
LOCO Lab (experimental)       (single tab, no sub-tabs)
```

### Backward compatibility

The legacy `workspaceTab` state (9 flat values) is mapped to/from `activeGroup` + `activeTab`:

```javascript
// Navigation.jsx
const GROUPS = {
  grupo1: { label: 'Grupo 1: Scribbles → Modelo', tabs: ['scribbles', 'run', 'review'] },
  grupo2: { label: 'Grupo 2: Dataset → LOCO Model', tabs: ['dataset', 'training', 'test'] },
  grupo3: { label: 'Grupo 3: LOCO Detector → Diam', tabs: ['detector', 'diameter', 'validation'] },
};

function legacyToGroup(legacyTab) {
  // 'scribbles' → ('grupo1', 'scribbles')
  // 'run'       → ('grupo1', 'run')
  // 'loco'      → ('loco', 'loco')  // special case
}

function groupToLegacy(groupKey, tabKey) {
  // ('grupo1', 'scribbles') → 'scribbles'
  // ('loco', 'loco')        → 'loco'
}
```

---

## Pipeline Architecture

### LOCO Circle Detection Pipeline

```
Input Image
    │
    ▼
Support Region (from scribble segmentation)
    │
    ▼
Candidate Generation
    ├── Grid-based center positions
    └── Multiple radii per center
    │
    ▼
Feature Extraction
    ├── Pixel patch (64×64)
    └── Tabular features (intensity, gradient, geometry)
    │
    ▼
Binary Classification (valid vs. non-circle)
    ├── Random Forest
    ├── ExtraTrees
    ├── XGBoost
    ├── CatBoost
    └── LightGBM
    │
    ▼
Multiclass Classification (valid / crossing / other)
    │
    ▼
Combined Decision Filter
    │
    ▼
Circle-NMS (Non-Maximum Suppression by IoU)
    │
    ▼
Spatial Final Filter (post-NMS overlap removal)
    │
    ▼
Accepted Circles
```

### Diameter Measurement Pipeline

```
Accepted Circles
    │
    ▼
For each circle:
    ├── Extract intensity profile along diameter
    ├── Measure diameter using methodology (v1/v2/v3)
    │   ├── v1: Basic profile analysis
    │   ├── v2: Weighted support + local component
    │   └── v3: Multiscale + edge pairs + geometry guard
    └── Record measurement
    │
    ▼
Scale Calibration (px → nm/μm)
    │
    ▼
Statistical Distribution (histogram)
    ├── Mean, median, std, min, max, N
    ├── Configurable bin count (5-50)
    └── CSV export
```

---

## Data Flow

```
User Action          Frontend              Backend               Storage
───────────          ────────              ───────               ───────
Load image    ──►   api.js fetch    ──►   /api/image/load  ──►  Session memory
Scribble      ──►   Canvas render   ──►   /api/scribble/... ──► Draft file
Run experiment ──►  api.js fetch    ──►   /api/experiments/run ─► CSV + PNG
Detect circles ──►  api.js fetch    ──►   /loco-models/detect  ─► Model files
Measure circles ──► api.js fetch    ──►   /loco-models/measure ─► Points
Calibrate      ──►  CalibrationPanel ──►  /calibration/save   ─► JSON file
View histogram ──►  Histogram.jsx   ──►   (client-side)       ─► (in-memory)
```

---

## Key Design Decisions

1. **Monolithic App.jsx** — The main component is ~9400 lines. This was kept as-is to avoid breaking existing functionality. New features are added as separate components (Navigation, Histogram, CalibrationPanel).

2. **In-memory session** — Sessions are stored in memory (not a database). Images and state are lost on server restart. Persisted data (calibration, runs, datasets) uses the filesystem.

3. **SVG histogram** — The histogram uses pure SVG rendering with no external charting library. This avoids dependency bloat and keeps the bundle small.

4. **Calibration as JSON** — Scale calibration is stored as individual JSON files per image ID. This makes it easy to backup, version, and share calibration data.

5. **Backward-compatible navigation** — The new hierarchical navigation maps to the old flat `workspaceTab` state, ensuring all existing code continues to work without modification.

---

## Security Architecture

See [`SECURITY.md`](../SECURITY.md) for full details.

Key points:
- Backend binds to `127.0.0.1:8011` (localhost only)
- CORS restricted to `http://localhost:5173`
- All inputs validated via Pydantic models
- File paths sanitized to prevent directory traversal
- Session isolation prevents cross-session data leaks
