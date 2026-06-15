# Repo Tree

Snapshot manual del repositorio enfocado en codigo, scripts y documentacion util. Se excluyen entornos, datos y artefactos generados como `venv/`, `.git/`, `node_modules/`, `outputs/`, `release-assets/`, `data/`, `img/`, `dist/`, `.env*` y caches.

```text
LOCO-detector/
├── backend/
│   ├── adapters/
│   │   └── __init__.py
│   ├── diameter_research/
│   │   ├── v3/
│   │   │   ├── __init__.py
│   │   │   ├── diagnostics.py
│   │   │   ├── edge_pairs.py
│   │   │   ├── fallback.py
│   │   │   ├── geometry_guard.py
│   │   │   ├── local_orientation.py
│   │   │   ├── local_preprocess.py
│   │   │   ├── methodologies.py
│   │   │   ├── multiscale.py
│   │   │   ├── pipeline.py
│   │   │   ├── recenter.py
│   │   │   └── support_roi.py
│   │   ├── __init__.py
│   │   ├── analysis_store.py
│   │   ├── api.py
│   │   ├── orientation.py
│   │   ├── persistence.py
│   │   ├── pipeline.py
│   │   ├── pipeline_v2.py
│   │   ├── profiles.py
│   │   ├── report.py
│   │   ├── support_region.py
│   │   └── validation.py
│   ├── plugins/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── experiments.py
│   │   └── helpers.py
│   ├── __init__.py
│   ├── app.py
│   ├── assist_models.py
│   ├── catalog.py
│   ├── features.py
│   ├── image_codec.py
│   ├── library_store.py
│   ├── main.py
│   ├── metrics.py
│   ├── persistence.py
│   ├── project_transfer.py
│   ├── projects_api.py
│   ├── projects_store.py
│   ├── registry.py
│   ├── runner.py
│   ├── scribble.py
│   └── session_store.py
├── docs/
│   ├── assets/
│   │   └── .gitkeep
│   ├── internal/
│   │   ├── encoding_audit.csv
│   │   ├── plan_tutorial.md
│   │   ├── seguimiento_guardado_circulo.md
│   │   └── seguimiento_traduccion_ui.md
│   ├── GITHUB_RELEASES_WINDOWS_MAC.md
│   ├── README_USUARIO.md
│   └── repo_tree.md
├── frontend/
│   ├── public/
│   │   ├── fonts/
│   │   │   ├── HankenGrotesk-Variable.ttf
│   │   │   └── JetBrainsMono-Variable.ttf
│   │   ├── icons/
│   │   │   └── MaterialSymbolsOutlined.woff2
│   │   ├── licenses/
│   │   │   ├── HankenGrotesk-OFL.txt
│   │   │   ├── JetBrainsMono-OFL.txt
│   │   │   └── MaterialSymbols-Apache-2.0.txt
│   │   └── tutorial/
│   │       ├── bad-example.jpg
│   │       └── overview-reference.png
│   ├── src/
│   │   ├── components/
│   │   │   ├── CalibrationPanel.jsx
│   │   │   ├── Histogram.jsx
│   │   │   ├── Navigation.jsx
│   │   │   ├── OriginToast.jsx
│   │   │   └── TutorialHub.jsx
│   │   ├── api.js
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   ├── styles.css
│   │   └── tutorials.js
│   ├── index.html
│   ├── nginx.conf
│   ├── package-lock.json
│   ├── package.json
│   └── vite.config.js
├── scripts/
│   └── encoding_audit.py
├── tests/
│   ├── test_api.py
│   ├── test_calibration.py
│   ├── test_core.py
│   ├── test_diameter_research.py
│   ├── test_distribution.py
│   ├── test_loco_to_diameter.py
│   ├── test_navigation.py
│   └── test_project_transfer.py
├── tools/
│   ├── check_docker_ports.ps1
│   ├── diagnose_ports.ps1
│   ├── preparar_release_windows.bat
│   ├── restore_checkpoint.bat
│   ├── restore_ui_stage.bat
│   └── stop_servers.ps1
├── .dockerignore
├── .editorconfig
├── .gitignore
├── actualizar_docker_local.bat
├── AGENT.md
├── app.py
├── CHANGELOG.md
├── detener_loco_detector.bat
├── detener_loco_detector.command
├── docker-compose.release.yml
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
├── iniciar_loco_detector.bat
├── iniciar_loco_detector.command
├── LICENSE
├── package.json
├── README.md
├── requirements.txt
├── run_docker.bat
├── run_local.bat
├── run_silent.bat
├── run_silent.vbs
└── SECURITY.md
```
