"""
FastAPI entry point for LOCO Detector backend.

This file is needed because run_local.bat and run_silent.vbs expect 'app.py'.
It re-exports the FastAPI app from main.py.
Run with: uvicorn backend.app:app --host 127.0.0.1 --port 8011 --reload
"""
import os, sys
from pathlib import Path

# Ensure the backend directory is in the Python path so relative imports work
backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir.parent))

from backend.main import app
