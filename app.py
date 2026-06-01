"""Entry point for LOCO Detector backend. Run: python app.py"""
import os, sys
from pathlib import Path

# Make sure the project root and backend dir are on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn
from backend.main import app

if __name__ == '__main__':
    host = os.environ.get('BACKEND_HOST', '127.0.0.1')
    port = int(os.environ.get('BACKEND_PORT', '8011'))
    reload = os.environ.get('DEV_RELOAD', '').lower() in ('true', '1', 'yes')
    uvicorn.run('backend.main:app', host=host, port=port, reload=reload)
