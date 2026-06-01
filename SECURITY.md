# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✅ Active |

## Network Security

### Localhost-only binding
The backend API server binds to `127.0.0.1` by default (configured via `BACKEND_HOST` in `.env`). This ensures the API is **not exposed** to the local network or internet. Only processes running on the same machine can communicate with the backend.

The port is configured in the root `.env` file (`BACKEND_PORT`, default `8011`). The entry point `app.py` reads these values at startup.

> ⚠️ **Do not change** the host to `0.0.0.0` unless you fully understand the network exposure implications.

### CORS (Cross-Origin Resource Sharing)
The frontend dev server runs on `http://localhost:5173` (configurable via `FRONTEND_PORT` in `.env`). CORS is configured to only allow requests from this origin:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
```

## Input Validation

### API request validation
All endpoints use Pydantic models (`BaseModel`) to validate incoming requests. Required fields are enforced, and type coercion is explicit:

```python
class SomeRequest(BaseModel):
    session_id: str
    image_id: str | None = None
    value: float = 0.0
```

Invalid requests return a `422 Unprocessable Entity` response with a clear error message describing which field failed validation.

### Parameter bounds
Numeric parameters are clamped to safe ranges:

```python
def _loco_float_param(params, key, default, *, lo=None, hi=None):
    try:
        v = float(params.get(key, default))
        if lo is not None:
            v = max(v, lo)
        if hi is not None:
            v = min(v, hi)
        return v
    except (TypeError, ValueError):
        return default
```

## File System Security

### Path sanitization
All user-provided file paths and image IDs are sanitized to prevent directory traversal attacks:

```python
def _calibration_safe_id(text: str) -> str:
    """Sanitize image ID for use as a filename."""
    safe = re.sub(r'[^a-zA-Z0-9_\-]', '_', str(text))
    return safe.strip('_') or 'default'
```

### Path resolution
File paths are resolved against known base directories, preventing access to arbitrary system files:

```python
def _resolve_existing_dir(path_text: str) -> Path:
    raw = str(path_text or '').strip().strip('"')
    p = Path(raw)
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=400, detail='Directory not found')
    return p
```

## Session Isolation

Each user session is isolated:

- Sessions are identified by a unique `session_id` string
- Images are loaded into specific sessions
- Operations are scoped to the active session and image
- Session state is stored in memory, not persisted to disk

```python
def _require_session(session_id: str):
    sess = SESSION_STORE.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail='Session not found')
    return sess
```

## Dependency Security

### Python dependencies
Regularly audit Python dependencies:

```bash
pip install pip-audit
pip-audit
```

### Node.js dependencies
Regularly audit frontend dependencies:

```bash
cd frontend
npm audit
npm audit fix
```

### Recommended updates
Keep these critical packages up to date:

| Package | Reason |
|---------|--------|
| `fastapi` | Web framework — security patches |
| `uvicorn` | ASGI server — network layer |
| `starlette` | ASGI framework — middleware |
| `numpy` | Numerical processing |
| `scikit-learn` | ML models |
| `react` / `react-dom` | Frontend framework |

## Reporting a Vulnerability

If you discover a security vulnerability in LOCO Detector, please report it by:

1. **Do not** open a public GitHub issue
2. Email the project maintainer directly (see GitHub profile)
3. Include a detailed description of the vulnerability
4. Include steps to reproduce (if applicable)

You can expect:

- **Acknowledgment** within 48 hours
- **Status update** within 5 business days
- **Fix timeline** depending on severity

## Best Practices for Users

1. **Run on a trusted machine** — LOCO Detector is designed for local use on TEM analysis workstations
2. **Keep dependencies updated** — Run `pip install -r requirements.txt --upgrade` periodically
3. **Review loaded images** — Only load TEM images from trusted sources
4. **Use firewall** — If running on a shared network, ensure port 8011 is not exposed
5. **Regular backups** — Back up the `data/` directory containing calibration files and trained models
