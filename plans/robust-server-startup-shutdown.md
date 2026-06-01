You are working on my local Windows project:

C:\Users\alejo\Documents\GitHub\LOCO-detector

Goal:
Fix the server startup/shutdown lifecycle so the project does not leave stale/orphan processes, does not report false success, and never interferes with ComfyUI running on http://127.0.0.1:8188/.

Important context:
- LOCO backend currently uses port 8011.
- LOCO frontend probably uses port 5173.
- ComfyUI uses port 8188 and must never be killed, modified, or managed by this project.
- Previous logs showed many LISTENING entries on 127.0.0.1:8011 with PIDs that Windows could not resolve as active processes.
- The old stop_servers.ps1 attempted to kill PID 0 / Idle and printed false success.
- Do not assume the root cause is definitely uvicorn reload. Treat it as one possible cause among stale TCP entries, reload child processes, or Windows networking inconsistency.

Main design rules:
1. A process can only be killed if BOTH are true:
   - It is using the configured LOCO backend/frontend port.
   - Its command line contains the exact LOCO-detector project path.
2. Port match alone is never enough to kill a process.
3. Never kill PID 0.
4. Never kill the current PowerShell process.
5. Never kill by generic process name only, such as python.exe, node.exe, npm, vite, or uvicorn.
6. Never touch port 8188.
7. Use a single shutdown implementation: stop_servers.ps1.
8. Any silent launcher must call stop_servers.ps1 instead of duplicating kill logic.
9. If a PID is shown by netstat or Get-NetTCPConnection but tasklist/Get-Process cannot find it, report it as a ghost/stale TCP entry and recommend changing port or restarting Windows.
10. Prefer safety over aggressive cleanup.

Tasks:

1. Inspect the repository
Search for:
- 8011
- 8012
- 5173
- 8188
- uvicorn
- fastapi
- reload=True
- npm run dev
- vite
- stop_servers.ps1
- run_local.bat
- run_silent.bat
- run_silent.vbs
- package.json
- frontend API base URLs

Report where ports are hardcoded before modifying anything.

2. Add centralized configuration
Create a root .env file if missing:

BACKEND_PORT=8011
BACKEND_HOST=127.0.0.1
FRONTEND_PORT=5173
FRONTEND_HOST=localhost
DEV_RELOAD=true
VITE_API_BASE=http://127.0.0.1:8011

Rules:
- Do not include port 8188 in .env.
- 8188 is ComfyUI-only and not part of LOCO configuration.
- If backend port changes, VITE_API_BASE must also change.

3. Modify app.py
Make app.py read the root .env manually, without adding python-dotenv.
It must read:
- BACKEND_PORT
- BACKEND_HOST
- DEV_RELOAD

Use:
- DEV_RELOAD=true for normal dev mode.
- DEV_RELOAD=false if Windows reload behavior causes orphan child processes.

The uvicorn startup should use those values.

4. Rewrite stop_servers.ps1
This is the most important file.

Requirements:
- Read BACKEND_PORT and FRONTEND_PORT from .env.
- Determine PROJECT_DIR from the script location.
- Print the project path and configured ports.
- Print clearly that ComfyUI port 8188 will not be touched.
- Query the configured backend and frontend ports.
- For each connection:
  - Skip PID 0.
  - Skip the current PowerShell PID.
  - Verify the PID exists with Get-Process.
  - Read CommandLine and ParentProcessId with Get-CimInstance Win32_Process.
  - Print PID, process name, parent PID, and command line.
  - Kill only if CommandLine contains PROJECT_DIR.
  - If CommandLine does not contain PROJECT_DIR, print SKIP and do not kill.
  - Use taskkill /F /T /PID only after the project-path check passes.
- After killing, re-check the port.
- Print [OK] only if the port is actually free.
- Print [WARN] if the port is still occupied.
- Do not kill generic python.exe or node.exe unless CommandLine contains PROJECT_DIR.
- Never reference port 8188 except in a message saying it is not touched.

5. Create diagnose_ports.ps1
This script must be read-only.

It should:
- Read .env.
- Show diagnostics for:
  - configured backend port
  - configured frontend port
  - ComfyUI port 8188, read-only only
- Use both:
  - Get-NetTCPConnection
  - netstat -ano
- For each PID, show:
  - PID
  - process name
  - parent PID
  - command line
  - TCP state
- If PID is 0, report it as PID 0 / stale entry.
- If PID appears in TCP output but no process exists, report it as ghost/stale TCP entry.
- Never kill anything.

6. Modify run_local.bat
Requirements:
- Read BACKEND_PORT, BACKEND_HOST, FRONTEND_PORT, and VITE_API_BASE from root .env.
- Before starting backend, check whether BACKEND_PORT is already in use.
- If the port is in use, do not auto-kill.
- Print a clear message:
  - run .\diagnose_ports.ps1 to inspect
  - run .\stop_servers.ps1 to stop LOCO processes safely
  - or change BACKEND_PORT in .env
- Sync VITE_API_BASE into frontend/.env before starting frontend.
- Start backend with python app.py.
- Start frontend with npm run dev.
- Do not touch 8188.

7. Modify run_silent.bat
Do not duplicate kill logic.
Make it a thin wrapper around stop_servers.ps1.

It should:
- Resolve PROJECT_DIR.
- cd into PROJECT_DIR.
- call PowerShell stop_servers.ps1.
- Nothing else.

8. Modify run_silent.vbs
Requirements:
- Read BACKEND_PORT, FRONTEND_PORT, and VITE_API_BASE from .env.
- Sync VITE_API_BASE into frontend/.env.
- Call stop_servers.ps1 before starting.
- Start backend with python app.py.
- Wait for backend health check on BACKEND_PORT.
- Start frontend.
- Open browser on FRONTEND_PORT.
- Provide the complete final VBS file. Do not leave truncated lines.

9. Modify frontend API usage
Find hardcoded backend URLs in frontend/src/App.jsx or related files.
Replace hardcoded http://127.0.0.1:8011 or http://127.0.0.1:8012 with the shared API_BASE from frontend/src/api.js.

For Vite:
- frontend/.env should contain VITE_API_BASE=http://127.0.0.1:8011
- frontend code should read import.meta.env.VITE_API_BASE

10. Modify package.json
Change backend dev script to use app.py instead of directly calling uvicorn, so .env and DEV_RELOAD are respected.

Example:
"backend:dev": ".venv\\Scripts\\python.exe app.py"

But inspect the actual environment first. If the project uses venv instead of .venv, use the correct path or make the script portable.

11. Documentation
Update README or docs only after code changes.
Document:
- How to change backend port.
- How to diagnose ports.
- How to stop LOCO safely.
- That ComfyUI on 8188 is never managed by these scripts.

12. Verification
After implementing, provide:
- List of modified files.
- Complete final content of every modified script.
- Commands to run:

.\diagnose_ports.ps1
.\stop_servers.ps1
.\run_local.bat

- Explain what output indicates success.
- Confirm explicitly:
  - No script kills port 8188.
  - stop_servers.ps1 only kills processes whose command line contains the LOCO-detector project path.
  - run_silent.bat delegates to stop_servers.ps1.
  - frontend/.env is synced from root .env.