# Stop LOCO Detector servers
# Safety rules:
# - Never kill PID 0 (System Idle Process / stale TCP entry)
# - Never kill the current PowerShell process
# - Never kill by process name alone (python.exe, node.exe, etc.)
# - Only kill if the process command line contains the LOCO-detector project path
$ErrorActionPreference = 'Stop'

# --- Resolve project directory ---
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_DIR = Resolve-Path (Join-Path $SCRIPT_DIR '..')

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  LOCO Detector - Stop Servers" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Project: $PROJECT_DIR" -ForegroundColor Cyan
Write-Host ""

# --- Read .env ---
$ENV_PATH = Join-Path $PROJECT_DIR '.env'
if (Test-Path $ENV_PATH) {
    $envLines = Get-Content $ENV_PATH
    $envMap = @{}
    foreach ($line in $envLines) {
        $line = $line.Trim()
        if ($line -and $line -notmatch '^\s*#') {
            $parts = $line -split '=', 2
            if ($parts.Count -eq 2) {
                $envMap[$parts[0].Trim()] = $parts[1].Trim()
            }
        }
    }
    $BACKEND_PORT = if ($envMap.ContainsKey('BACKEND_PORT')) { $envMap['BACKEND_PORT'] } else { '8011' }
    $FRONTEND_PORT = if ($envMap.ContainsKey('FRONTEND_PORT')) { $envMap['FRONTEND_PORT'] } else { '5173' }
} else {
    Write-Host "  [WARN] .env not found, using defaults." -ForegroundColor Yellow
    $BACKEND_PORT = '8011'
    $FRONTEND_PORT = '5173'
}

Write-Host "  Backend port : $BACKEND_PORT" -ForegroundColor Cyan
Write-Host "  Frontend port: $FRONTEND_PORT" -ForegroundColor Cyan
Write-Host ""

# --- Helper: get process info by PID ---
function Get-ProcessInfo($procId) {
    $info = Get-CimInstance Win32_Process -Filter "ProcessId = $procId" -ErrorAction SilentlyContinue
    if ($info) {
        return @{
            ProcessId      = $info.ProcessId
            Name           = $info.Name
            CommandLine    = $info.CommandLine
            ParentProcessId = $info.ParentProcessId
        }
    }
    return $null
}

# --- Helper: kill a process safely ---
function Stop-ProjectProcess($procId, $portLabel) {
    # SAFETY 1: Never kill PID 0
    if ($procId -eq 0) {
        Write-Host "  [SKIP] PID 0 (stale TCP entry) on $portLabel" -ForegroundColor Yellow
        return
    }

    # SAFETY 2: Never kill the current PowerShell process
    if ($procId -eq $PID) {
        Write-Host "  [SKIP] PID $procId is the current PowerShell process on $portLabel" -ForegroundColor Yellow
        return
    }

    # Verify the process actually exists
    $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
    if (-not $proc) {
        Write-Host "  [GHOST] PID $procId on $portLabel - process no longer exists (stale TCP entry)" -ForegroundColor Yellow
        return
    }

    # Get command line
    $info = Get-ProcessInfo $procId
    if (-not $info) {
        Write-Host "  [SKIP] PID $procId on $portLabel - cannot read process info" -ForegroundColor Yellow
        return
    }

    $cmdLine = $info.CommandLine
    $procName = $info.Name
    $parentPid = $info.ParentProcessId

    Write-Host "  PID $procId ($procName, parent PID $parentPid)" -ForegroundColor Gray
    Write-Host "    CommandLine: $cmdLine" -ForegroundColor Gray

    # SAFETY 3: Only kill if command line contains the project directory
    if ($cmdLine -notmatch [regex]::Escape($PROJECT_DIR)) {
        Write-Host "  [SKIP] PID $procId does not belong to this project (command line does not contain project path)" -ForegroundColor Yellow
        return
    }

    # All safety checks passed — kill
    Write-Host "  [KILL] Stopping PID $procId ($procName) on $portLabel ..." -ForegroundColor Red
    try {
        taskkill /F /T /PID $procId 2>&1 | Out-Null
        Write-Host "  [OK] PID $procId terminated" -ForegroundColor Green
    } catch {
        Write-Host "  [WARN] Failed to kill PID $procId : $_" -ForegroundColor Red
    }
}

# --- Query ports ---
$PORTS_TO_CHECK = @($BACKEND_PORT, $FRONTEND_PORT)
$PORT_LABELS = @{ $BACKEND_PORT = "backend port $BACKEND_PORT"; $FRONTEND_PORT = "frontend port $FRONTEND_PORT" }

foreach ($port in $PORTS_TO_CHECK) {
    $label = $PORT_LABELS[$port]
    Write-Host "--- Checking port $port ($label) ---" -ForegroundColor Cyan

    # Method 1: Get-NetTCPConnection
    $connections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($connections) {
        foreach ($conn in $connections) {
            Stop-ProjectProcess $conn.OwningProcess $label
        }
    } else {
        Write-Host "  [--] No active connection on port $port" -ForegroundColor Gray
    }

    # Method 2: netstat -ano (catches entries that Get-NetTCPConnection might miss)
    $netstatOutput = netstat -ano -p tcp 2>$null | Select-String ":$port\s"
    if ($netstatOutput) {
        foreach ($line in $netstatOutput) {
            $tokens = $line -split '\s+'
            $pidFromNetstat = $tokens[-1]
            if ($pidFromNetstat -and $pidFromNetstat -match '^\d+$') {
                $pidVal = [int]$pidFromNetstat
                # Only process if we haven't already killed it
                $existing = Get-Process -Id $pidVal -ErrorAction SilentlyContinue
                if ($existing) {
                    Stop-ProjectProcess $pidVal "$label (netstat)"
                }
            }
        }
    }
    Write-Host ""
}

# --- Final verification ---
Write-Host "--- Final verification ---" -ForegroundColor Cyan
$allFree = $true
foreach ($port in $PORTS_TO_CHECK) {
    $label = $PORT_LABELS[$port]
    $check = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($check) {
        Write-Host "  [WARN] Port $port ($label) is still in use!" -ForegroundColor Red
        $allFree = $false
    } else {
        Write-Host "  [OK] Port $port ($label) is free" -ForegroundColor Green
    }
}

Write-Host ""
if ($allFree) {
    Write-Host "Done. All LOCO ports are free." -ForegroundColor Cyan
} else {
    Write-Host "Done. Some ports could not be freed. Run .\tools\diagnose_ports.ps1 for details." -ForegroundColor Yellow
}
