# Diagnose ports used by LOCO Detector
# READ-ONLY: This script never kills any process.
# It shows diagnostics for:
#   - configured backend port
#   - configured frontend port

$ErrorActionPreference = 'SilentlyContinue'

# --- Resolve project directory ---
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_DIR = Resolve-Path $SCRIPT_DIR

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  LOCO Detector - Port Diagnostics" -ForegroundColor Cyan
Write-Host "  READ-ONLY - No processes will be killed" -ForegroundColor Yellow
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
    Write-Host "  Configured backend port : $BACKEND_PORT" -ForegroundColor Cyan
    Write-Host "  Configured frontend port: $FRONTEND_PORT" -ForegroundColor Cyan
} else {
    Write-Host "  [WARN] .env not found, using defaults." -ForegroundColor Yellow
    $BACKEND_PORT = '8011'
    $FRONTEND_PORT = '5173'
}
Write-Host ""

# --- Helper: get process info by PID ---
function Get-ProcessInfo($targetPid) {
    if ($targetPid -eq 0) {
        return @{
            ProcessId       = 0
            Name            = 'Idle (PID 0)'
            CommandLine     = 'N/A - PID 0 is the System Idle Process or a stale TCP entry'
            ParentProcessId = 0
        }
    }
    $info = Get-CimInstance Win32_Process -Filter "ProcessId = $targetPid" -ErrorAction SilentlyContinue
    if ($info) {
        return @{
            ProcessId       = $info.ProcessId
            Name            = $info.Name
            CommandLine     = $info.CommandLine
            ParentProcessId = $info.ParentProcessId
        }
    }
    return $null
}

# --- Helper: print connection info ---
function Show-Connection($port, $label, $conn, $source) {
    $owningPid = $conn.OwningProcess
    $state = $conn.State
    $localAddr = "$($conn.LocalAddress):$($conn.LocalPort)"
    $remoteAddr = "$($conn.RemoteAddress):$($conn.RemotePort)"

    Write-Host "  [$source] $localAddr -> $remoteAddr ($state)" -ForegroundColor Gray
    Write-Host "    Owning PID: $owningPid" -ForegroundColor Gray

    $info = Get-ProcessInfo $owningPid
    if ($info) {
        Write-Host "    Process: $($info.Name)" -ForegroundColor Gray
        Write-Host "    Parent PID: $($info.ParentProcessId)" -ForegroundColor Gray
        Write-Host "    CommandLine: $($info.CommandLine)" -ForegroundColor Gray
        if ($owningPid -eq 0) {
            Write-Host "    => PID 0 / stale TCP entry (not a real process)" -ForegroundColor Yellow
        } elseif ($info.CommandLine -match [regex]::Escape($PROJECT_DIR)) {
            Write-Host "    => BELONGS TO THIS PROJECT" -ForegroundColor Green
        } else {
            Write-Host "    => EXTERNAL PROCESS (not from this project)" -ForegroundColor Magenta
        }
    } else {
        Write-Host "    Process: NOT FOUND (ghost/stale TCP entry)" -ForegroundColor Yellow
    }
    Write-Host ""
}

# --- Helper: show netstat entries for a port ---
function Show-NetstatForPort($port, $label) {
    $netstatOutput = netstat -ano -p tcp 2>$null | Select-String ":$port\s"
    if (-not $netstatOutput) {
        Write-Host "  [netstat] No entries for port $port ($label)" -ForegroundColor Gray
        Write-Host ""
        return
    }
    foreach ($line in $netstatOutput) {
        $tokens = $line -split '\s+'
        if ($tokens.Count -ge 5) {
            $localAddr = $tokens[1]
            $remoteAddr = $tokens[2]
            $state = $tokens[3]
            $pidFromNetstat = $tokens[-1]

            Write-Host "  [netstat] $localAddr -> $remoteAddr ($state)" -ForegroundColor Gray
            Write-Host "    Owning PID: $pidFromNetstat" -ForegroundColor Gray

            if ($pidFromNetstat -match '^\d+$') {
                $pidVal = [int]$pidFromNetstat
                $info = Get-ProcessInfo $pidVal
                if ($info) {
                    Write-Host "    Process: $($info.Name)" -ForegroundColor Gray
                    Write-Host "    Parent PID: $($info.ParentProcessId)" -ForegroundColor Gray
                    Write-Host "    CommandLine: $($info.CommandLine)" -ForegroundColor Gray
                    if ($pidVal -eq 0) {
                        Write-Host "    => PID 0 / stale TCP entry (not a real process)" -ForegroundColor Yellow
                    } elseif ($info.CommandLine -match [regex]::Escape($PROJECT_DIR)) {
                        Write-Host "    => BELONGS TO THIS PROJECT" -ForegroundColor Green
                    } else {
                        Write-Host "    => EXTERNAL PROCESS (not from this project)" -ForegroundColor Magenta
                    }
                } else {
                    Write-Host "    Process: NOT FOUND (ghost/stale TCP entry)" -ForegroundColor Yellow
                }
            }
            Write-Host ""
        }
    }
}

# --- Diagnose each port ---
$PORTS = @(
    @{ Port = $BACKEND_PORT; Label = "LOCO Backend (configured)" },
    @{ Port = $FRONTEND_PORT; Label = "LOCO Frontend (configured)" }
)

foreach ($entry in $PORTS) {
    $port = $entry.Port
    $label = $entry.Label
    Write-Host "========== Port $port - $label ==========" -ForegroundColor Cyan

    # Method 1: Get-NetTCPConnection
    Write-Host "--- Get-NetTCPConnection ---" -ForegroundColor Cyan
    $connections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($connections) {
        foreach ($conn in $connections) {
            Show-Connection $port $label $conn "PowerShell"
        }
    } else {
        Write-Host "  No connections found via Get-NetTCPConnection" -ForegroundColor Gray
        Write-Host ""
    }

    # Method 2: netstat -ano
    Write-Host "--- netstat -ano ---" -ForegroundColor Cyan
    Show-NetstatForPort $port $label
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Diagnostics complete." -ForegroundColor Cyan
Write-Host "  No processes were killed." -ForegroundColor Green
Write-Host "  To stop LOCO processes safely:" -ForegroundColor Gray
Write-Host "    .\stop_servers.ps1" -ForegroundColor Gray
Write-Host "============================================" -ForegroundColor Cyan
