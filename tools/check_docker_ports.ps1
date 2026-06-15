param(
    [int]$BackendPort = 8011,
    [int]$FrontendPort = 5178
)

$allowedDockerProcesses = @('com.docker.backend', 'wslrelay')
$ports = @($BackendPort, $FrontendPort)
$blocked = @(
    Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
        Where-Object { $ports -contains $_.LocalPort } |
        ForEach-Object {
            $listener = $_
            $process = Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue
            if ($process -and $allowedDockerProcesses -notcontains $process.ProcessName) {
                [pscustomobject]@{
                    Port = $listener.LocalPort
                    Address = $listener.LocalAddress
                    PID = $listener.OwningProcess
                    Process = $process.ProcessName
                    Path = $process.Path
                }
            }
        }
)

if ($blocked.Count -gt 0) {
    Write-Host '[ERROR] Hay procesos locales ocupando puertos requeridos por Docker.' -ForegroundColor Red
    $blocked | Sort-Object Port, PID -Unique | Format-Table -AutoSize
    Write-Host 'Deten esos procesos o cambia DOCKER_BACKEND_PORT / DOCKER_FRONTEND_PORT.' -ForegroundColor Yellow
    exit 1
}

Write-Host '[INFO] Puertos Docker disponibles o ya gestionados por Docker Desktop.'
