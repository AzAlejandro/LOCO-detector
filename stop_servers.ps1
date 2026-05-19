# Stop LOCO Detector servers
Write-Host "Stopping LOCO Detector servers..." -ForegroundColor Cyan

# Stop backend (Python process on port 8011)
$backend = Get-NetTCPConnection -LocalPort 8011 -ErrorAction SilentlyContinue
if ($backend) {
    $proc = Get-Process -Id $backend.OwningProcess -ErrorAction SilentlyContinue
    if ($proc) {
        Stop-Process -Id $proc.Id -Force
        Write-Host "  [OK] Backend (PID $($proc.Id)) stopped" -ForegroundColor Green
    }
} else {
    Write-Host "  [--] Backend not running" -ForegroundColor Yellow
}

# Stop frontend (Node.js process on port 5173)
$frontend = Get-NetTCPConnection -LocalPort 5173 -ErrorAction SilentlyContinue
if ($frontend) {
    $proc = Get-Process -Id $frontend.OwningProcess -ErrorAction SilentlyContinue
    if ($proc) {
        Stop-Process -Id $proc.Id -Force
        Write-Host "  [OK] Frontend (PID $($proc.Id)) stopped" -ForegroundColor Green
    }
} else {
    Write-Host "  [--] Frontend not running" -ForegroundColor Yellow
}

# Also kill any orphaned python.exe running app.py
$orphans = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -match 'app.py'
}
foreach ($p in $orphans) {
    Stop-Process -Id $p.Id -Force
    Write-Host "  [OK] Orphaned backend (PID $($p.Id)) stopped" -ForegroundColor Green
}

Write-Host ""
Write-Host "Done." -ForegroundColor Cyan
