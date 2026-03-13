$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (Test-Path $venvPython) {
    $pythonCmd = $venvPython
} else {
    $pythonCmd = "python"
}

try {
    & $pythonCmd --version | Out-Null
} catch {
    Write-Error "Python was not found. Create .venv or make sure 'python' is available in PATH."
}

if (-not (Test-Path (Join-Path $projectRoot "dashboard_api.py"))) {
    Write-Error "dashboard_api.py is missing. Run this script from the project root."
}

Write-Host "Starting BTCUSD MLflow dashboard at http://127.0.0.1:8000"
& $pythonCmd -m uvicorn dashboard_api:app --host 127.0.0.1 --port 8000
