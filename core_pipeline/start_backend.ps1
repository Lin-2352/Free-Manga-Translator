param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8766,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
$env:PYTHONIOENCODING = "utf-8"

$LocalPython = Join-Path $ProjectRoot "..\.venv\Scripts\python.exe"

if ($Python) {
    $PythonExe = $Python
} elseif (Test-Path $LocalPython) {
    $PythonExe = $LocalPython
} elseif ($env:FMT_PYTHON) {
    $PythonExe = $env:FMT_PYTHON
} else {
    $PythonExe = (Get-Command python -ErrorAction Stop).Source
}

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

Write-Host "Using Python: $PythonExe"
Write-Host "Starting Free Manga Translator backend on http://$HostName`:$Port"
& $PythonExe -m uvicorn backend_api.app.main:app --host $HostName --port $Port
