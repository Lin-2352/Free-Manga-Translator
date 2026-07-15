param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8766,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
$env:PYTHONIOENCODING = "utf-8"

$LocalPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
# The project's real venv lives two levels up (alongside the repo root), not inside
# core_pipeline itself -- checked here so this fallback chain can't silently land on a
# mismatched system Python (see DEVELOPMENT_NOTES.md, 2026-07-14, for the crash that caused).
$RepoRootPython = Join-Path $ProjectRoot "..\..\.venv\Scripts\python.exe"

if ($Python) {
    $PythonExe = $Python
} elseif (Test-Path $LocalPython) {
    $PythonExe = $LocalPython
} elseif (Test-Path $RepoRootPython) {
    $PythonExe = $RepoRootPython
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
