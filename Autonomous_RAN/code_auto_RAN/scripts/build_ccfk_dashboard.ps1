# Build CCFK + Autonomous RAN CCFK dashboard
$ErrorActionPreference = "Stop"
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$ccfk = Join-Path $root "ccfk"
$dash = Join-Path $root "code_auto_RAN\ccfk-dashboard"

Write-Host "Building CCFK at $ccfk ..."
Set-Location $ccfk
npm install --legacy-peer-deps
npm run build-ts

Write-Host "Building CCFK dashboard at $dash ..."
Set-Location $dash
npm install --legacy-peer-deps
npm run build

Write-Host "Done. Restart API: python scripts/run_api_server.py"
Write-Host "CCFK Dashboard: http://localhost:8080/dashboard/ccfk"
