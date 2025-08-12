<#
Lance le test de fumée 10 minutes avec l'interpréteur du venv (.venv).
Usage:
  powershell -ExecutionPolicy Bypass -File .\scripts\win\run-smoke.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Test-Path .venv)) {
	Write-Error "Environnement virtuel introuvable. Lancez d'abord scripts\\win\\venv-create.ps1."
}

# Exécute le module directement via l'exécutable Python du venv
$python = Join-Path (Join-Path (Get-Location) '.venv') 'Scripts\\python.exe'
& $python -m src.tools.run_smoke_10min
