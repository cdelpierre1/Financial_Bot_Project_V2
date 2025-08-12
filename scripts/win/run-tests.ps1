<#
Lance la suite de tests pytest en utilisant le .venv du projet.
Utilisation:
	powershell -ExecutionPolicy Bypass -File .\scripts\win\run-tests.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Test-Path .venv)) {
	Write-Error "Environnement virtuel introuvable. Lancez d'abord scripts\\win\\venv-create.ps1."
}

./.venv/Scripts/python.exe -m pytest -q tests
