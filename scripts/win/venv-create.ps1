<#
Crée un environnement virtuel Python local (.venv) et installe les dépendances du projet.
Utilisation (PowerShell):
	powershell -ExecutionPolicy Bypass -File .\scripts\win\venv-create.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

Write-Host "[venv] Création de .venv si absent..."
if (-not (Test-Path .venv)) {
		python -m venv .venv
}

Write-Host "[venv] Mise à niveau de pip/setuptools/wheel..."
./.venv/Scripts/python.exe -m pip install -U pip setuptools wheel

Write-Host "[venv] Installation des dépendances..."
./.venv/Scripts/python.exe -m pip install -r requirements.txt

Write-Host "[venv] Terminé. Activez avec: .\\scripts\\win\\venv-activate.ps1"
