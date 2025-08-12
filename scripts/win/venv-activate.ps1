<#
Active l'environnement virtuel local du projet (.venv) dans la session PowerShell courante.
Utilisation:
	. .\scripts\win\venv-activate.ps1
(notez le point initial + espace)
#>

param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$venvActivate = Join-Path (Join-Path (Get-Location) '.venv') 'Scripts\Activate.ps1'
if (-not (Test-Path $venvActivate)) {
	Write-Error "Environnement virtuel introuvable. Lancez d'abord scripts\\win\\venv-create.ps1."
}

. $venvActivate
Write-Host "Environnement .venv activé"
