<#
Exécute l'entrée CLI du projet en utilisant l'interpréteur du venv (.venv).
Utilisation:
	powershell -ExecutionPolicy Bypass -File .\scripts\win\run-cli.ps1 -- [arguments]
Tous les arguments après -- sont transmis à main.py
#>

param(
	[Parameter(ValueFromRemainingArguments = $true)]
	[string[]]$Rest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Test-Path .venv)) {
	Write-Error "Environnement virtuel introuvable. Lancez d'abord scripts\\win\\venv-create.ps1."
}

./.venv/Scripts/python.exe .\src\main.py @Rest
