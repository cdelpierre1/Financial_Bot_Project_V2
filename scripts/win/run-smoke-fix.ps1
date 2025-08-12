# Script PowerShell pour le test ciblé des corrections
# Teste SEULEMENT les parties qui ont échoué dans le test complet

Write-Host "🔧 Test ciblé des corrections identifiées" -ForegroundColor Cyan
Write-Host "🎯 Ce test corrige :" -ForegroundColor Yellow
Write-Host "   • CLI prevoir : syntaxe --horizon → --unit + --value" -ForegroundColor White
Write-Host "   • Entraînement ML : attente de collecte de données five_min" -ForegroundColor White
Write-Host "   • Train-incremental : vérification après collecte suffisante" -ForegroundColor White
Write-Host ""

# Vérification du venv
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "❌ Environnement virtuel .venv non trouvé" -ForegroundColor Red
    Write-Host "💡 Exécutez d'abord: scripts\win\venv-create.ps1" -ForegroundColor Yellow
    exit 1
}

# Activation du venv et exécution
Write-Host "🐍 Activation de l'environnement virtuel..." -ForegroundColor Green
& ".\.venv\Scripts\python.exe" -m src.tools.run_smoke_fix

Write-Host ""
Write-Host "✅ Test ciblé terminé" -ForegroundColor Green
Write-Host "📁 Vérifiez les logs dans: logs/test_runs/" -ForegroundColor Cyan
