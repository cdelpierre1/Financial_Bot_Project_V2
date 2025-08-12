# Script PowerShell pour exécuter le test de fumée complet 45 minutes
# Teste TOUTES les fonctionnalités : MVP + ML avancé

Write-Host "🧪 Lancement du test de fumée complet (45 minutes)" -ForegroundColor Cyan
Write-Host "📋 Ce test valide :" -ForegroundColor Yellow
Write-Host "   • MVP : Scheduler, collecteurs, prédictions basiques" -ForegroundColor White
Write-Host "   • ML Avancé : 4 modèles (LR, RF, LGB, XGB), auto-sélection" -ForegroundColor White
Write-Host "   • Fonctionnalités nouvelles : confidence-metrics, train-incremental" -ForegroundColor White
Write-Host "   • Cycle complet : entraînement → prédiction → évaluation" -ForegroundColor White
Write-Host ""

# Vérification du venv
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "❌ Environnement virtuel .venv non trouvé" -ForegroundColor Red
    Write-Host "💡 Exécutez d'abord: scripts\win\venv-create.ps1" -ForegroundColor Yellow
    exit 1
}

# Activation du venv et exécution
Write-Host "🐍 Activation de l'environnement virtuel..." -ForegroundColor Green
& ".\.venv\Scripts\python.exe" -m src.tools.run_smoke_45min

Write-Host ""
Write-Host "✅ Test de fumée 45 minutes terminé" -ForegroundColor Green
Write-Host "📁 Vérifiez les logs dans: logs/test_runs/" -ForegroundColor Cyan
