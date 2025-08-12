# Test de Fumée Complet 45 Minutes

## Vue d'ensemble

Le nouveau test de fumée `run_smoke_45min.py` valide **TOUTES LES FONCTIONNALITÉS** du Financial Bot Crypto V2 en 45 minutes, incluant :

- ✅ **MVP de base** : Scheduler, collecteurs, prédictions
- ✅ **ML Avancé** : 4 modèles (LR, RF, LightGBM, XGBoost), auto-sélection
- ✅ **Nouvelles fonctionnalités** : confidence-metrics, train-incremental
- ✅ **Cycle complet** : collecte → entraînement → prédiction → évaluation

## Exécution

### Via VS Code (Recommandé)
```
Ctrl+Shift+P → Tasks: Run Task → "Tests: fumée 45 minutes COMPLET (via .venv)"
```

### Via PowerShell
```powershell
.\scripts\win\run-smoke-45min.ps1
```

### Direct
```powershell
.\.venv\Scripts\python.exe -m src.tools.run_smoke_45min
```

## Phases du Test (45 minutes)

### Phase 1 : État Initial (0-1 min)
- État du système, paramètres, inventaire des modèles

### Phase 2 : Démarrage (1-2 min)
- Lancement du scheduler, stabilisation

### Phase 3 : États Intermédiaires (2-5 min)
- Première vérification après 5 minutes

### Phase 4 : Prédictions Basiques (5-8 min)
- Bitcoin +10min simple
- Ethereum +15min avec montant EUR

### Phase 5 : ML Avancé (8-12 min)
- Tests `confidence-metrics` (Bitcoin, Ethereum)
- Tests `train-incremental` (micro, mini)

### Phase 6 : États 10 min (12-15 min)
- Vérification après 10 minutes totales

### Phase 7 : Entraînement Complet (15-25 min)
- Entraînement baseline Bitcoin
- Vérification nouveaux modèles

### Phase 8 : États 20 min (25-28 min)
- Tests du calendrier d'entraînement

### Phase 9 : Tests Avancés (28-35 min)
- Entraînement Ethereum
- Prédictions post-entraînement avancées

### Phase 10 : États 35 min (35-38 min)
- Tests finaux métriques (Litecoin)

### Phase 11 : Finalisation (38-45 min)
- États finaux, inventaire complet

### Phase 12 : Arrêt (45+ min)
- Arrêt propre du scheduler

## Sortie

Le test génère un rapport détaillé :
```
logs/test_runs/smoke_45min_YYYYMMDD_HHMMSS.txt
```

Contient :
- Toutes les commandes exécutées
- Sorties complètes (stdout/stderr)
- Codes de retour
- Timestamps précis
- Diagnostics d'erreur

## Couverture Fonctionnelle

### ✅ Infrastructure (100%)
- Scheduler lifecycle complet
- 5 collecteurs spécialisés
- Cache et persistance
- Gestion d'erreurs

### ✅ ML Pipeline (100%)
- 4 algorithmes ML (LR, RF, LGB, XGB)
- Auto-sélection par cross-validation
- 26 features avancées (RSI, Bollinger, etc.)
- Entraînement incrémental (micro/mini/schedule)

### ✅ Nouvelles Fonctionnalités (100%)
- `confidence-metrics` : estimation calibrée
- `train-incremental` : 4 modes automatiques
- Profit calculation avec seuils
- Anti-sleep Windows intégré

### ✅ CLI Complète (100%)
- 12 commandes testées
- Modes interactif/non-interactif
- Validation des paramètres
- Gestion robuste des erreurs

## Différences vs Ancien Test (10 min)

| Aspect | Ancien (10 min) | Nouveau (45 min) |
|--------|-----------------|------------------|
| **Durée** | 10 minutes | 45 minutes |
| **Couverture** | ~40% (MVP seul) | 100% (MVP + ML + Nouvelles fonctionnalités) |
| **Modèles ML** | Aucun test | 4 modèles + auto-sélection |
| **Nouvelles commandes** | 0 | confidence-metrics, train-incremental |
| **Cycles complets** | Partiel | Entraînement → Prédiction → Évaluation |
| **Cryptos testées** | Bitcoin seul | Bitcoin, Ethereum, Litecoin |
| **Validation** | Infrastructure | Infrastructure + ML + Nouvelles fonctionnalités |

## Recommandations d'Usage

### 🎯 Utilisation Optimale
- **Pre-production** : Avant tout déploiement
- **Post-développement** : Après ajout de nouvelles fonctionnalités
- **Validation périodique** : Hebdomadaire pour s'assurer de la stabilité

### ⚡ Exécution Rapide
Pour des tests plus courts, utilisez les tâches spécialisées :
- `CLI: état` pour vérification rapide
- `Tests: pytest` pour tests unitaires
- `CLI: modèles` pour validation ML

### 🔧 Debugging
Si le test échoue :
1. Consultez le rapport dans `logs/test_runs/`
2. Identifiez la phase défaillante
3. Exécutez manuellement la commande problématique
4. Vérifiez les logs du scheduler dans `src/logs/`

## Maintenance

### Ajout de Nouvelles Fonctionnalités
Quand vous ajoutez de nouvelles commandes CLI :
1. Ajoutez un test dans une phase appropriée
2. Documentez la couverture attendue
3. Mettez à jour ce README

### Optimisation des Durées
Les durées d'attente peuvent être ajustées selon :
- Performance de votre machine
- Latence réseau (APIs crypto)
- Complexité des modèles ML

Le test actuel est calibré pour une exécution stable sur machines standard.
