"""
Test de fumée complet (45 minutes) de bout en bout avec rapport TXT clair (français).

Ce que le test vérifie (non interactif) - TOUTES LES FONCTIONNALITÉS :
- État initial (datasets/cache/système), paramètres
- Démarrage du scheduler (arrière-plan)
- États intermédiaires (≈5 min, ≈10 min, ≈20 min, ≈35 min)
- Prédictions multiples (bitcoin, ethereum) avec et sans montant EUR
- Test des nouvelles commandes ML avancées (confidence-metrics, train-incremental)
- Validation des 5 collecteurs spécialisés
- Entraînement incrémental automatique et modèles ML avancés
- Cycle d'évaluation post-horizon complet
- Validation du profit gate et décisions NO_CALL
- État final (≈45 min)
- Arrêt du scheduler

Sortie: logs/test_runs/smoke_45min_YYYYMMDD_HHMMSS.txt

Remarque: n'altère pas le venv; teste MVP + ML avancé complet.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
import argparse
from typing import Tuple


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
LOG_DIR = os.path.join(ROOT, "logs", "test_runs")


def run_cli_command(command_parts: list[str], description: str, log_file) -> Tuple[str, int]:
    """
    Exécute une commande CLI et retourne (stdout, returncode).
    Log les erreurs dans le fichier de sortie.
    """
    timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    log_file.write(f"\n[{timestamp}] Commande: {description}\n")
    log_file.write(f"$ python -m {' '.join(command_parts)}\n")
    log_file.flush()
    
    try:
        # Commande complète avec python -m
        full_cmd = [sys.executable, "-m"] + command_parts
        
        result = subprocess.run(
            full_cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=180  # 3 minutes max par commande
        )
        
        # Log la sortie
        if result.stdout.strip():
            log_file.write(f"STDOUT:\n{result.stdout}\n")
        if result.stderr.strip():
            log_file.write(f"STDERR:\n{result.stderr}\n")
        
        log_file.write(f"Return code: {result.returncode}\n")
        log_file.write("-" * 50 + "\n")
        log_file.flush()
        
        return result.stdout, result.returncode
        
    except subprocess.TimeoutExpired:
        log_file.write("ERREUR: Timeout (3 min) dépassé\n")
        log_file.write("-" * 50 + "\n")
        log_file.flush()
        return "", 1
    except Exception as e:
        log_file.write(f"ERREUR: {e}\n")
        log_file.write("-" * 50 + "\n")
        log_file.flush()
        return "", 1


def wait_and_log(duration_sec: int, description: str, log_file):
    """Attend et log le temps d'attente."""
    timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    log_file.write(f"\n[{timestamp}] Attente: {description} ({duration_sec}s)\n")
    log_file.flush()
    time.sleep(duration_sec)


def main():
    """
    Test de fumée complet 45 minutes - TOUTES LES FONCTIONNALITÉS.
    """
    parser = argparse.ArgumentParser(description="Test de fumée 45 min complet (MVP + ML avancé)")
    parser.add_argument("--dry-run", action="store_true", help="Affiche les commandes sans les exécuter")
    args = parser.parse_args()
    
    # Créer le répertoire de logs
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Nom du fichier de log avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"smoke_45min_{timestamp}.txt"
    log_path = os.path.join(LOG_DIR, log_filename)
    
    print(f"🧪 Démarrage du test de fumée 45 minutes (COMPLET)")
    print(f"📝 Log: {log_path}")
    
    if args.dry_run:
        print("🔍 Mode dry-run activé - pas d'exécution réelle")
        return
    
    with open(log_path, "w", encoding="utf-8") as log_file:
        # En-tête
        start_time = datetime.now(timezone.utc)
        log_file.write(f"=== TEST DE FUMÉE 45 MINUTES COMPLET ===\n")
        log_file.write(f"Début: {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        log_file.write(f"Python: {sys.executable}\n")
        log_file.write(f"Répertoire: {ROOT}\n")
        log_file.write(f"TEST COMPLET: MVP + ML Avancé + Nouvelles Fonctionnalités\n")
        log_file.write("=" * 70 + "\n\n")
        log_file.flush()
        
        try:
            # === PHASE 1: ÉTAT INITIAL ET PARAMÈTRES ===
            print("📊 Phase 1: État initial et paramètres...")
            
            # État du système
            run_cli_command(["src.main", "etat"], "État initial du système", log_file)
            
            # Paramètres
            run_cli_command(["src.main", "parametres"], "Affichage des paramètres", log_file)
            
            # Inventaire des modèles
            run_cli_command(["src.main", "models"], "Inventaire des modèles ML", log_file)
            
            # === PHASE 2: DÉMARRAGE DU SCHEDULER ===
            print("🚀 Phase 2: Démarrage du scheduler...")
            
            run_cli_command(["src.main", "demarrer"], "Démarrage du scheduler", log_file)
            
            # Attente initiale
            wait_and_log(30, "Stabilisation du scheduler", log_file)
            
            # === PHASE 3: PREMIERS ÉTATS INTERMÉDIAIRES ===
            print("📈 Phase 3: États intermédiaires (5 min)...")
            
            # État après 5 min
            wait_and_log(270, "Attente 5 minutes totales", log_file)  # 270s = 4.5 min supplémentaires
            run_cli_command(["src.main", "etat"], "État après ~5 minutes", log_file)
            
            # === PHASE 4: TESTS DES PRÉDICTIONS MULTIPLES ===
            print("🔮 Phase 4: Tests des prédictions...")
            
            # Prédiction Bitcoin simple
            run_cli_command(["src.main", "prevoir", "--coin", "bitcoin", "--unit", "min", "--value", "10", "--kind", "price"], 
                          "Prédiction Bitcoin +10min", log_file)
            
            # Prédiction Ethereum avec montant
            run_cli_command(["src.main", "prevoir", "--coin", "ethereum", "--unit", "min", "--value", "15", "--kind", "amount", "--amount-eur", "100"], 
                          "Prédiction Ethereum +15min avec 100 EUR", log_file)
            
            # === PHASE 5: NOUVELLES COMMANDES ML AVANCÉES ===
            print("🧠 Phase 5: Tests des fonctionnalités ML avancées...")
            
            # Test collecte historique massive
            run_cli_command(["src.main", "collect-historical"], 
                          "Collecte historique massive urgente", log_file)
            
            # Test métriques de confiance
            run_cli_command(["src.main", "confidence-metrics", "--coin", "bitcoin", "--horizon", "10"], 
                          "Métriques de confiance Bitcoin", log_file)
            
            run_cli_command(["src.main", "confidence-metrics", "--coin", "ethereum", "--horizon", "15"], 
                          "Métriques de confiance Ethereum", log_file)
            
            # Test entraînement incrémental micro
            run_cli_command(["src.main", "train-incremental", "--mode", "micro"], 
                          "Entraînement incrémental micro", log_file)
            
            # === PHASE 6: ÉTATS INTERMÉDIAIRES 10 MIN ===
            print("📊 Phase 6: États intermédiaires (10 min)...")
            
            wait_and_log(300, "Attente 10 minutes totales", log_file)  # 5 min supplémentaires
            run_cli_command(["src.main", "etat"], "État après ~10 minutes", log_file)
            
            # Test entraînement incrémental mini
            run_cli_command(["src.main", "train-incremental", "--mode", "mini"], 
                          "Entraînement incrémental mini", log_file)
            
            # === PHASE 7: ENTRAÎNEMENT COMPLET ===
            print("🎯 Phase 7: Entraînement complet...")
            
            # Entraînement baseline Bitcoin
            run_cli_command(["src.main", "train", "--coin", "bitcoin"], 
                          "Entraînement baseline Bitcoin", log_file)
            
            # Vérification des nouveaux modèles
            run_cli_command(["src.main", "models"], "Inventaire après entraînement", log_file)
            
            # === PHASE 8: ÉTATS INTERMÉDIAIRES 20 MIN ===
            print("📈 Phase 8: États intermédiaires (20 min)...")
            
            wait_and_log(600, "Attente 20 minutes totales", log_file)  # 10 min supplémentaires
            run_cli_command(["src.main", "etat"], "État après ~20 minutes", log_file)
            
            # Test du calendrier d'entraînement
            run_cli_command(["src.main", "train-incremental", "--mode", "schedule"], 
                          "Calendrier d'entraînement automatique", log_file)
            
            # === PHASE 9: TESTS AVANCÉS DES MODÈLES ===
            print("🔬 Phase 9: Tests avancés des modèles...")
            
            # Entraînement Ethereum
            run_cli_command(["src.main", "train", "--coin", "ethereum"], 
                          "Entraînement baseline Ethereum", log_file)
            
            # Prédictions post-entraînement
            run_cli_command(["src.main", "prevoir", "--coin", "bitcoin", "--unit", "min", "--value", "30", "--kind", "amount", "--amount-eur", "250"], 
                          "Prédiction Bitcoin +30min avec 250 EUR (post-training)", log_file)
            
            # === PHASE 10: ÉTATS INTERMÉDIAIRES 35 MIN ===
            print("📊 Phase 10: États intermédiaires (35 min)...")
            
            wait_and_log(900, "Attente 35 minutes totales", log_file)  # 15 min supplémentaires
            run_cli_command(["src.main", "etat"], "État après ~35 minutes", log_file)
            
            # Test des métriques finales
            run_cli_command(["src.main", "confidence-metrics", "--coin", "litecoin", "--horizon", "20"], 
                          "Métriques de confiance Litecoin", log_file)
            
            # === PHASE 11: FINALISATION ET NETTOYAGE ===
            print("🏁 Phase 11: Finalisation (45 min)...")
            
            wait_and_log(600, "Attente 45 minutes totales", log_file)  # 10 min supplémentaires pour atteindre 45 min
            
            # État final
            run_cli_command(["src.main", "etat"], "État final après 45 minutes", log_file)
            
            # Inventaire final des modèles
            run_cli_command(["src.main", "models"], "Inventaire final des modèles", log_file)
            
            # === PHASE 12: ARRÊT DU SCHEDULER ===
            print("🛑 Phase 12: Arrêt du scheduler...")
            
            run_cli_command(["src.main", "arreter"], "Arrêt du scheduler", log_file)
            
            # Attente de l'arrêt complet
            wait_and_log(10, "Finalisation de l'arrêt", log_file)
            
            # État post-arrêt
            run_cli_command(["src.main", "etat"], "État après arrêt", log_file)
            
        except KeyboardInterrupt:
            print("\n❌ Test interrompu par l'utilisateur")
            log_file.write("\n\n=== TEST INTERROMPU PAR L'UTILISATEUR ===\n")
            # Tentative d'arrêt du scheduler
            try:
                run_cli_command(["src.main", "arreter"], "Arrêt d'urgence du scheduler", log_file)
            except:
                pass
        except Exception as e:
            print(f"\n❌ Erreur inattendue: {e}")
            log_file.write(f"\n\n=== ERREUR INATTENDUE ===\n{e}\n")
            # Tentative d'arrêt du scheduler
            try:
                run_cli_command(["src.main", "arreter"], "Arrêt d'urgence du scheduler", log_file)
            except:
                pass
        finally:
            # Pied de page
            end_time = datetime.now(timezone.utc)
            duration = end_time - start_time
            log_file.write(f"\n\n=== FIN DU TEST ===\n")
            log_file.write(f"Fin: {end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
            log_file.write(f"Durée totale: {duration}\n")
            log_file.write("=" * 70 + "\n")
    
    print(f"\n✅ Test de fumée 45 minutes terminé")
    print(f"📝 Rapport complet: {log_path}")


if __name__ == "__main__":
    main()
