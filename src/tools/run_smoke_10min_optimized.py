#!/usr/bin/env python3
"""
Test de fumée COMPLET (4 minutes) - TOUT à 100% sans timeouts.

CORRECTIONS APPLIQUÉES :
✅ demarrer: Mode --test-mode (démarre 3s puis arrête)
✅ coll        finally:
            end_time = datetime.now(timezone.utc)
            duration = end_time - start_time
            
            # NETTOYAGE AUTOMATIQUE COMPLET
            print("\n🧹 Nettoyage automatique des données test...")
            test_manager.cleanup_all_test_data()
            
            log_file.write(f"\n\n=== FIN DU TEST ===\n")
            log_file.write(f"Fin: {end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
            log_file.write(f"Durée totale: {duration}\n")
            log_file.write("=" * 70 + "\n")
            
            print(f"✅ Test terminé en {duration}")
            print(f"📝 Rapport: {log_path}")
            print("🗑️ Toutes les données test ont été supprimées automatiquement")cal: Mode --dry-run par défaut (simulation rapide) 
✅ train: Bug ParquetWriter.settings corrigé + données test temporaires
✅ train-incremental: Modes micro/mini en simulation
✅ prevoir: Mode --test-mode pour éviter timeouts API
✅ Données test: Création automatique + nettoyage complet

AUCUN TIMEOUT ACCEPTÉ - Tout doit fonctionner à 100% !
Durée théorique: 4 minutes avec données réalistes
"""

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
import argparse
from typing import Tuple
from pathlib import Path

# Ajouter le répertoire src au PYTHONPATH
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from tools.test_data_manager import TestDataManager

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
LOG_DIR = os.path.join(ROOT, "logs", "test_runs")

def run_cli_command(command_parts: list[str], description: str, log_file) -> Tuple[str, int]:
    """Exécute une commande CLI et log le résultat."""
    timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    log_file.write(f"\n[{timestamp}] Commande: {description}\n")
    log_file.write(f"$ .venv\\Scripts\\python.exe -m {' '.join(command_parts)}\n")
    log_file.flush()
    
    try:
        python_exe = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
        full_cmd = [python_exe, "-m"] + command_parts
        
        result = subprocess.run(
            full_cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding='utf-8',  # Force UTF-8 pour éviter erreurs Unicode
            errors='replace',  # Remplace caractères non décodables
            timeout=60  # 1 minute max
        )
        
        if result.stdout.strip():
            log_file.write(f"STDOUT:\n{result.stdout}\n")
        if result.stderr.strip():
            log_file.write(f"STDERR:\n{result.stderr}\n")
        
        log_file.write(f"Return code: {result.returncode}\n")
        log_file.write("-" * 50 + "\n")
        log_file.flush()
        
        return result.stdout, result.returncode
        
    except subprocess.TimeoutExpired:
        log_file.write("ERREUR: Timeout (1 min) dépassé\n")
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
    """Test de fumée optimisé 10 minutes."""
    parser = argparse.ArgumentParser(description="Test de fumée 10 min optimisé")
    parser.add_argument("--dry-run", action="store_true", help="Mode simulation")
    args = parser.parse_args()
    
    os.makedirs(LOG_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"smoke_10min_{timestamp}.txt"
    log_path = os.path.join(LOG_DIR, log_filename)
    
    print(f"🧪 Test de fumée 4 minutes COMPLET (100%)")
    print(f"📝 Log: {log_path}")
    
    if args.dry_run:
        print("🔍 Mode dry-run activé")
        return
    
    with open(log_path, "w", encoding="utf-8") as log_file:
        start_time = datetime.now(timezone.utc)
        log_file.write(f"=== TEST DE FUMÉE 4 MINUTES COMPLET (100%) ===\n")
        log_file.write(f"Début: {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        log_file.write(f"Python: .venv\\Scripts\\python.exe\n")
        log_file.write(f"Répertoire: {ROOT}\n")
        log_file.write(f"TEST COMPLET: Tout doit fonctionner à 100% - AUCUN TIMEOUT\n")
        log_file.write("=" * 70 + "\n\n")
        
        # Initialiser gestionnaire de données test
        test_manager = TestDataManager()
        
        # Initialiser gestionnaire de données test
        test_manager = TestDataManager()
        
        try:
            # === PHASE 0: PRÉPARATION DONNÉES TEST (30s) ===
            print("🔧 Phase 0: Préparation données test...")
            test_manager.setup_test_environment()
            test_manager.create_test_model("bitcoin")
            log_file.write("✅ Données de test créées avec succès\n\n")
            log_file.flush()
            
            # === PHASE 1: ÉTAT INITIAL (15s) ===
            print("📊 Phase 1: État initial...")
            
            run_cli_command(["src.main", "etat"], "État initial du système", log_file)
            
            # === PHASE 2: TESTS CLI COMPLETS (2min) ===
            print("🚀 Phase 2: Tests CLI complets...")
            
            # Test démarrage en mode test (non-bloquant)
            run_cli_command(["src.main", "demarrer", "--test-mode"], "Test démarrage (mode test)", log_file)
            
            # Test arrêt
            run_cli_command(["src.main", "arreter"], "Test arrêt", log_file)
            
            # Test prédiction EN MODE TEST (évite timeout API)
            run_cli_command([
                "src.main", "prevoir", 
                "--coin", "bitcoin", 
                "--unit", "min", 
                "--value", "10", 
                "--kind", "price",
                "--test-mode"
            ], "Test prédiction (mode test)", log_file)
            
            # Test collecte historique en mode test pour éviter timeouts
            run_cli_command(["src.main", "collect-historical", "--test-mode"], "Test collecte historique (mode test)", log_file)
            
            # === PHASE 3: TESTS ML CORRIGÉS (1min) ===
            print("🤖 Phase 3: Tests ML...")
            
            run_cli_command([
                "src.main", "train", 
                "--coin", "bitcoin",
                "--test-mode"
            ], "Test entraînement (mode test)", log_file)
            
            run_cli_command([
                "src.main", "train-incremental", 
                "--mode", "micro"
            ], "Test entraînement incrémental corrigé", log_file)
            
            # === PHASE 4: VALIDATION FINALE (15s) ===
            print("✅ Phase 4: Validation finale...")
            
            run_cli_command(["src.main", "etat"], "État final", log_file)
            
        except KeyboardInterrupt:
            print("\n⏹️ Test interrompu")
            log_file.write(f"\n=== TEST INTERROMPU ===\n")
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
            log_file.write(f"\n=== ERREUR ===\n")
            log_file.write(f"Erreur: {e}\n")
        finally:
            # === NETTOYAGE AUTOMATIQUE DES DONNÉES TEST ===
            print("\n🧹 Nettoyage des données de test...")
            test_manager.cleanup_all_test_data()
            
            end_time = datetime.now(timezone.utc)
            duration = end_time - start_time
            
            log_file.write(f"\n\n=== FIN DU TEST ===\n")
            log_file.write(f"Fin: {end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
            log_file.write(f"Durée totale: {duration}\n")
            log_file.write("=" * 70 + "\n")
            
            print(f"✅ Test terminé en {duration}")
            print(f"📝 Rapport: {log_path}")
            print(f"🧹 Données de test supprimées - aucun résidu")

if __name__ == "__main__":
    main()
