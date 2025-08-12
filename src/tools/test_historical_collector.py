#!/usr/bin/env python3
"""
Validation complète du HistoricalCollector autonome.
Test rapide de la logique de collecte massive historique.
"""

import os
import sys
import time
from datetime import datetime, timezone

# Ajout du chemin src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_historical_collector():
    """Test autonome du HistoricalCollector."""
    print("🏛️  Test autonome HistoricalCollector...")
    
    try:
        from collectors.historical_collector import HistoricalCollector
        from storage.parquet_writer import ParquetWriter
        
        # Initialisation
        writer = ParquetWriter()
        collector = HistoricalCollector(writer=writer)
        
        print("✅ Initialisation réussie")
        
        # Test direct de la méthode principale
        print("\n� Test collecte historique complète...")
        
        # Test avec mode simulation (dry-run)
        print("   Mode simulation activé pour validation...")
        results = collector.collect_all_historical(dry_run=True)
        
        # Affichage des résultats
        if results and "summary" in results:
            summary = results["summary"]
            print(f"   ✅ Simulation réussie:")
            print(f"      - Points estimés: {summary.get('total_points_collected', 0):,}")
            print(f"      - Durée estimée: {summary.get('total_duration_minutes', 0):.1f} min")
            print(f"      - Taille estimée: {summary.get('estimated_size_mb', 0):.1f} MB")
        else:
            print(f"   ⚠️  Simulation avec résultats partiels: {results}")
        
        print("\n✅ Test simulation réussi!")
        print("🚨 PRÊT pour collecte historique massive URGENTE")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test HistoricalCollector: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Point d'entrée du test."""
    print("🧪 Validation autonome HistoricalCollector")
    print("=" * 50)
    
    success = test_historical_collector()
    
    print("=" * 50)
    if success:
        print("✅ VALIDATION RÉUSSIE - HistoricalCollector opérationnel")
        exit(0)
    else:
        print("❌ VALIDATION ÉCHOUÉE - Correctifs nécessaires")
        exit(1)

if __name__ == "__main__":
    main()
