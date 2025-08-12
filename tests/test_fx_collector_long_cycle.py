"""
Test fonctionnel du collecteur FX sur une période prolongée (60 minutes)

Ce test vérifie que le collecteur FX fonctionne correctement sur une durée prolongée
de 60 minutes, avec des appels à ExchangeRate-API correctement espacés.

IMPORTANT: Ce test est désactivé par défaut car il prend 60 minutes à exécuter.
Pour l'activer, définissez RUN_LONG_FX_TEST=1 dans votre environnement.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
import pytest


@pytest.mark.skipif(
    os.environ.get("RUN_LONG_FX_TEST") != "1", 
    reason="Test long de 60 minutes désactivé par défaut. Activez avec RUN_LONG_FX_TEST=1"
)
def test_fx_collector_60min_cycle():
    """Test que le collecteur FX fonctionne correctement pendant 60 min"""
    from src.collectors.fx_collector import FxCollector
    
    # Pour éviter d'écrire en Parquet pendant le test
    collector = FxCollector(writer=None)
    
    # Durée totale du test: 60 minutes
    end_time = time.time() + 60 * 60
    
    # Collecte initiale
    start_ts = int(datetime.now(timezone.utc).timestamp())
    df, _ = collector.collect(write=False)
    
    # Vérifier que la collecte initiale a fonctionné
    assert not df.empty, "La collecte initiale devrait retourner des données"
    assert "rate_usd_per_eur" in df.columns
    assert df["rate_usd_per_eur"].iloc[0] is not None
    
    # Log initial
    print(f"[{datetime.now()}] Test FX démarré, valeur initiale EUR/USD: {df['rate_usd_per_eur'].iloc[0]}")
    
    # Nombre d'appels réussis
    success_count = 1  # Déjà une collecte réussie
    
    # Cycles de collecte toutes les 5 minutes
    while time.time() < end_time:
        # Attendre 5 minutes entre les collectes (simuler le scheduler)
        time.sleep(300)
        
        try:
            # Collecter à nouveau
            df, _ = collector.collect(write=False)
            
            # Vérifier les résultats
            assert not df.empty, "La collecte devrait retourner des données"
            assert "rate_usd_per_eur" in df.columns
            assert df["rate_usd_per_eur"].iloc[0] is not None
            
            # Incrémenter le compteur de succès
            success_count += 1
            
            # Log
            print(f"[{datetime.now()}] Collecte #{success_count} réussie, valeur EUR/USD: {df['rate_usd_per_eur'].iloc[0]}")
        
        except Exception as e:
            # Log de l'erreur
            print(f"[{datetime.now()}] ERREUR de collecte: {str(e)}")
            
    # Le test doit avoir fait au moins 12 collectes (une toutes les 5 min pendant 60 min)
    assert success_count >= 12, f"Le test devrait faire au moins 12 collectes (a fait {success_count})"
    
    elapsed_sec = int(time.time() - start_ts)
    print(f"[{datetime.now()}] Test FX terminé: {success_count} collectes en {elapsed_sec} secondes")
