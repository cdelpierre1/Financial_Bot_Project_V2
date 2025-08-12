"""
Test Data Manager - Générateur de données de test temporaires avec nettoyage automatique

Crée des données de test réalistes pour les smoke tests puis les supprime automatiquement.
Aucun résidu ne reste après les tests.
"""
from __future__ import annotations

import os
import json
import shutil
import tempfile
from typing import Dict, List, Tuple
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

from storage.parquet_writer import ParquetWriter


class TestDataManager:
    def __init__(self):
        self.writer = ParquetWriter()
        self.test_paths: List[str] = []
        self.test_models: List[str] = []
        self.created_files: List[str] = []
        
    def create_fake_five_min_data(self, coin_id: str = "bitcoin", hours: int = 48) -> pd.DataFrame:
        """Crée des données five_min factices réalistes"""
        print(f"🧪 Création données test five_min pour {coin_id} ({hours}h)")
        
        # Point de départ réaliste pour Bitcoin
        base_price = 45000.0
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(hours=hours)
        
        # Générer timestamps toutes les 5 minutes
        timestamps = []
        current = start_time
        while current <= now:
            timestamps.append(int(current.timestamp() * 1000))
            current += timedelta(minutes=5)
        
        # Générer prix réalistes avec volatilité
        np.random.seed(42)  # Reproductible
        prices = [base_price]
        
        for i in range(1, len(timestamps)):
            # Volatilité réaliste Bitcoin: ±0.5% par 5min
            change_pct = np.random.normal(0, 0.005)
            new_price = prices[-1] * (1 + change_pct)
            prices.append(max(1000, new_price))  # Prix minimum 1000$
        
        # Créer DataFrame avec schéma five_min EXACT
        data = []
        for i, (ts, price) in enumerate(zip(timestamps, prices)):
            # OHLC simulé autour du prix
            volatility = price * 0.001  # 0.1% volatility intra-5min
            
            high = price + np.random.uniform(0, volatility)
            low = price - np.random.uniform(0, volatility)
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            
            # Volume réaliste (millions USD)
            volume = np.random.uniform(50_000_000, 200_000_000)
            
            data.append({
                "ts_utc_ms": int(ts),  # Explicitement int64
                "coin_id": str(coin_id),  # Explicitement string
                "o": float(round(open_price, 2)),  # Explicitement float64
                "h": float(round(high, 2)),
                "l": float(round(low, 2)),
                "c": float(round(close_price, 2)),
                "volume": float(round(volume, 2)),
                "agg_method": "5min"  # Toujours string
            })
        
        # Créer DataFrame avec types explicites
        df = pd.DataFrame(data)
        
        # Forcer les types pour éviter incompatibilités Parquet
        df["ts_utc_ms"] = df["ts_utc_ms"].astype("int64")
        df["coin_id"] = df["coin_id"].astype("string")  
        df["o"] = df["o"].astype("float64")
        df["h"] = df["h"].astype("float64")
        df["l"] = df["l"].astype("float64")
        df["c"] = df["c"].astype("float64")
        df["volume"] = df["volume"].astype("float64")
        df["agg_method"] = df["agg_method"].astype("string")
        
        print(f"✅ Données test créées: {len(df)} points sur {hours}h")
        print(f"📊 Types: {df.dtypes.to_dict()}")
        return df
    
    def setup_test_environment(self) -> None:
        """Configure environnement de test avec données factices"""
        print("🔧 Configuration environnement de test...")
        
        # NETTOYER les données existantes pour éviter conflits de schéma
        self._cleanup_existing_data()
        
        # Créer données five_min pour Bitcoin
        df_bitcoin = self.create_fake_five_min_data("bitcoin", hours=48)
        
        # Sauvegarder temporairement
        written = self.writer.write("five_min", df_bitcoin, 
                                   dedup_keys=["coin_id","ts_utc_ms"], 
                                   partition_cols=None)
        print(f"💾 Données écrites: {written} points")
        
        # Enregistrer pour nettoyage
        data_path = self.writer.settings["paths"]["data_parquet"]
        five_min_path = os.path.join(data_path, "five_min")
        if os.path.exists(five_min_path):
            self.test_paths.append(five_min_path)
            self.created_files.append(five_min_path)
    
    def _cleanup_existing_data(self) -> None:
        """Nettoie TOUTES les données existantes pour éviter conflits de schéma"""
        data_path = self.writer.settings["paths"]["data_parquet"]
        
        print("🧹 Nettoyage COMPLET de toutes les données existantes...")
        
        # Supprimer TOUS les datasets pour éviter conflits
        datasets = ["five_min", "hourly", "daily", "markets", "tickers_spread", "fx", "predictions", "eval_results"]
        
        for dataset in datasets:
            dataset_path = os.path.join(data_path, dataset)
            if os.path.exists(dataset_path):
                try:
                    shutil.rmtree(dataset_path)
                    print(f"🗑️ Supprimé: {dataset}")
                except Exception as e:
                    print(f"⚠️ Erreur suppression {dataset}: {e}")
        
        # Supprimer le répertoire racine data_parquet et le recréer propre
        if os.path.exists(data_path):
            try:
                shutil.rmtree(data_path)
                os.makedirs(data_path, exist_ok=True)
                print("✅ Répertoire données recréé proprement")
            except Exception as e:
                print(f"⚠️ Erreur recréation: {e}")
    
    def create_test_model(self, coin_id: str = "bitcoin") -> None:
        """Crée un modèle de test factice"""
        from prediction.model_store import ModelStore
        
        print(f"🤖 Création modèle test pour {coin_id}")
        
        store = ModelStore()
        
        # Modèle factice (juste un dictionnaire)
        fake_model = {
            "type": "test_model",
            "version": "1.0.0", 
            "created_for_test": True,
            "coefficients": [1.0, 2.0, 3.0],  # Factices
            "bias": 0.5
        }
        
        # Metadata factice
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "test_data": True,
            "mae": 0.001,  # Très bon pour le test
            "mse": 0.000001,
            "r2": 0.99
        }
        
        # Sauvegarder temporairement
        model_path = store.save(coin_id, 10, fake_model, metadata=metadata, do_backup=False)
        self.test_models.append(model_path)
        print(f"💾 Modèle test sauvé: {model_path}")
    
    def cleanup_all_test_data(self) -> None:
        """Supprime TOUT les données et modèles de test"""
        print("\n🧹 NETTOYAGE COMPLET des données de test...")
        
        # Supprimer données test
        for path in self.test_paths:
            if os.path.exists(path):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                    print(f"🗑️ Supprimé: {path}")
                except Exception as e:
                    print(f"❌ Erreur suppression {path}: {e}")
        
        # Supprimer modèles test
        for model_path in self.test_models:
            if os.path.exists(model_path):
                try:
                    os.remove(model_path)
                    print(f"🗑️ Modèle supprimé: {model_path}")
                except Exception as e:
                    print(f"❌ Erreur suppression modèle {model_path}: {e}")
        
        # Supprimer autres fichiers créés
        for file_path in self.created_files:
            if os.path.exists(file_path) and file_path not in self.test_paths:
                try:
                    if os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    else:
                        os.remove(file_path)
                    print(f"🗑️ Fichier supprimé: {file_path}")
                except Exception as e:
                    print(f"❌ Erreur suppression {file_path}: {e}")
        
        # Reset des listes
        self.test_paths.clear()
        self.test_models.clear() 
        self.created_files.clear()
        
        print("✅ Nettoyage complet terminé - aucun résidu de test")


def main():
    """Test du gestionnaire de données"""
    manager = TestDataManager()
    
    try:
        # Créer environnement test
        manager.setup_test_environment()
        manager.create_test_model("bitcoin")
        
        print("\n📊 Données de test créées avec succès")
        print("Les données seront automatiquement supprimées...")
        
    finally:
        # Nettoyage automatique
        manager.cleanup_all_test_data()


if __name__ == "__main__":
    main()
