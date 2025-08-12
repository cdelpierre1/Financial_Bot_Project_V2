"""
Prediction Tracker - Suivi et vérification des prédictions CLI

Fonctionnalités :
- Stockage des prédictions faites par l'utilisateur
- Vérification automatique après le délai prévu
- Calcul des erreurs et ajustement des modèles
- Feedback loop pour amélioration continue
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd

from storage.parquet_writer import ParquetWriter
from prediction.model_store import ModelStore
from prediction.trainer import Trainer


class PredictionTracker:
    def __init__(self):
        self.writer = ParquetWriter()
        self.store = ModelStore()
        self.trainer = Trainer(verbose=False)  # Mode silencieux pour les ajustements
        
    def save_prediction(
        self, 
        coin_id: str, 
        horizon_minutes: int, 
        current_price: float,
        predicted_price: float,
        model_algo: str,
        user_session_id: str = "cli"
    ) -> str:
        """Sauvegarde une prédiction pour vérification future."""
        
        prediction_id = f"{coin_id}_{horizon_minutes}_{int(time.time() * 1000)}"
        prediction_time = datetime.now(timezone.utc)
        verification_time = prediction_time + timedelta(minutes=horizon_minutes)
        
        prediction_data = {
            "prediction_id": prediction_id,
            "ts_utc_ms": int(prediction_time.timestamp() * 1000),
            "verification_ts_utc_ms": int(verification_time.timestamp() * 1000),
            "coin_id": coin_id,
            "horizon_minutes": horizon_minutes,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "model_algo": model_algo,
            "user_session_id": user_session_id,
            "status": "pending",  # pending, verified, failed
            "actual_price": None,
            "error_pct": None,
            "created_at_utc": prediction_time.isoformat()
        }
        
        # Convertir en DataFrame et sauvegarder
        df = pd.DataFrame([prediction_data])
        
        try:
            self.writer.append_data("predictions", df)
            print(f"📝 Prédiction sauvegardée (ID: {prediction_id})")
            local_verification_time = verification_time.astimezone()
            print(f"⏰ Vérification prévue à: {local_verification_time.strftime('%H:%M:%S')} (heure locale)")
            return prediction_id
        except Exception as e:
            print(f"❌ Erreur sauvegarde prédiction: {e}")
            return ""
    
    def check_pending_predictions(self) -> List[Dict[str, Any]]:
        """Vérifie les prédictions en attente dont le délai est écoulé."""
        
        try:
            # Charger toutes les prédictions
            data_path = self.writer.settings["paths"]["data_parquet"]
            predictions_path = os.path.join(data_path, "predictions")
            
            if not os.path.exists(predictions_path):
                return []
            
            # Lire les fichiers de prédictions
            dfs = []
            for root, dirs, files in os.walk(predictions_path):
                for file in files:
                    if file.endswith('.parquet'):
                        try:
                            file_path = os.path.join(root, file)
                            df_part = pd.read_parquet(file_path)
                            dfs.append(df_part)
                        except Exception:
                            continue
            
            if not dfs:
                return []
            
            df_predictions = pd.concat(dfs, ignore_index=True)
            
            # Filtrer les prédictions en attente dont le délai est écoulé
            now_ms = int(time.time() * 1000)
            pending_predictions = df_predictions[
                (df_predictions['status'] == 'pending') & 
                (df_predictions['verification_ts_utc_ms'] <= now_ms)
            ]
            
            return pending_predictions.to_dict('records')
            
        except Exception as e:
            print(f"❌ Erreur lecture prédictions: {e}")
            return []
    
    def verify_prediction(self, prediction: Dict[str, Any]) -> bool:
        """Vérifie une prédiction spécifique et calcule l'erreur."""
        
        try:
            coin_id = prediction['coin_id']
            predicted_price = prediction['predicted_price']
            
            # Obtenir le prix actuel réel
            actual_price = self._get_actual_price(coin_id)
            
            if actual_price is None:
                print(f"❌ Prix actuel non disponible pour {coin_id}")
                return False
            
            # Calculer l'erreur
            error_pct = abs((predicted_price - actual_price) / actual_price) * 100
            
            # Mettre à jour la prédiction
            self._update_prediction_status(
                prediction['prediction_id'],
                actual_price,
                error_pct,
                'verified'
            )
            
            print(f"✅ Prédiction vérifiée {coin_id}:")
            print(f"   🎯 Prédit: ${predicted_price:.4f}")
            print(f"   💰 Réel: ${actual_price:.4f}")
            print(f"   📊 Erreur: {error_pct:.2f}%")
            
            # Si erreur > 5%, déclencher un réajustement
            if error_pct > 5.0:
                print(f"⚠️ Erreur élevée ({error_pct:.2f}%) - Réajustement du modèle...")
                self._retrain_model_for_error(prediction)
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur vérification prédiction: {e}")
            return False
    
    def _get_actual_price(self, coin_id: str) -> Optional[float]:
        """Récupère le prix actuel réel depuis les données five_min."""
        
        try:
            data_path = self.writer.settings["paths"]["data_parquet"]
            five_min_path = os.path.join(data_path, "five_min")
            
            # Lire les données récentes
            dfs = []
            for root, dirs, files in os.walk(five_min_path):
                for file in files:
                    if file.endswith('.parquet'):
                        try:
                            file_path = os.path.join(root, file)
                            df_part = pd.read_parquet(file_path)
                            if 'coin_id' in df_part.columns:
                                df_coin_part = df_part[df_part['coin_id'] == coin_id]
                                if not df_coin_part.empty:
                                    dfs.append(df_coin_part)
                        except Exception:
                            continue
            
            if not dfs:
                return None
            
            df_coin = pd.concat(dfs, ignore_index=True).sort_values("ts_utc_ms")
            
            if df_coin.empty:
                return None
            
            # Retourner le prix de clôture le plus récent
            return df_coin.iloc[-1]["c"]
            
        except Exception:
            return None
    
    def _update_prediction_status(
        self, 
        prediction_id: str, 
        actual_price: float, 
        error_pct: float, 
        status: str
    ):
        """Met à jour le statut d'une prédiction."""
        
        # Pour simplifier, on crée une nouvelle entrée avec les résultats
        # En production, on pourrait modifier le fichier existant
        update_data = {
            "prediction_id": prediction_id + "_result",
            "ts_utc_ms": int(time.time() * 1000),
            "original_prediction_id": prediction_id,
            "actual_price": actual_price,
            "error_pct": error_pct,
            "status": status,
            "verified_at_utc": datetime.now(timezone.utc).isoformat()
        }
        
        df = pd.DataFrame([update_data])
        try:
            self.writer.append_data("eval_results", df)
        except Exception as e:
            print(f"❌ Erreur mise à jour prédiction: {e}")
    
    def _retrain_model_for_error(self, prediction: Dict[str, Any]):
        """Réentraîne un modèle spécifique en cas d'erreur élevée."""
        
        try:
            coin_id = prediction['coin_id']
            horizon_minutes = prediction['horizon_minutes']
            
            print(f"🔄 Réentraînement ciblé: {coin_id} horizon {horizon_minutes}min")
            
            # Charger les données récentes pour ce coin
            data_path = self.writer.settings["paths"]["data_parquet"]
            five_min_path = os.path.join(data_path, "five_min")
            
            dfs = []
            for root, dirs, files in os.walk(five_min_path):
                for file in files:
                    if file.endswith('.parquet'):
                        try:
                            file_path = os.path.join(root, file)
                            df_part = pd.read_parquet(file_path)
                            if 'coin_id' in df_part.columns:
                                df_coin_part = df_part[df_part['coin_id'] == coin_id]
                                if not df_coin_part.empty:
                                    dfs.append(df_coin_part)
                        except Exception:
                            continue
            
            if not dfs:
                print(f"❌ Pas de données pour réentraînement {coin_id}")
                return
            
            df_coin = pd.concat(dfs, ignore_index=True).sort_values("ts_utc_ms").tail(3000)
            
            # Réentraîner le modèle
            result = self.trainer.train(df_coin, coin_id, horizon_minutes)
            
            if result.get("status") != "NO_DATA":
                print(f"✅ Modèle réentraîné avec succès pour {coin_id}")
            else:
                print(f"❌ Échec réentraînement {coin_id}")
                
        except Exception as e:
            print(f"❌ Erreur réentraînement: {e}")
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques des prédictions."""
        
        try:
            data_path = self.writer.settings["paths"]["data_parquet"]
            eval_path = os.path.join(data_path, "eval_results")
            
            if not os.path.exists(eval_path):
                return {"total": 0, "verified": 0, "avg_error": 0}
            
            dfs = []
            for root, dirs, files in os.walk(eval_path):
                for file in files:
                    if file.endswith('.parquet'):
                        try:
                            file_path = os.path.join(root, file)
                            df_part = pd.read_parquet(file_path)
                            dfs.append(df_part)
                        except Exception:
                            continue
            
            if not dfs:
                return {"total": 0, "verified": 0, "avg_error": 0}
            
            df_results = pd.concat(dfs, ignore_index=True)
            
            verified_predictions = df_results[df_results['status'] == 'verified']
            
            stats = {
                "total": len(df_results),
                "verified": len(verified_predictions),
                "avg_error": verified_predictions['error_pct'].mean() if len(verified_predictions) > 0 else 0
            }
            
            return stats
            
        except Exception:
            return {"total": 0, "verified": 0, "avg_error": 0}
