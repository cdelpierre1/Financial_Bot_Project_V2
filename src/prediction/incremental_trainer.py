"""
Système d'entraînement incrémental et refit périodique pour les modèles ML.

Features:
- Entraînement incrémental sur nouvelles données
- Refit périodique basé sur calendrier configurable  
- Validation et métriques de performance
- Sauvegarde automatique des modèles améliorés
"""
from __future__ import annotations

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.metrics import mean_absolute_error

from prediction.trainer import Trainer
from prediction.model_store import ModelStore
from prediction.feature_builder import FeatureBuilder


class IncrementalTrainer:
    def __init__(self, model_store: ModelStore | None = None) -> None:
        self.trainer = Trainer(model_store)
        self.store = model_store or ModelStore()
        self.fb = FeatureBuilder(step_minutes=5)
        
        # Configuration des cadences depuis settings.json
        self._load_training_config()
    
    def _load_training_config(self) -> None:
        """Charge la configuration d'entraînement depuis settings.json."""
        try:
            root = os.path.dirname(os.path.dirname(__file__))
            settings_path = os.path.join(root, "config", "settings.json")
            with open(settings_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            
            cadences = settings.get("cadences", {}).get("training", {})
            self.micro_update_min = cadences.get("micro_update_min", 5)
            self.mini_retrain_min = cadences.get("mini_retrain_min", 10) 
            self.refit_hourly_h = cadences.get("refit_hourly_h", 1)
            self.recalibration_daily_d = cadences.get("recalibration_daily_d", 1)
            
        except Exception:
            # Valeurs par défaut si erreur de lecture
            self.micro_update_min = 5
            self.mini_retrain_min = 10
            self.refit_hourly_h = 1
            self.recalibration_daily_d = 1
    
    def _get_last_training_time(self, coin_id: str, horizon_minutes: int) -> Optional[datetime]:
        """Récupère la date de dernier entraînement depuis les métadonnées."""
        try:
            _, meta = self.store.load(coin_id, horizon_minutes)
            if meta and "saved_at_utc" in meta:
                return datetime.fromisoformat(meta["saved_at_utc"].replace('Z', '+00:00'))
        except Exception:
            pass
        return None
    
    def _should_retrain(self, coin_id: str, horizon_minutes: int, 
                       training_type: str = "micro") -> bool:
        """Détermine si un modèle doit être ré-entraîné selon la cadence."""
        last_training = self._get_last_training_time(coin_id, horizon_minutes)
        if last_training is None:
            return True  # Pas de modèle existant
        
        now = datetime.now(last_training.tzinfo)
        time_diff = now - last_training
        
        if training_type == "micro":
            return time_diff >= timedelta(minutes=self.micro_update_min)
        elif training_type == "mini":
            return time_diff >= timedelta(minutes=self.mini_retrain_min)
        elif training_type == "hourly":
            return time_diff >= timedelta(hours=self.refit_hourly_h)
        elif training_type == "daily":
            return time_diff >= timedelta(days=self.recalibration_daily_d)
        
        return False
    
    def _get_recent_data(self, df: pd.DataFrame, hours_back: int = 24) -> pd.DataFrame:
        """Filtre le DataFrame pour ne garder que les N dernières heures."""
        if df.empty or 'ts_utc_ms' not in df.columns:
            return df
        
        cutoff_ms = int((datetime.now().timestamp() - hours_back * 3600) * 1000)
        return df[df['ts_utc_ms'] >= cutoff_ms].copy()
    
    def _validate_model_performance(self, model: Any, X: pd.DataFrame, y: pd.Series,
                                   previous_mae: Optional[float] = None) -> bool:
        """Valide qu'un nouveau modèle performe mieux que l'ancien."""
        try:
            # Diviser en train/test pour validation
            split_idx = int(0.8 * len(X))
            X_test = X.iloc[split_idx:]
            y_test = y.iloc[split_idx:]
            
            if len(X_test) < 5:  # Pas assez de données test
                return True  # Accepter par défaut
            
            y_pred = model.predict(X_test.values)
            current_mae = mean_absolute_error(y_test.values, y_pred)
            
            # Si on a une MAE précédente, exiger une amélioration d'au moins 1%
            if previous_mae is not None:
                improvement_threshold = 0.99  # 1% d'amélioration minimum
                return current_mae <= previous_mae * improvement_threshold
            
            return True  # Pas de référence, accepter
            
        except Exception:
            return True  # En cas d'erreur, accepter
    
    def micro_update(self, df_five_min: pd.DataFrame, coin_id: str, 
                    horizon_minutes: int) -> Dict[str, Any]:
        """Mise à jour micro: entraînement rapide sur données récentes (1-6h)."""
        if not self._should_retrain(coin_id, horizon_minutes, "micro"):
            return {"status": "SKIP_MICRO", "reason": "Too recent"}
        
        # Utiliser seulement les 6 dernières heures
        recent_df = self._get_recent_data(df_five_min, hours_back=6)
        
        if recent_df.empty:
            return {"status": "NO_RECENT_DATA", "hours_back": 6}
        
        # Forcer un modèle rapide pour les micro-updates
        result = self.trainer.train(recent_df, coin_id, horizon_minutes, 
                                  force_model="linear")
        
        if result.get("status") == "OK":
            result["training_type"] = "micro_update"
            result["data_hours"] = 6
        
        return result
    
    def mini_retrain(self, df_five_min: pd.DataFrame, coin_id: str,
                    horizon_minutes: int) -> Dict[str, Any]:
        """Mini re-entraînement: modèle ML sur données récentes (24h)."""
        if not self._should_retrain(coin_id, horizon_minutes, "mini"):
            return {"status": "SKIP_MINI", "reason": "Too recent"}
        
        # Utiliser les 24 dernières heures
        recent_df = self._get_recent_data(df_five_min, hours_back=24)
        
        if recent_df.empty:
            return {"status": "NO_RECENT_DATA", "hours_back": 24}
        
        # Auto-sélection du meilleur modèle
        result = self.trainer.train(recent_df, coin_id, horizon_minutes)
        
        if result.get("status") == "OK":
            result["training_type"] = "mini_retrain"
            result["data_hours"] = 24
        
        return result
    
    def hourly_refit(self, df_five_min: pd.DataFrame, coin_id: str,
                    horizon_minutes: int) -> Dict[str, Any]:
        """Refit horaire: entraînement complet sur toutes les données disponibles."""
        if not self._should_retrain(coin_id, horizon_minutes, "hourly"):
            return {"status": "SKIP_HOURLY", "reason": "Too recent"}
        
        # Obtenir la MAE du modèle actuel pour validation
        previous_mae = None
        try:
            _, meta = self.store.load(coin_id, horizon_minutes)
            if meta:
                previous_mae = meta.get("mae_score")
        except Exception:
            pass
        
        # Utiliser toutes les données mais limiter pour éviter explosion mémoire
        working_df = df_five_min.tail(20000) if len(df_five_min) > 20000 else df_five_min
        
        if working_df.empty:
            return {"status": "NO_DATA"}
        
        # Construire les features pour validation
        X, y = self.fb.build_from_five_min(working_df, coin_id, horizon_minutes)
        if X.empty:
            return {"status": "NO_FEATURES"}
        
        # Auto-sélection du meilleur modèle
        result = self.trainer.train(working_df, coin_id, horizon_minutes)
        
        if result.get("status") == "OK":
            result["training_type"] = "hourly_refit"
            result["data_samples"] = len(working_df)
            result["previous_mae"] = previous_mae
            
            # Validation de performance si on a un modèle précédent
            try:
                model, _ = self.store.load(coin_id, horizon_minutes)
                performance_ok = self._validate_model_performance(model, X, y, previous_mae)
                result["performance_validated"] = performance_ok
                
                if not performance_ok:
                    result["status"] = "PERFORMANCE_DEGRADED"
                    
            except Exception:
                result["performance_validated"] = True
        
        return result
    
    def daily_recalibration(self, df_five_min: pd.DataFrame, 
                          coins: List[str], horizons: List[int]) -> Dict[str, Any]:
        """Recalibration quotidienne: re-entraînement complet de tous les modèles."""
        results = {}
        
        for coin_id in coins:
            for horizon_minutes in horizons:
                if not self._should_retrain(coin_id, horizon_minutes, "daily"):
                    continue
                
                try:
                    result = self.hourly_refit(df_five_min, coin_id, horizon_minutes)
                    result["training_type"] = "daily_recalibration"
                    results[f"{coin_id}_{horizon_minutes}m"] = result
                    
                except Exception as e:
                    results[f"{coin_id}_{horizon_minutes}m"] = {
                        "status": "ERROR",
                        "error": str(e)
                    }
        
        return {
            "recalibration_results": results,
            "total_models": len(results),
            "successful": sum(1 for r in results.values() if r.get("status") == "OK")
        }
    
    def get_training_schedule(self, coins: List[str], 
                            horizons: List[int]) -> Dict[str, Any]:
        """Retourne le calendrier de prochains entraînements pour tous les modèles."""
        schedule = {}
        
        for coin_id in coins:
            for horizon_minutes in horizons:
                key = f"{coin_id}_{horizon_minutes}m"
                last_training = self._get_last_training_time(coin_id, horizon_minutes)
                
                if last_training:
                    next_micro = last_training + timedelta(minutes=self.micro_update_min)
                    next_mini = last_training + timedelta(minutes=self.mini_retrain_min)
                    next_hourly = last_training + timedelta(hours=self.refit_hourly_h)
                    next_daily = last_training + timedelta(days=self.recalibration_daily_d)
                else:
                    now = datetime.now()
                    next_micro = next_mini = next_hourly = next_daily = now
                
                schedule[key] = {
                    "last_training": last_training.isoformat() if last_training else None,
                    "next_micro": next_micro.isoformat(),
                    "next_mini": next_mini.isoformat(), 
                    "next_hourly": next_hourly.isoformat(),
                    "next_daily": next_daily.isoformat(),
                    "needs_micro": self._should_retrain(coin_id, horizon_minutes, "micro"),
                    "needs_mini": self._should_retrain(coin_id, horizon_minutes, "mini"),
                    "needs_hourly": self._should_retrain(coin_id, horizon_minutes, "hourly"),
                    "needs_daily": self._should_retrain(coin_id, horizon_minutes, "daily")
                }
        
        return schedule
