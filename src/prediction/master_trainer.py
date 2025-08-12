"""
Master Model Trainer — Système de modèle maître combinant daily/hourly/5min

Architecture:
- 3 sous-modèles spécialisés par timeframe (daily, hourly, 5min)
- 1 modèle maître qui apprend à combiner leurs prédictions
- Auto-entraînement continu et adaptation dynamique aux conditions de marché
"""
from __future__ import annotations

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from prediction.trainer import Trainer
from prediction.model_store import ModelStore
from prediction.feature_builder import FeatureBuilder
from storage.parquet_writer import ParquetWriter


class MasterModelTrainer:
    """
    Entraîneur de modèles maîtres pour système hybride multi-timeframe.
    
    Combine les prédictions de sous-modèles spécialisés (daily, hourly, 5min)
    en un modèle maître plus robuste et précis.
    """
    
    def __init__(self):
        self.trainer = Trainer()
        self.store = ModelStore()
        self.fb = FeatureBuilder()
        self.pw = ParquetWriter()
    
    def train_sub_models(self, df: pd.DataFrame, coin_id: str, horizon_minutes: int) -> Dict[str, Any]:
        """Entraîne les 3 sous-modèles (daily, hourly, 5min) pour un horizon donné."""
        results = {"sub_models": {}, "errors": []}
        
        timeframes = ['daily', 'hourly', '5min']
        
        for timeframe in timeframes:
            try:
                print(f"    🧠 Entraînement sous-modèle {timeframe}...")
                
                # Entraîner le sous-modèle
                result = self.trainer.train(df, coin_id, horizon_minutes)
                
                print(f"      🔍 Debug result: {result}")
                
                if result.get("status") == "OK" and (result.get("mae") is not None or result.get("mae_score") is not None):
                    # Récupérer MAE (peut être "mae" ou "mae_score")
                    mae = result.get("mae") or result.get("mae_score")
                    print(f"      ✅ Sous-modèle {timeframe}: MAE={mae:.4f}")
                    
                    # Sauvegarder avec nom spécifique au timeframe
                    model_path = result.get("path") or result.get("model_path")
                    timeframe_path = model_path.replace(f"__{horizon_minutes}m.pkl", f"__{timeframe}_{horizon_minutes}m.pkl")
                    
                    # Copier le modèle vers le path spécifique au timeframe
                    import shutil
                    shutil.copy2(model_path, timeframe_path)
                    
                    results["sub_models"][timeframe] = {
                        "path": timeframe_path,
                        "mae": mae,
                        "status": "OK"
                    }
                else:
                    status = result.get("status", "UNKNOWN")
                    print(f"      ❌ Échec sous-modèle {timeframe} - Status: {status}")
                    if "rows" in result:
                        print(f"      📊 Données disponibles: {result['rows']} lignes")
                    results["errors"].append(f"Échec sous-modèle {timeframe}: {status}")
                    
            except Exception as e:
                print(f"      ❌ ERREUR sous-modèle {timeframe}: {e}")
                import traceback
                print(f"      📍 Détails: {traceback.format_exc()}")
                results["errors"].append(f"Erreur sous-modèle {timeframe}: {e}")
        
        return results
    
    def train_master_model(self, coin_id: str, test_data: Dict[str, pd.DataFrame], 
                          horizon_minutes: int = 10) -> Dict[str, Any]:
        """Entraîne le modèle maître qui combine les prédictions des sous-modèles."""
        try:
            print(f"    🎯 Entraînement modèle maître...")
            
            # Collecter toutes les prédictions des sous-modèles
            sub_predictions = {}
            all_features = {}
            all_targets = {}
            
            # Phase 1: Charger tous les sous-modèles et générer leurs prédictions
            for timeframe, df in test_data.items():
                if df.empty:
                    continue
                    
                try:
                    # Charger le sous-modèle
                    timeframe_path = self.store.model_path(coin_id, horizon_minutes).replace(f"__{horizon_minutes}m.pkl", f"__{timeframe}_{horizon_minutes}m.pkl")
                    
                    print(f"      🔍 Recherche sous-modèle: {timeframe_path}")
                    if not os.path.exists(timeframe_path):
                        print(f"      ❌ Sous-modèle {timeframe} introuvable: {timeframe_path}")
                        continue
                    
                    print(f"      📂 Chargement sous-modèle {timeframe}...")
                    import pickle
                    with open(timeframe_path, "rb") as f:
                        sub_model = pickle.load(f)
                    
                    print(f"      🔧 Génération features...")
                    # Générer features et targets
                    X, y = self.fb.build_from_five_min(df, coin_id, horizon_minutes)
                    if X is None or y is None or X.empty or y.empty:
                        print(f"      ⚠️ Pas de features pour {timeframe}")
                        continue
                    
                    # CORRECTION: Filtrer les NaN dans y pour les prédictions
                    valid_mask = ~y.isna()
                    X_valid = X[valid_mask]
                    y_valid = y[valid_mask]
                    
                    if len(y_valid) == 0:
                        print(f"      ⚠️ Pas de targets valides pour {timeframe}")
                        continue
                    
                    print(f"      🎯 Prédictions {timeframe} sur {len(X_valid)} échantillons...")
                    # Prédictions du sous-modèle
                    predictions = sub_model.predict(X_valid.values)
                    
                    # Stocker les prédictions et données
                    sub_predictions[timeframe] = predictions
                    all_features[timeframe] = X_valid
                    all_targets[timeframe] = y_valid
                    
                    print(f"      ✅ {timeframe}: {len(predictions)} prédictions OK")
                    
                except Exception as e:
                    print(f"      ❌ ERREUR sous-modèle {timeframe}: {e}")
                    import traceback
                    print(f"      📍 Détails: {traceback.format_exc()}")
                    continue
            
            if len(sub_predictions) < 2:
                print(f"      ❌ Pas assez de sous-modèles: {len(sub_predictions)} < 2")
                return {"status": "INSUFFICIENT_MODELS", "models": len(sub_predictions)}
            
            # Phase 2: Construire les features du modèle maître de manière alignée
            timeframes = list(sub_predictions.keys())
            min_samples = min(len(sub_predictions[tf]) for tf in timeframes)
            
            print(f"      📊 Alignement sur {min_samples} échantillons")
            print(f"      🔍 Debug: Sous-modèles utilisés = {timeframes}")
            
            # Construire features maître: [features_originales + prédictions_sous_modèles]
            base_timeframe = timeframes[0]  # Utiliser le premier timeframe comme référence
            base_features = all_features[base_timeframe].iloc[:min_samples].values
            base_targets = all_targets[base_timeframe].iloc[:min_samples].values
            
            # Ajouter les prédictions de tous les sous-modèles comme features
            prediction_features = []
            for tf in timeframes:
                pred_column = sub_predictions[tf][:min_samples].reshape(-1, 1)
                prediction_features.append(pred_column)
            
            # Combiner features originales + toutes les prédictions
            if prediction_features:
                all_predictions = np.column_stack(prediction_features)
                master_features = np.column_stack([base_features, all_predictions])
            else:
                master_features = base_features
            
            master_targets = base_targets
            
            print(f"      📊 Données maître: {master_features.shape}")
            print(f"      🔍 Debug: Features shape = {master_features.shape}")
            print(f"      🔍 Debug: Targets shape = {master_targets.shape}")
            
            if len(master_features) < 10:
                print(f"      ❌ Pas assez de données maître: {len(master_features)} < 10")
                return {"status": "INSUFFICIENT_DATA", "samples": len(master_features)}
            
            print(f"      🧠 Entraînement RandomForest...")
            # Utiliser RandomForest pour le modèle maître
            master_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=1)
            master_model.fit(master_features, master_targets)
            
            # Évaluer
            mae_score = mean_absolute_error(master_targets, master_model.predict(master_features))
            
            # Sauvegarder modèle maître
            master_path = self.store.model_path(coin_id, horizon_minutes).replace(f"__{horizon_minutes}m.pkl", f"__master_{horizon_minutes}m.pkl")
            
            meta = {
                "algo": "master_random_forest",
                "horizon_minutes": horizon_minutes,
                "mae_score": float(mae_score),
                "training_samples": len(master_features),
                "sub_models": list(sub_predictions.keys()),
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            import pickle
            with open(master_path, "wb") as f:
                pickle.dump(master_model, f)
            
            # Sauvegarder métadonnées
            meta_path = master_path.replace(".pkl", "_meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            
            print(f"      ✅ Modèle maître: MAE={mae_score:.4f}, samples={len(master_features)}")
            
            return {
                "status": "OK",
                "path": master_path,
                "mae": float(mae_score),
                "samples": len(master_features)
            }
            
        except Exception as e:
            print(f"      ❌ ERREUR modèle maître: {e}")
            import traceback
            print(f"      📍 Détails: {traceback.format_exc()}")
            return {"status": "ERROR", "error": str(e)}
    
    def train_full_pipeline(self, coin_id: str, daily_df: pd.DataFrame, hourly_df: pd.DataFrame,
                           five_min_df: pd.DataFrame, horizon_minutes: int = 10, 
                           split_ratio: float = 0.7) -> Dict[str, Any]:
        """Entraînement complet: sous-modèles + modèle maître."""
        results = {"sub_models": {}, "master_model": {}, "errors": []}
        
        try:
            # Entraîner les sous-modèles
            sub_results = self.train_sub_models(
                five_min_df, coin_id, horizon_minutes
            )
            results["sub_models"] = sub_results["sub_models"]
            results["errors"].extend(sub_results["errors"])
            
            # Diviser les données pour test du modèle maître
            daily_split = self._split_data(daily_df, split_ratio)
            hourly_split = self._split_data(hourly_df, split_ratio) 
            five_min_split = self._split_data(five_min_df, split_ratio)
            
            test_data = {
                "daily": daily_split.get("test", pd.DataFrame()),
                "hourly": hourly_split.get("test", pd.DataFrame()),
                "5min": five_min_split.get("test", pd.DataFrame())
            }
            
            master_result = self.train_master_model(coin_id, test_data, horizon_minutes)
            results["master_model"] = master_result
            
            return results
            
        except Exception as e:
            results["errors"].append(f"Erreur pipeline complet: {e}")
            return results
    
    def _split_data(self, df: pd.DataFrame, split_ratio: float) -> Dict[str, pd.DataFrame]:
        """Division train/test des données."""
        if df.empty:
            return {"train": pd.DataFrame(), "test": pd.DataFrame()}
        
        split_idx = int(len(df) * split_ratio)
        return {
            "train": df.iloc[:split_idx].copy(),
            "test": df.iloc[split_idx:].copy()
        }
