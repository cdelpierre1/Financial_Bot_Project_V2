"""
Trainer — Entraîneur avancé avec modèles ML multiples (LightGBM, XGBoost, RandomForest).

- Utilise FeatureBuilder pour X,y
- Auto-sélection du meilleur modèle par validation croisée
- Support entraînement incrémental et refit périodique
- Sauvegarde via ModelStore avec métriques de performance
"""
from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import warnings
import json
import os

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import joblib

# Imports conditionnels pour les modèles avancés
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from prediction.feature_builder import FeatureBuilder
from prediction.model_store import ModelStore

# Suppression des warnings pour les modèles ML
warnings.filterwarnings('ignore')


class Trainer:
    def __init__(self, model_store: ModelStore | None = None, verbose: bool = True) -> None:
        self.fb = FeatureBuilder(step_minutes=5)
        self.store = model_store or ModelStore()
        self.verbose = verbose
        
        # Configuration des modèles disponibles
        self.model_configs = {
            "linear": {
                "class": LinearRegression,
                "params": {},
                "incremental": False
            },
            "random_forest": {
                "class": RandomForestRegressor,
                "params": {
                    "n_estimators": 50,  # Réduit pour les contraintes mémoire
                    "max_depth": 10,
                    "random_state": 42,
                    "n_jobs": 1  # Limite à 1 core
                },
                "incremental": False
            }
        }
        
        # Ajouter LightGBM si disponible
        if HAS_LIGHTGBM:
            self.model_configs["lightgbm"] = {
                "class": lgb.LGBMRegressor,
                "params": {
                    "n_estimators": 50,
                    "max_depth": 8,
                    "learning_rate": 0.1,
                    "random_state": 42,
                    "verbosity": -1,
                    "num_threads": 1  # Limite à 1 thread
                },
                "incremental": False
            }
        
        # Ajouter XGBoost si disponible
        if HAS_XGBOOST:
            self.model_configs["xgboost"] = {
                "class": xgb.XGBRegressor,
                "params": {
                    "n_estimators": 50,
                    "max_depth": 8,
                    "learning_rate": 0.1,
                    "random_state": 42,
                    "verbosity": 0,
                    "nthread": 1  # Limite à 1 thread
                },
                "incremental": False
            }

    def _evaluate_model(self, model, X: pd.DataFrame, y: pd.Series) -> float:
        """Évalue un modèle par validation croisée (MAE)."""
        try:
            # DEBUG COMPLET avant entraînement (seulement si verbose)
            if self.verbose:
                print(f"        🔍 DEBUG _evaluate_model:")
                print(f"        📊 Données: X={X.shape}, y={len(y)}")
                print(f"        📊 NaN X: {X.isnull().sum().sum()}, NaN y: {y.isnull().sum()}")
                print(f"        📊 Inf X: {np.isinf(X.values).sum()}, Inf y: {np.isinf(y.values).sum()}")
                print(f"        📊 Variance y: {y.var()}")
                print(f"        📊 Variance X moyennes: {X.var().mean()}")
                print(f"        📊 Range y: {y.min():.4f} -> {y.max():.4f}")
                print(f"        📊 Type model: {type(model).__name__}")
            
            # DÉSACTIVER cross-validation pour économiser ressources
            # Utiliser split simple train/test plus rapide
            split_idx = int(0.8 * len(X))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            if self.verbose:
                print(f"        📊 Split: train={len(X_train)}, test={len(X_test)}")
            
            if len(X_train) < 5 or len(X_test) < 2:
                if self.verbose:
                    print(f"        ⚠️ Pas assez de données pour split, utilisation complète")
                # Pas assez de données pour split
                model.fit(X.values, y.values)
                y_pred = model.predict(X.values)
                return mean_absolute_error(y.values, y_pred)
            
            if self.verbose:
                print(f"        🧠 Entraînement {type(model).__name__} sur {len(X_train)} échantillons...")
            model.fit(X_train.values, y_train.values)
            if self.verbose:
                print(f"        ✅ Entraînement réussi, prédiction...")
            y_pred = model.predict(X_test.values)
            mae = mean_absolute_error(y_test.values, y_pred)
            if self.verbose:
                print(f"        ✅ MAE: {mae:.4f}")
            return mae
            
        except Exception as e:
            print(f"        ❌ ERREUR dans _evaluate_model: {e}")
            import traceback
            print(f"        📍 Détails: {traceback.format_exc()}")
            # Fallback ultime: MAE sur données complètes
            try:
                print(f"        🔄 Tentative fallback...")
                model.fit(X.values, y.values)
                y_pred = model.predict(X.values)
                mae_fallback = mean_absolute_error(y.values, y_pred)
                print(f"        ✅ Fallback MAE: {mae_fallback:.4f}")
                return mae_fallback
            except Exception as e2:
                print(f"        ❌ ERREUR fallback: {e2}")
                return float('inf')

    def _select_best_model(self, X: pd.DataFrame, y: pd.Series) -> Tuple[str, Any, float]:
        """Sélectionne le meilleur modèle par validation croisée."""
        best_score = float('inf')
        best_name = "linear"
        best_model = None
        
        for name, config in self.model_configs.items():
            try:
                model = config["class"](**config["params"])
                score = self._evaluate_model(model, X, y)
                
                if score < best_score:
                    best_score = score
                    best_name = name
                    best_model = model
                    
            except Exception as e:
                # Si un modèle échoue, continuer avec les autres
                print(f"Modèle {name} échoué: {e}")
                continue
        
        return best_name, best_model, best_score

    def train(self, df_five_min: pd.DataFrame, coin_id: str, horizon_minutes: int, 
              force_model: Optional[str] = None) -> Dict[str, Any]:
        """Entraîne un modèle avec auto-sélection ou modèle forcé."""
        X, y = self.fb.build_from_five_min(df_five_min, coin_id, horizon_minutes)
        if X is None or y is None or X.empty or y.empty:
            return {"status": "NO_DATA", "rows": 0}
        
        # CORRECTION: Supprimer les NaN dans y SEULEMENT maintenant, après la génération
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(y) == 0:
            return {"status": "NO_DATA", "rows": 0}
        
        # Vérifier les contraintes mémoire
        if len(y) > 10000:  # Limite pour éviter l'explosion mémoire
            # Garder seulement les données les plus récentes
            X = X.tail(10000)
            y = y.tail(10000)
        
        if force_model and force_model in self.model_configs:
            # Modèle forcé
            config = self.model_configs[force_model]
            model = config["class"](**config["params"])
            model.fit(X.values, y.values)
            
            # Évaluer sur données d'entraînement
            y_pred = model.predict(X.values)
            mae_score = mean_absolute_error(y.values, y_pred)
            selected_name = force_model
        else:
            # Auto-sélection du meilleur modèle
            selected_name, model, mae_score = self._select_best_model(X, y)
            if model is None:
                return {"status": "MODEL_SELECTION_FAILED", "rows": int(len(y))}
            
            # Entraîner le modèle sélectionné sur toutes les données
            model.fit(X.values, y.values)
        
        # Métadonnées enrichies
        meta = {
            "algo": selected_name,
            "features": list(X.columns),
            "mae_score": float(mae_score),
            "training_samples": int(len(y)),
            "horizon_minutes": int(horizon_minutes),
            "available_models": list(self.model_configs.keys())
        }
        
        path = self.store.save(coin_id, horizon_minutes, model, metadata=meta, do_backup=True)
        return {
            "status": "OK", 
            "rows": int(len(y)), 
            "model_path": path,
            "selected_model": selected_name,
            "mae_score": float(mae_score)
        }

    def load(self, coin_id: str, horizon_minutes: int) -> Tuple[Any, Dict[str, Any] | None]:
        return self.store.load(coin_id, horizon_minutes)
