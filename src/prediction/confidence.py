"""
Incertitude & erreur attendue avec calibration sur données historiques

Améliorations:
- Calcul MAE réelle basée sur données d'évaluation historiques
- Estimation d'incertitude calibrée sur variance observée
- Combinaison intelligente des différentes sources d'erreur
- Cache des métriques pour optimiser les performances
"""
from __future__ import annotations

from typing import Dict, Optional
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from prediction.threshold_policy import interpolate_error_threshold


ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(ROOT, "config")


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _settings() -> Dict:
    return _load_json(os.path.join(CONFIG_DIR, "settings.json"))


def _thresholds_cfg() -> Dict:
    return _load_json(os.path.join(CONFIG_DIR, "thresholds.json"))


class ConfidenceEstimator:
    def __init__(self, thresholds_cfg: Dict | None = None) -> None:
        self.th = thresholds_cfg or _thresholds_cfg()
        self._settings = _settings()
        self._cache = {}  # Cache pour les métriques calculées
        self._cache_ttl = 300  # 5 minutes de TTL pour le cache

    def _get_eval_results_df(self) -> Optional[pd.DataFrame]:
        """Charge les résultats d'évaluation historiques."""
        try:
            parquet_path = self._settings.get("paths", {}).get("data_parquet", "src/data/parquet")
            # ROOT pointe déjà vers src/, donc on retire "src/" du chemin
            if parquet_path.startswith("src/"):
                parquet_path = parquet_path[4:]  # Enlever "src/"
            eval_path = os.path.join(ROOT, parquet_path, "eval_results")
            
            if not os.path.exists(eval_path):
                return None
                
            # Lire tous les fichiers parquet d'évaluation (un par un pour éviter conflits de schéma)
            frames = []
            for root, _, files in os.walk(eval_path):
                for file in files:
                    if file.endswith('.parquet'):
                        try:
                            file_path = os.path.join(root, file)
                            df = pd.read_parquet(file_path)
                            # Convertir date en string pour éviter conflit de schéma
                            if 'date' in df.columns:
                                df['date'] = df['date'].astype(str)
                            frames.append(df)
                        except Exception as e:
                            print(f"Warning: Could not read {file}: {e}")
                            continue
            
            if not frames:
                return None
                
            combined = pd.concat(frames, ignore_index=True)
            
            # Convertir timestamp si nécessaire
            if 'ts_utc_ms' in combined.columns:
                combined['datetime'] = pd.to_datetime(combined['ts_utc_ms'], unit='ms')
            
            return combined
            
        except Exception as e:
            print(f"Error loading eval results: {e}")
            return None

    def _calculate_historical_mae(self, coin_id: str, horizon_minutes: int, days_back: int = 30) -> Optional[float]:
        """Calcule la MAE historique réelle sur les N derniers jours."""
        cache_key = f"mae_{coin_id}_{horizon_minutes}_{days_back}"
        
        # Vérifier le cache
        if cache_key in self._cache:
            cached_time, cached_value = self._cache[cache_key]
            if (datetime.now() - cached_time).seconds < self._cache_ttl:
                return cached_value
        
        df = self._get_eval_results_df()
        if df is None or df.empty:
            return None
            
        try:
            # Extraire coin_id et horizon depuis prediction_id format: "bitcoin_10_timestamp"
            def extract_coin_horizon(pred_id):
                parts = pred_id.split('_')
                if len(parts) >= 2:
                    coin = parts[0]
                    try:
                        horizon = int(parts[1])
                        return coin, horizon
                    except ValueError:
                        return None, None
                return None, None
            
            # Ajouter colonnes extraites
            df[['extracted_coin', 'extracted_horizon']] = df['prediction_id'].apply(
                lambda x: pd.Series(extract_coin_horizon(x))
            )
            
            # Filtrer par coin et horizon
            mask = (df['extracted_coin'] == coin_id) & (df['extracted_horizon'] == horizon_minutes)
            filtered = df[mask].copy()
            
            if filtered.empty:
                return None
            
            # Garder seulement les derniers jours
            if 'datetime' in filtered.columns:
                cutoff = datetime.now() - timedelta(days=days_back)
                filtered = filtered[filtered['datetime'] >= cutoff]
            
            if filtered.empty or 'error_pct' not in filtered.columns:
                return None
                
            # Calculer la MAE moyenne pondérée (plus de poids aux données récentes)
            filtered = filtered.sort_values('datetime') if 'datetime' in filtered.columns else filtered
            weights = np.linspace(0.5, 1.0, len(filtered))  # Poids croissants
            mae = np.average(filtered['error_pct'].values, weights=weights)
            
            # Mettre en cache
            self._cache[cache_key] = (datetime.now(), float(mae))
            return float(mae)
            
        except Exception:
            return None

    def _calculate_historical_volatility(self, coin_id: str, horizon_minutes: int, days_back: int = 30) -> Optional[float]:
        """Calcule la volatilité historique des erreurs de prédiction."""
        cache_key = f"vol_{coin_id}_{horizon_minutes}_{days_back}"
        
        # Vérifier le cache
        if cache_key in self._cache:
            cached_time, cached_value = self._cache[cache_key]
            if (datetime.now() - cached_time).seconds < self._cache_ttl:
                return cached_value
        
        df = self._get_eval_results_df()
        if df is None or df.empty:
            return None
            
        try:
            # Extraire coin_id et horizon depuis prediction_id format: "bitcoin_10_timestamp"
            def extract_coin_horizon(pred_id):
                parts = pred_id.split('_')
                if len(parts) >= 2:
                    coin = parts[0]
                    try:
                        horizon = int(parts[1])
                        return coin, horizon
                    except ValueError:
                        return None, None
                return None, None
            
            # Ajouter colonnes extraites
            df[['extracted_coin', 'extracted_horizon']] = df['prediction_id'].apply(
                lambda x: pd.Series(extract_coin_horizon(x))
            )
            
            # Filtrer par coin et horizon
            mask = (df['extracted_coin'] == coin_id) & (df['extracted_horizon'] == horizon_minutes)
            filtered = df[mask].copy()
            
            if filtered.empty or 'error_pct' not in filtered.columns:
                return None
            
            # Garder seulement les derniers jours
            if 'datetime' in filtered.columns:
                cutoff = datetime.now() - timedelta(days=days_back)
                filtered = filtered[filtered['datetime'] >= cutoff]
            
            if len(filtered) < 5:  # Besoin d'au moins 5 points pour calculer la volatilité
                return None
                
            volatility = float(filtered['error_pct'].std())
            
            # Mettre en cache
            self._cache[cache_key] = (datetime.now(), volatility)
            return volatility
            
        except Exception:
            return None

    def mae_pct(self, coin_id: str, horizon_minutes: int) -> float:
        """MAE calibrée: utilise données historiques si disponibles, sinon seuil interpolé."""
        historical_mae = self._calculate_historical_mae(coin_id, horizon_minutes)
        
        if historical_mae is not None and historical_mae > 0:
            # Appliquer un facteur de sécurité de 10% sur la MAE historique
            return min(historical_mae * 1.1, 0.10)  # Cap à 10%
        
        # Fallback: utiliser le seuil interpolé
        return float(interpolate_error_threshold(horizon_minutes, self.th))

    def uncertainty_pct(self, coin_id: str, horizon_minutes: int) -> float:
        """Incertitude calibrée basée sur la volatilité historique des erreurs."""
        historical_vol = self._calculate_historical_volatility(coin_id, horizon_minutes)
        
        if historical_vol is not None and historical_vol > 0:
            # L'incertitude est approximativement 1.5x la volatilité historique (intervalle ~90%)
            uncertainty = historical_vol * 1.5
            return min(uncertainty, 0.08)  # Cap à 8%
        
        # Fallback: fraction du seuil interpolé
        base = float(interpolate_error_threshold(horizon_minutes, self.th))
        return 0.8 * base

    def expected_error_pct(self, coin_id: str, horizon_minutes: int, costs_pct: float) -> float:
        """Erreur attendue totale avec calibration sur données historiques."""
        # Composantes d'erreur
        mae = self.mae_pct(coin_id, horizon_minutes)
        uncertainty = self.uncertainty_pct(coin_id, horizon_minutes)
        threshold = float(interpolate_error_threshold(horizon_minutes, self.th))
        
        # Combinaison intelligente: moyenne pondérée des composantes
        # Plus de poids sur la MAE historique si disponible
        historical_mae = self._calculate_historical_mae(coin_id, horizon_minutes)
        if historical_mae is not None:
            # 60% MAE historique, 30% incertitude, 10% seuil
            core_error = 0.6 * mae + 0.3 * uncertainty + 0.1 * threshold
        else:
            # 40% seuil, 40% incertitude, 20% MAE (estimée)
            core_error = 0.4 * threshold + 0.4 * uncertainty + 0.2 * mae
        
        # Ajouter les coûts
        total_error = core_error + max(0.0, float(costs_pct))
        
        # Appliquer les bornes
        return max(0.0, min(1.0, total_error))

    def ci_bounds(self, mid_price_usd: float, coin_id: str, horizon_minutes: int) -> tuple[float, float]:
        """Calcule p10/p90 calibrés avec incertitude historique."""
        uncertainty = self.uncertainty_pct(coin_id, horizon_minutes)
        
        # Utiliser une distribution asymétrique pour les cryptos (plus de risque baissier)
        downside_factor = 1.2  # 20% plus de risque à la baisse
        upside_factor = 0.8    # 20% moins de risque à la hausse
        
        price = float(mid_price_usd)
        lo = price * (1.0 - uncertainty * downside_factor)
        hi = price * (1.0 + uncertainty * upside_factor)
        
        return max(0.0, lo), hi

    def get_confidence_metrics(self, coin_id: str, horizon_minutes: int) -> Dict[str, float]:
        """Retourne toutes les métriques de confiance pour diagnostic."""
        historical_mae = self._calculate_historical_mae(coin_id, horizon_minutes)
        historical_vol = self._calculate_historical_volatility(coin_id, horizon_minutes)
        
        return {
            "historical_mae_pct": historical_mae if historical_mae is not None else -1,
            "historical_volatility_pct": historical_vol if historical_vol is not None else -1,
            "calibrated_mae_pct": self.mae_pct(coin_id, horizon_minutes),
            "calibrated_uncertainty_pct": self.uncertainty_pct(coin_id, horizon_minutes),
            "threshold_pct": float(interpolate_error_threshold(horizon_minutes, self.th)),
            "has_historical_data": historical_mae is not None
        }
