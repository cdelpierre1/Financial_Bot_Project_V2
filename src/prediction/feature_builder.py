"""
FeatureBuilder — construction de features avancées à partir du dataset five_min.

Améliorations:
- Features techniques: RSI, moyennes mobiles, volatilité
- Features temporelles: heure, jour de la semaine
- Features de momentum et tendance
- Normalisation et scaling appropriés
"""
from __future__ import annotations

from typing import Tuple
import math
import pandas as pd
import numpy as np


class FeatureBuilder:
    def __init__(self, step_minutes: int = 5) -> None:
        self.step_minutes = max(1, int(step_minutes))

    def _calculate_rsi(self, prices: pd.Series, periods: int = 14) -> pd.Series:
        """Calcule le RSI (Relative Strength Index)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_bollinger_bands(self, prices: pd.Series, periods: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcule les bandes de Bollinger."""
        ma = prices.rolling(window=periods).mean()
        std = prices.rolling(window=periods).std()
        upper = ma + (2 * std)
        lower = ma - (2 * std)
        return ma, upper, lower

    def _calculate_volatility(self, prices: pd.Series, periods: int = 20) -> pd.Series:
        """Calcule la volatilité glissante."""
        returns = prices.pct_change()
        return returns.rolling(window=periods).std() * np.sqrt(periods)

    def build_from_five_min(self, df: pd.DataFrame, coin_id: str, horizon_minutes: int) -> Tuple[pd.DataFrame, pd.Series]:
        if df is None or df.empty:
            return pd.DataFrame(), pd.Series(dtype=float)
        
        need_cols = {"ts_utc_ms", "coin_id", "c"}
        # Colonnes optionnelles pour features avancées
        optional_cols = {"o", "h", "l", "volume"}
        
        if not need_cols.issubset(df.columns):
            raise ValueError(f"Colonnes requises manquantes: {sorted(need_cols - set(df.columns))}")
        
        steps = max(1, int(horizon_minutes // self.step_minutes))
        g = df[df["coin_id"] == coin_id].sort_values("ts_utc_ms").copy()
        if g.empty:
            return pd.DataFrame(), pd.Series(dtype=float)
        
        # Convertir timestamp pour features temporelles
        g["datetime"] = pd.to_datetime(g["ts_utc_ms"], unit='ms')
        
        # Features de base
        g["lag1"] = g["c"].shift(1)
        g["lag5"] = g["c"].shift(5)
        g["diff1"] = g["c"] - g["lag1"]
        g["ret1"] = g["c"].pct_change()
        
        # Moyennes mobiles
        g["ma5"] = g["c"].rolling(window=5).mean()
        g["ma20"] = g["c"].rolling(window=20).mean()
        g["ma_ratio"] = g["ma5"] / g["ma20"]
        
        # Features techniques
        g["rsi"] = self._calculate_rsi(g["c"])
        g["volatility"] = self._calculate_volatility(g["c"])
        
        # Bandes de Bollinger
        bb_ma, bb_upper, bb_lower = self._calculate_bollinger_bands(g["c"])
        g["bb_position"] = (g["c"] - bb_lower) / (bb_upper - bb_lower)
        
        # Features temporelles
        g["hour"] = g["datetime"].dt.hour
        g["day_of_week"] = g["datetime"].dt.dayofweek
        g["hour_sin"] = np.sin(2 * np.pi * g["hour"] / 24)
        g["hour_cos"] = np.cos(2 * np.pi * g["hour"] / 24)
        g["dow_sin"] = np.sin(2 * np.pi * g["day_of_week"] / 7)
        g["dow_cos"] = np.cos(2 * np.pi * g["day_of_week"] / 7)
        
        # Features de volume si disponible
        if "volume" in g.columns:
            g["volume_ma"] = g["volume"].rolling(window=20).mean()
            g["volume_ratio"] = g["volume"] / g["volume_ma"]
        else:
            g["volume_ratio"] = 1.0  # Valeur par défaut
        
        # Features OHLC si disponibles
        if all(col in g.columns for col in ["o", "h", "l"]):
            g["hl_ratio"] = g["h"] / g["l"]
            g["oc_ratio"] = g["o"] / g["c"]
            g["body_size"] = abs(g["c"] - g["o"]) / g["o"]
        else:
            g["hl_ratio"] = 1.0
            g["oc_ratio"] = 1.0
            g["body_size"] = 0.0
        
        # Momentum features
        g["momentum_5"] = g["c"] / g["c"].shift(5) - 1
        g["momentum_20"] = g["c"] / g["c"].shift(20) - 1
        
        # Cible future
        g["target_future"] = g["c"].shift(-steps)
        g["y"] = (g["target_future"] - g["c"]) / g["c"]
        
        # CORRECTION FINALE: On garde TOUTES les données, même celles avec target NaN
        # Le split train/test va gérer ça naturellement !
        # Les derniers points sans targets ne seront pas dans le training
        
        # Supprimer seulement les NaN dans les features de base (pas les targets)
        g = g.dropna(subset=["c", "lag1", "lag5"])
        
        if len(g) == 0:
            print(f"        ⚠️ Aucune donnée avec features valides")
            return None, None
        
        # Sélection des features finales
        feature_cols = [
            "c", "lag1", "lag5", "diff1", "ret1",
            "ma5", "ma20", "ma_ratio",
            "rsi", "volatility", "bb_position",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "volume_ratio", "hl_ratio", "oc_ratio", "body_size",
            "momentum_5", "momentum_20"
        ]
        
        # Normalisation relative au prix courant pour certaines features
        price_relative_cols = ["lag1", "lag5", "ma5", "ma20", "diff1"]
        for col in price_relative_cols:
            if col in g.columns:
                g[f"{col}_norm"] = g[col] / g["c"]
                feature_cols.append(f"{col}_norm")
        
        # Conserver uniquement les lignes où les features et la cible sont présentes
        mask = (~g["lag1"].isna()) & (~g["y"].isna())
        
        # Filtrer les features existantes dans le DataFrame
        available_features = [col for col in feature_cols if col in g.columns]
        
        # LIMITER LA TAILLE pour éviter overflow mémoire
        filtered_data = g.loc[mask, available_features]
        if len(filtered_data) > 5000:  # Limiter à 5000 échantillons max
            filtered_data = filtered_data.tail(5000)
            mask_limited = filtered_data.index
            X = filtered_data.copy()
            y = g.loc[mask_limited, "y"].astype(float)
        else:
            X = filtered_data.copy()
            y = g.loc[mask, "y"].astype(float)
        
        # Remplacer les NaN par 0 dans les features
        X = X.fillna(0)
        
        return X, y
