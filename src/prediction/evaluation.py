"""
Évaluation post-horizon (MVP)

Fonctions:
- save_prediction: persist une prédiction émise (statut implicite PENDING)
- EvaluationJob.evaluate_pending: quand l'horizon est atteint, calcule l'erreur
  à partir du realized price et écrit un résultat dans le dataset eval_results.

Datasets Parquet:
- predictions: lignes immuables avec prediction_id, ts_pred, target_ts, coin_id, horizon, mid_price_usd, value_pred (optionnel)
- eval_results: lignes avec prediction_id, realized_mid_usd, abs_error_pct

Remarques:
- Pas de mise à jour en place du dataset predictions; l'association se fait via prediction_id.
- Réalisation: on cherche le prix réalisé autour de target_ts dans five_min; fallback prix courant.
"""
from __future__ import annotations

import os
import json
import time
import uuid
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd

from storage.parquet_writer import ParquetWriter
from collectors.price_collector import PriceCollector


ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(ROOT, "config")


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _settings() -> Dict[str, Any]:
    return _load_json(os.path.join(CONFIG_DIR, "settings.json"))


def _now_ms() -> int:
    return int(time.time() * 1000)


def generate_prediction_id() -> str:
    return str(uuid.uuid4())


def save_prediction(
    *,
    prediction_id: str,
    coin_id: str,
    horizon_minutes: int,
    ts_pred_utc_ms: int,
    target_ts_utc_ms: int,
    mid_price_usd: float,
    value_pred: Optional[float],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Écrit une ligne dans le dataset Parquet 'predictions'."""
    s = _settings()
    writer = ParquetWriter()
    record = {
        "ts_utc_ms": int(ts_pred_utc_ms),  # utilisé par ParquetWriter pour la partition date
        "prediction_id": prediction_id,
        "coin_id": coin_id,
        "horizon_minutes": int(horizon_minutes),
        "target_ts_utc_ms": int(target_ts_utc_ms),
        "mid_price_usd": float(mid_price_usd),
        "value_pred": float(value_pred) if value_pred is not None else None,
    }
    if extra:
        # Aplatir quelques champs utiles (best-effort)
        for k in ["erreur_attendue_pct", "spread_pct", "fx_rate_usd_per_eur"]:
            if k in extra and extra[k] is not None:
                record[k] = float(extra[k])
    df = pd.DataFrame([record])
    writer.write("predictions", df, dedup_keys=["prediction_id"], partition_cols=None)


class EvaluationJob:
    def __init__(self) -> None:
        self.s = _settings()
        self.data_root = self.s["paths"]["data_parquet"]
        self.writer = ParquetWriter()
        self.pc = PriceCollector()

    def _read_parquet_dir(self, dataset: str) -> Optional[pd.DataFrame]:
        path = os.path.join(self.data_root, dataset)
        if not os.path.isdir(path):
            return None
        try:
            # Tentative direct (le moteur pyarrow peut lire un dataset partitionné)
            df = pd.read_parquet(path)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception:
            df = None
        # Fallback: lecture récursive de tous les fichiers .parquet sous le dossier
        frames: list[pd.DataFrame] = []
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".parquet"):
                    fp = os.path.join(root, f)
                    try:
                        frames.append(pd.read_parquet(fp))
                    except Exception:
                        pass
        if frames:
            try:
                out = pd.concat(frames, ignore_index=True)
                return out if not out.empty else None
            except Exception:
                return None
        return None

    def _realized_mid_usd(self, coin_id: str, target_ts_ms: int) -> Optional[float]:
        """Trouve le prix réalisé autour de target_ts dans five_min; fallback au prix courant."""
        df5 = self._read_parquet_dir("five_min")
        if df5 is not None and not df5.empty and "ts_utc_ms" in df5.columns:
            try:
                sub = df5[df5["coin_id"] == coin_id]
                if not sub.empty:
                    # fenêtre ±10 minutes (600000 ms)
                    w = 600000
                    sub = sub[(sub["ts_utc_ms"] >= target_ts_ms - w) & (sub["ts_utc_ms"] <= target_ts_ms + w)]
                    if not sub.empty:
                        # Choisit le point le plus proche du target_ts
                        sub = sub.assign(_dist=(sub["ts_utc_ms"] - target_ts_ms).abs())
                        row = sub.sort_values("_dist").iloc[0]
                        # five_min stocke o=h=l=c=price; on prend 'c' si présent sinon 'price_usd' ou 'o'
                        for col in ["c", "price_usd", "o"]:
                            if col in row and pd.notna(row[col]):
                                return float(row[col])
            except Exception:
                pass
        # Fallback: prix courant
        try:
            payload = self.pc.get_prices([coin_id])
            return float(payload.get("prices", {}).get(coin_id, 0.0)) or None
        except Exception:
            return None

    def evaluate_pending(self) -> int:
        """Évalue les prédictions échues et écrit 'eval_results'. Retourne le nombre traités."""
        preds = self._read_parquet_dir("predictions")
        if preds is None or preds.empty:
            return 0
        # filtre 'due'
        now_ms = _now_ms()
        due = preds[preds["target_ts_utc_ms"] <= now_ms].copy()
        if due.empty:
            return 0
        # exclure celles déjà évaluées
        eval_df = self._read_parquet_dir("eval_results")
        done_ids: set = set(eval_df["prediction_id"].unique()) if (eval_df is not None and not eval_df.empty) else set()
        todo = due[~due["prediction_id"].isin(done_ids)]
        if todo.empty:
            return 0

        results: List[Dict[str, Any]] = []
        for _, row in todo.iterrows():
            pred_id = str(row["prediction_id"])
            coin_id = str(row["coin_id"])
            target_ts = int(row["target_ts_utc_ms"])
            pred_mid = float(row.get("mid_price_usd", 0.0) or 0.0)
            realized = self._realized_mid_usd(coin_id, target_ts)
            if realized is None or pred_mid <= 0:
                continue
            abs_err_pct = abs(realized - pred_mid) / pred_mid
            results.append({
                "ts_utc_ms": int(target_ts),  # ancrage sur le target
                "prediction_id": pred_id,
                "coin_id": coin_id,
                "horizon_minutes": int(row.get("horizon_minutes", 0)),
                "pred_mid_price_usd": pred_mid,
                "realized_mid_price_usd": float(realized),
                "abs_error_pct": float(abs_err_pct),
            })

        if not results:
            return 0
        df_out = pd.DataFrame(results)
        self.writer.write("eval_results", df_out, dedup_keys=["prediction_id"], partition_cols=None)
        return len(results)
