"""
Pipeline de prédiction minimale (MVP):
- Prépare les intrants (coin_id, horizon_minutes, prix courant, fx, spread estimé, amount_eur optionnel)
- Estime un coût roundtrip via CostModel
- Applique une décision via decision_engine.decide

Note: Cette pipeline ne fait pas de modélisation; value_pred/CI/erreur_attendue sont des placeholders.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import os
import json
import math

from prediction.cost_model import CostModel
from prediction.decision_engine import decide, interpolate_target_eur
from prediction.threshold_policy import interpolate_error_threshold
from prediction.confidence import ConfidenceEstimator
from collectors.tickers_collector import TickersCollector
from prediction.model_store import ModelStore
from prediction.feature_builder import FeatureBuilder
import pandas as pd


class PredictionPipeline:
	def __init__(self, model_store: ModelStore | None = None, feature_builder: FeatureBuilder | None = None) -> None:
		self.cost = CostModel()
		self._tc = TickersCollector()
		self._store = model_store or ModelStore()
		self._fb = feature_builder or FeatureBuilder(step_minutes=5)
		# Charger config coûts (marge spread, stratégie médiane)
		root = os.path.dirname(os.path.dirname(__file__))
		cfg_path = os.path.join(root, "config", "costs.json")
		try:
			with open(cfg_path, "r", encoding="utf-8") as f:
				self._costs_cfg = json.load(f)
		except Exception:
			self._costs_cfg = {}

	def _read_parquet_dir(self, dataset: str) -> Optional[pd.DataFrame]:
		"""Lecture robuste d'un dossier dataset Parquet, avec fallback récursif.
		Retourne None si introuvable.
		"""
		try:
			s = self._settings()
			root = s["paths"]["data_parquet"]
			path = os.path.join(root, dataset)
			if not os.path.isdir(path):
				return None
			try:
				return pd.read_parquet(path)
			except Exception:
				frames: list[pd.DataFrame] = []
				for r, _, files in os.walk(path):
					for f in files:
						if f.endswith(".parquet"):
							fp = os.path.join(r, f)
							try:
								frames.append(pd.read_parquet(fp))
							except Exception:
								pass
				if frames:
					return pd.concat(frames, ignore_index=True)
				return None
		except Exception:
			return None

	@staticmethod
	def _settings() -> dict:
		root = os.path.dirname(os.path.dirname(__file__))
		cfg_path = os.path.join(root, "config", "settings.json")
		with open(cfg_path, "r", encoding="utf-8") as f:
			return json.load(f)

	def _predict_with_model(self, coin_id: str, horizon_minutes: int, mid_price_usd: float, recent_five_min_df: Optional[pd.DataFrame] = None) -> Optional[float]:
		"""Si un modèle entraîné existe, produit une valeur prédite (prix) en USD.
		Retourne None si modèle/données indisponibles.
		"""
		try:
			model, _ = self._store.load(coin_id, horizon_minutes)
		except Exception:
			return None
		try:
			df = recent_five_min_df
			if df is None:
				df = self._read_parquet_dir("five_min")
			if df is None or df.empty:
				return None
			X, _ = self._fb.build_from_five_min(df, coin_id, horizon_minutes)
			if X is None or X.empty:
				return None
			# Utiliser la dernière ligne de features disponibles
			x_last = X.tail(1).values
			ret_rel = float(model.predict(x_last)[0])
			if not math.isfinite(ret_rel):
				return None
			return float(mid_price_usd) * (1.0 + ret_rel)
		except Exception:
			return None

	def _estimate_spread_pct(self, coin_id: str) -> Optional[float]:
		"""Estime un spread relatif à partir des tickers récents (TTL 10 min).

		- Utilise la médiane des spreads non filtrés si possible (paramétrable via costs.json: spread_proxy.use_median)
		- Ajoute une marge de sécurité (costs.json: spread_proxy.margin_pct)
		"""
		margin = float(self._costs_cfg.get("spread_proxy", {}).get("margin_pct", 0.002))
		use_median = bool(self._costs_cfg.get("spread_proxy", {}).get("use_median", True))
		try:
			payload = self._tc.get_tickers(coin_id)
			df = self._tc.to_dataframe(coin_id, payload)
			if df is None or df.empty:
				return None
			cand = df[df["spread_pct"].notna()]
			if "filtered" in cand.columns:
				cand = cand[cand["filtered"] == False]  # noqa: E712
			if cand.empty:
				return None
			if use_median:
				base = float(cand["spread_pct"].median())
			else:
				base = float(cand["spread_pct"].mean())
			# Propreté
			if not math.isfinite(base) or base < 0:
				return None
			return max(0.0, base + margin)
		except Exception:
			return None

	def _estimate_expected_profit(self, coin_id: str, horizon_minutes: int, amount_eur: float, 
	                            predicted_price: Optional[float], current_price: float,
	                            total_costs_eur: float) -> Optional[float]:
		"""Calcule le profit net attendu basé sur la prédiction de prix."""
		if predicted_price is None or amount_eur <= 0:
			return None
			
		try:
			# Calcul du profit brut attendu
			# profit_brut = amount_eur * (predicted_price / current_price - 1)
			expected_return_pct = (predicted_price / current_price) - 1
			gross_profit_eur = amount_eur * expected_return_pct
			
			# Profit net = profit brut - coûts
			net_profit_eur = gross_profit_eur - total_costs_eur
			
			# Ajustement conservateur: appliquer un facteur de confiance
			confidence_factor = self._get_confidence_factor(coin_id, horizon_minutes)
			conservative_profit = net_profit_eur * confidence_factor
			
			return max(0.0, conservative_profit)  # Ne retourner que les profits positifs
			
		except Exception:
			return None

	def _get_confidence_factor(self, coin_id: str, horizon_minutes: int) -> float:
		"""Facteur de confiance basé sur la qualité des prédictions historiques."""
		try:
			conf = ConfidenceEstimator()
			metrics = conf.get_confidence_metrics(coin_id, horizon_minutes)
			
			# Si on a des données historiques, ajuster selon la performance
			if metrics.get("has_historical_data", False):
				historical_mae = metrics.get("historical_mae_pct", 0.05)
				# Plus la MAE est faible, plus le facteur de confiance est élevé
				# Facteur varie entre 0.5 (MAE élevée) et 0.9 (MAE faible)
				confidence = max(0.5, min(0.9, 1.0 - historical_mae))
			else:
				# Sans données historiques, facteur conservateur
				confidence = 0.7
				
			return confidence
			
		except Exception:
			return 0.7  # Valeur par défaut conservatrice

	def calculate_required_investment(
		self,
		*,
		coin_id: str,
		horizon_minutes: int,
		target_profit_eur: float,
		mid_price_usd: float,
		fx_rate_usd_per_eur: Optional[float],
		spread_pct: Optional[float] = None,
	) -> Dict[str, Any]:
		"""Calcul inverse: combien investir pour atteindre un profit cible."""
		
		# Stratégie hybride pour prédiction
		horizon_strategy = self._get_optimal_horizon_strategy(horizon_minutes)
		
		# Prédiction de prix (même logique que run())
		if horizon_strategy["strategy"] == "exact":
			value_pred = self._predict_with_model(coin_id, horizon_strategy["horizon"], mid_price_usd)
		elif horizon_strategy["strategy"] == "interpolation":
			pred1 = self._predict_with_model(coin_id, horizon_strategy["horizon_1"], mid_price_usd)
			pred2 = self._predict_with_model(coin_id, horizon_strategy["horizon_2"], mid_price_usd)
			if pred1 is not None and pred2 is not None:
				value_pred = pred1 * horizon_strategy["weight_1"] + pred2 * horizon_strategy["weight_2"]
			else:
				value_pred = pred1 or pred2
		elif horizon_strategy["strategy"] == "extrapolation":
			base_pred = self._predict_with_model(coin_id, horizon_strategy["base_horizon"], mid_price_usd)
			if base_pred is not None:
				price_change = base_pred - mid_price_usd
				adjusted_change = price_change * horizon_strategy["factor"] * 0.8
				value_pred = mid_price_usd + adjusted_change
			else:
				value_pred = None
		else:  # fallback
			value_pred = self._predict_with_model(coin_id, horizon_strategy["horizon"], mid_price_usd)
		
		if value_pred is None or value_pred <= mid_price_usd:
			return {
				"success": False,
				"error": "Prédiction négative ou impossible pour le profit cible",
				"coin_id": coin_id,
				"target_profit_eur": target_profit_eur,
			}
		
		# Calcul du rendement attendu
		expected_return_pct = (value_pred / mid_price_usd) - 1
		
		if expected_return_pct <= 0:
			return {
				"success": False,
				"error": f"Rendement négatif prévu ({expected_return_pct:.2%})",
				"coin_id": coin_id,
				"target_profit_eur": target_profit_eur,
			}
		
		# Calcul inverse avec ajustement pour frais
		estimated_investment = target_profit_eur / expected_return_pct
		costs = self.cost.estimate_transaction_costs(
			coin_id=coin_id,
			amount_eur=estimated_investment,
			mid_price_usd=mid_price_usd,
			fx_rate_usd_per_eur=fx_rate_usd_per_eur or 1.0,
			spread_pct=spread_pct
		)
		
		# Ajustement final
		required_investment = (target_profit_eur + costs.get("total_cost_eur", 0)) / expected_return_pct
		
		return {
			"success": True,
			"coin_id": coin_id,
			"horizon_minutes": horizon_minutes,
			"target_profit_eur": target_profit_eur,
			"required_investment_eur": round(required_investment, 2),
			"expected_return_pct": round(expected_return_pct * 100, 2),
			"estimated_costs_eur": round(costs.get("total_cost_eur", 0), 2),
			"predicted_price_usd": round(value_pred, 4),
			"current_price_usd": round(mid_price_usd, 4),
			"strategy_used": horizon_strategy["strategy"],
		}

	def _get_optimal_horizon_strategy(self, target_horizon: int) -> Dict[str, Any]:
		"""Stratégie hybride: sélection/interpolation intelligente des horizons."""
		available_horizons = [10, 60, 360, 1440, 10080]  # 10min, 1h, 6h, 1d, 7d
		
		# Si horizon exact disponible
		if target_horizon in available_horizons:
			return {"strategy": "exact", "horizon": target_horizon}
		
		# Interpolation entre deux horizons proches
		for i in range(len(available_horizons) - 1):
			h1, h2 = available_horizons[i], available_horizons[i + 1]
			if h1 < target_horizon < h2:
				# Pondération linéaire
				weight_h1 = (h2 - target_horizon) / (h2 - h1)
				weight_h2 = (target_horizon - h1) / (h2 - h1)
				return {
					"strategy": "interpolation",
					"horizon_1": h1, "weight_1": weight_h1,
					"horizon_2": h2, "weight_2": weight_h2
				}
		
		# Extrapolation limitée (max 2x l'horizon le plus proche)
		if target_horizon < available_horizons[0]:
			closest = available_horizons[0]
			if target_horizon >= closest / 2:
				return {"strategy": "extrapolation", "base_horizon": closest, "factor": target_horizon / closest}
		
		if target_horizon > available_horizons[-1]:
			closest = available_horizons[-1]
			if target_horizon <= closest * 2:
				return {"strategy": "extrapolation", "base_horizon": closest, "factor": target_horizon / closest}
		
		# Fallback: horizon le plus proche
		closest = min(available_horizons, key=lambda h: abs(h - target_horizon))
		return {"strategy": "fallback", "horizon": closest}

	def run(
		self,
		*,
		coin_id: str,
		horizon_minutes: int,
		mid_price_usd: float,
		fx_rate_usd_per_eur: Optional[float],
		spread_pct: Optional[float] = None,
		amount_eur: Optional[float] = None,
		recent_five_min_df: Optional[pd.DataFrame] = None,
	) -> Dict[str, Any]:
		# Stratégie hybride pour l'horizon demandé
		horizon_strategy = self._get_optimal_horizon_strategy(horizon_minutes)
		
		# Prédiction selon la stratégie
		if horizon_strategy["strategy"] == "exact":
			value_pred = self._predict_with_model(coin_id, horizon_strategy["horizon"], mid_price_usd, recent_five_min_df=recent_five_min_df)
		elif horizon_strategy["strategy"] == "interpolation":
			pred1 = self._predict_with_model(coin_id, horizon_strategy["horizon_1"], mid_price_usd, recent_five_min_df=recent_five_min_df)
			pred2 = self._predict_with_model(coin_id, horizon_strategy["horizon_2"], mid_price_usd, recent_five_min_df=recent_five_min_df)
			if pred1 is not None and pred2 is not None:
				value_pred = pred1 * horizon_strategy["weight_1"] + pred2 * horizon_strategy["weight_2"]
			else:
				value_pred = pred1 or pred2  # Fallback sur prédiction disponible
		elif horizon_strategy["strategy"] == "extrapolation":
			base_pred = self._predict_with_model(coin_id, horizon_strategy["base_horizon"], mid_price_usd, recent_five_min_df=recent_five_min_df)
			if base_pred is not None:
				# Extrapolation conservative (réduction de l'amplitude)
				price_change = base_pred - mid_price_usd
				adjusted_change = price_change * horizon_strategy["factor"] * 0.8  # Facteur conservateur
				value_pred = mid_price_usd + adjusted_change
			else:
				value_pred = None
		else:  # fallback
			value_pred = self._predict_with_model(coin_id, horizon_strategy["horizon"], mid_price_usd, recent_five_min_df=recent_five_min_df)
		
		# Essayer un modèle entraîné si présent, sinon placeholder
		# value_pred = self._predict_with_model(coin_id, horizon_minutes, mid_price_usd, recent_five_min_df=recent_five_min_df)
		# Incertitude & erreur attendue via ConfidenceEstimator (calibré)
		conf = ConfidenceEstimator()

		# Spread: préférer estimation tickers si non fourni
		used_spread = spread_pct
		if used_spread is None:
			est = self._estimate_spread_pct(coin_id)
			used_spread = float(est) if (est is not None and est >= 0) else 0.002

		# Coûts: si amount_eur fourni, calcul détaillé; sinon coût unitaire pour 1 EUR
		used_amount_eur = amount_eur if amount_eur is not None else 1.0
		costs = self.cost.estimate_roundtrip_costs(
			mid_price_usd=mid_price_usd,
			spread_pct=used_spread,
			amount_eur=used_amount_eur,
			fx_rate_usd_per_eur=fx_rate_usd_per_eur,
		)
		# Convertir coûts en pourcentage approx. du montant (si amount fourni), sinon coût unitaire (1 EUR)
		base_amount = used_amount_eur if used_amount_eur and used_amount_eur > 0 else 1.0
		costs_pct = float(costs.get("total_cost_eur", 0.0)) / float(base_amount)

		ci_lo, ci_hi = conf.ci_bounds(mid_price_usd, coin_id, horizon_minutes)
		ci_p10 = ci_lo
		ci_p90 = ci_hi
		err_attendue = conf.expected_error_pct(coin_id, horizon_minutes, costs_pct)

		# Cible de profit net interpolée (best‑effort)
		target_eur = None
		try:
			from prediction.decision_engine import _targets
			target_eur = interpolate_target_eur(coin_id, horizon_minutes, _targets())
		except Exception:
			pass

		# Profit net attendu basé sur la prédiction réelle
		expected_profit_net_eur = None
		final_amount_eur = None
		if amount_eur is not None and value_pred is not None:
			expected_profit_net_eur = self._estimate_expected_profit(
				coin_id, horizon_minutes, amount_eur, value_pred, 
				mid_price_usd, costs.get("total_cost_eur", 0.0)
			)
			# NOUVEAU: Calcul du montant final total
			final_amount_eur = amount_eur + (expected_profit_net_eur or 0.0)

		decision = decide(
			coin_id=coin_id,
			horizon_minutes=horizon_minutes,
			expected_error_pct=err_attendue,
			expected_profit_net_eur=expected_profit_net_eur,
		)

		return {
			"coin_id": coin_id,
			"horizon_minutes": horizon_minutes,
			"mid_price_usd": mid_price_usd,
			"fx_rate_usd_per_eur": fx_rate_usd_per_eur,
			"spread_pct": used_spread,
			"inputs": {
				"amount_eur": amount_eur,
			},
			"estimation": {
				"value_pred": value_pred,
				"ci_p10": ci_p10,
				"ci_p90": ci_p90,
				"erreur_attendue_pct": err_attendue,
				"target_net_eur": target_eur,
				"expected_profit_net_eur": expected_profit_net_eur,
				"final_amount_eur": final_amount_eur,  # NOUVEAU: Montant final total
			},
			"costs": costs,
			"decision": decision,
		}
