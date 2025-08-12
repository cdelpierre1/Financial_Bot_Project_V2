"""
Modèle de coûts de marché (frais, spread, micro‑slippage) avec sorties en EUR.

Contrat minimal:
- Entrées: mid_price_usd (float), spread_pct (proportion, ex 0.002), amount_eur (float), fx_rate_usd_per_eur (float)
- Paramètres: trading_fee_eur_per_order (fixe), micro_slippage_pct (proportion)
- Sorties: dictionnaire détaillant coûts par composantes et prix effectifs achat/vente

Notes:
- Le roundtrip considère un achat à l'ask et une vente au bid, avec micro‑slippage
- Les frais par ordre sont appliqués deux fois (achat + vente)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import json
import os


ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(ROOT, "config")


def _load_json(path: str) -> Dict[str, Any]:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def _costs_cfg() -> Dict[str, Any]:
	return _load_json(os.path.join(CONFIG_DIR, "costs.json"))


@dataclass
class CostParams:
	trading_fee_eur_per_order: float = 1.0
	micro_slippage_pct: float = 0.0005  # 0.05%

	@staticmethod
	def from_config(cfg: Optional[Dict[str, Any]] = None) -> "CostParams":
		c = cfg or _costs_cfg()
		return CostParams(
			trading_fee_eur_per_order=float(c.get("trading_fee_eur_per_order", 1.0)),
			micro_slippage_pct=float(c.get("micro_slippage_pct", 0.0005)),
		)


class CostModel:
	def __init__(self, params: Optional[CostParams] = None) -> None:
		self.params = params or CostParams.from_config()

	@staticmethod
	def compute_effective_prices(
		mid_price_usd: float,
		spread_pct: float,
		micro_slippage_pct: float,
	) -> Dict[str, float]:
		"""Calcule les prix effectifs achat (ask) et vente (bid) incluant le slippage.

		- spread_pct est la largeur relative (ex 0.002 = 0.2%); bid=mid*(1-s/2), ask=mid*(1+s/2)
		- micro_slippage_pct est appliqué sur chaque jambe: ask *= (1+slip), bid *= (1-slip)
		"""
		s = max(0.0, float(spread_pct))
		slip = max(0.0, float(micro_slippage_pct))
		ask = mid_price_usd * (1.0 + s / 2.0)
		bid = mid_price_usd * (1.0 - s / 2.0)
		ask *= (1.0 + slip)
		bid *= (1.0 - slip)
		return {"ask_usd": ask, "bid_usd": bid}

	def estimate_roundtrip_costs(
		self,
		mid_price_usd: float,
		spread_pct: float,
		amount_eur: float,
		fx_rate_usd_per_eur: Optional[float],
	) -> Dict[str, float]:
		"""Estime le coût total d'un aller‑retour (achat puis vente) pour un montant en EUR.

		Paramètres:
		- mid_price_usd: prix médian USD de l'actif
		- spread_pct: proportion (ex: 0.002 pour 0.2%)
		- amount_eur: montant en EUR à investir
		- fx_rate_usd_per_eur: USD par EUR (ex: 1.10). Si None, fallback 1.10.

		Retourne un dict avec:
		- trading_fee_eur_total
		- spread_cost_eur
		- slippage_cost_eur
		- total_cost_eur
		- effective_buy_price_usd
		- effective_sell_price_usd
		"""
		if amount_eur <= 0:
			return {
				"trading_fee_eur_total": 0.0,
				"spread_cost_eur": 0.0,
				"slippage_cost_eur": 0.0,
				"total_cost_eur": 0.0,
				"effective_buy_price_usd": float(mid_price_usd),
				"effective_sell_price_usd": float(mid_price_usd),
			}

		r = float(fx_rate_usd_per_eur) if fx_rate_usd_per_eur else 1.10
		prices = self.compute_effective_prices(mid_price_usd, spread_pct, self.params.micro_slippage_pct)
		ask = prices["ask_usd"]
		bid = prices["bid_usd"]

		# Conversion EUR → USD
		amount_usd = amount_eur * r

		# Achat à l'ask → quantité achetée
		qty = amount_usd / ask if ask > 0 else 0.0

		# Revente au bid → produit en USD
		proceeds_usd = qty * bid
		roundtrip_spread_slip_cost_usd = max(0.0, amount_usd - proceeds_usd)
		roundtrip_spread_slip_cost_eur = roundtrip_spread_slip_cost_usd / r

		# Frais fixes par ordre (EUR), fois 2 (achat+vente)
		trading_fee_eur_total = 2.0 * self.params.trading_fee_eur_per_order

		total_cost_eur = trading_fee_eur_total + roundtrip_spread_slip_cost_eur

		# Pour visibilité, estimer la part slippage ~ proportionnelle au micro_slippage_pct
		# Approche heuristique: slippage_share ≈ min(1.0, 2*micro_slippage_pct / max(spread_pct,1e-9))
		s = max(1e-12, float(spread_pct))
		slip = max(0.0, float(self.params.micro_slippage_pct))
		slippage_share = max(0.0, min(1.0, 2.0 * slip / s)) if s > 0 else 1.0
		slippage_cost_eur = roundtrip_spread_slip_cost_eur * slippage_share
		spread_cost_eur = roundtrip_spread_slip_cost_eur - slippage_cost_eur

		return {
			"trading_fee_eur_total": float(trading_fee_eur_total),
			"spread_cost_eur": float(spread_cost_eur),
			"slippage_cost_eur": float(slippage_cost_eur),
			"total_cost_eur": float(total_cost_eur),
			"effective_buy_price_usd": float(ask),
			"effective_sell_price_usd": float(bid),
		}
