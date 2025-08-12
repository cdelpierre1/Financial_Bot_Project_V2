"""
Moteur de décision minimal pour le bot:
- Interpole seuils d'erreur et cibles de profit net en EUR pour un horizon arbitraire
- Applique cap global et "profit net gate"
- Produit une structure de décision simple {status, raisons[]}
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import json
import os

from prediction.threshold_policy import interpolate_error_threshold


ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(ROOT, "config")


def _load_json(path: str) -> Dict[str, Any]:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def _thresholds() -> Dict[str, Any]:
	return _load_json(os.path.join(CONFIG_DIR, "thresholds.json"))


def _targets() -> Dict[str, Any]:
	return _load_json(os.path.join(CONFIG_DIR, "targets.json"))


def interpolate_target_eur(coin_id: str, minutes: int, targets_cfg: Dict[str, Any]) -> Optional[float]:
	items = targets_cfg.get("targets", [])
	entry = next((x for x in items if x.get("id") == coin_id), None)
	if not entry:
		return None
	t6 = float(entry.get("profit_net_eur", {}).get("6h", 0.0))
	t24 = float(entry.get("profit_net_eur", {}).get("24h", 0.0))
	m = max(1, int(minutes))
	if m == 360:
		return t6
	if m == 1440:
		return t24
	slope = (t24 - t6) / (1440 - 360)
	return t6 + slope * (m - 360)


def decide(
	*,
	coin_id: str,
	horizon_minutes: int,
	expected_error_pct: Optional[float],
	expected_profit_net_eur: Optional[float],
) -> Dict[str, Any]:
	"""Décision simple: compare erreur attendue au seuil interpolé et profit net à la cible interpolée.

	- expected_error_pct: proportion (0.05 pour 5%). Si None, on utilise seulement le profit gate.
	- expected_profit_net_eur: profit net attendu en EUR. Si None, gate non appliqué.
	"""
	th = _thresholds()
	tg = _targets()

	seuil = interpolate_error_threshold(horizon_minutes, th)
	cible = interpolate_target_eur(coin_id, horizon_minutes, tg)

	raisons = []

	if expected_error_pct is not None and expected_error_pct > seuil:
		raisons.append("Erreur attendue au‑dessus du seuil")

	if cible is not None and expected_profit_net_eur is not None and expected_profit_net_eur < cible:
		raisons.append("Profit net attendu inférieur à la cible")

	status = "OK" if not raisons else "NO_CALL"
	return {
		"status": status,
		"seuil_pct": seuil,
		"cible_profit_eur": cible,
		"raisons": raisons,
	}
