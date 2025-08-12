"""Placeholder prediction.threshold_policy — SPEC UNIQUEMENT.

Politique de seuils d'erreur attendue en fonction de l'horizon.

- Interpolation linéaire entre points d'ancrage: +10min, +6h, +24h
- Plancher à 10 min, extrapolation au‑delà de 24h mais plafonnée par cap_global
- Renvoie des pourcentages sous forme décimale (0.02 = 2%)
"""

from __future__ import annotations

from typing import Any, Dict


def interpolate_error_threshold(minutes: int, thresholds_cfg: Dict[str, Any]) -> float:
    """Calcule le seuil d'erreur autorisé pour un horizon arbitraire (minutes).

    Paramètres:
    - minutes: horizon en minutes (>0)
    - thresholds_cfg: dict issu de src/config/thresholds.json

    Retour:
    - seuil en proportion (ex: 0.02 = 2%)
    """
    m = max(1, int(minutes))
    h = thresholds_cfg.get("horizons", {})
    a10 = float(h.get("+10min", 0.02))
    a6h = float(h.get("+6h", 0.05))
    a24h = float(h.get("+24h", 0.08))
    cap = float(thresholds_cfg.get("cap_global", 0.10))

    if m <= 10:
        return a10
    if m >= 1440:
        # extrap linéaire au-delà de 24h avec pente 6h→24h, plafonnée à cap
        slope = (a24h - a6h) / (1440 - 360)
        val = a24h + slope * (m - 1440)
        return min(val, cap)
    if m <= 360:
        # interp 10min → 6h
        slope = (a6h - a10) / (360 - 10)
        return a10 + slope * (m - 10)
    # interp 6h → 24h
    slope = (a24h - a6h) / (1440 - 360)
    return a6h + slope * (m - 360)


def get_cap_global(thresholds_cfg: Dict[str, Any]) -> float:
    return float(thresholds_cfg.get("cap_global", 0.10))


def get_alert_level(thresholds_cfg: Dict[str, Any]) -> float:
    return float(thresholds_cfg.get("alert_level", 0.80))
