import json
import os

from src.prediction.threshold_policy import interpolate_error_threshold


def _load_thresholds() -> dict:
	root = os.path.join(os.path.dirname(__file__), "..", "src", "config")
	with open(os.path.abspath(os.path.join(root, "thresholds.json")), "r", encoding="utf-8") as f:
		return json.load(f)


def test_threshold_policy_anchors_and_interpolation():
	th = _load_thresholds()
	# Ancrages
	assert interpolate_error_threshold(10, th) == th["horizons"]["+10min"]
	assert interpolate_error_threshold(360, th) == th["horizons"]["+6h"]
	assert interpolate_error_threshold(1440, th) == th["horizons"]["+24h"]

	# Plancher: <10 min → seuil de +10 min
	assert interpolate_error_threshold(5, th) == th["horizons"]["+10min"]

	# Interpolation ~60 min entre 2% et 5%
	v60 = interpolate_error_threshold(60, th)
	assert th["horizons"]["+10min"] < v60 < th["horizons"]["+6h"]

	# Extrapolation plafonnée par cap à 48h
	v48h = interpolate_error_threshold(2880, th)
	assert 0.0 < v48h <= float(th.get("cap_global", 0.10))
