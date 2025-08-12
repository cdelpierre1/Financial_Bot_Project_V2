import pandas as pd
from pathlib import Path

import pytest

from src.prediction.trainer import Trainer

try:
    import sklearn  # noqa: F401
except Exception:
    sklearn = None

pytestmark = pytest.mark.skipif(sklearn is None, reason="scikit-learn not available in environment")


def test_trainer_linear_regression_saves_model(tmp_path, monkeypatch):
    # Fake settings paths via ModelStore injection
    from src.prediction.model_store import ModelStore

    ms = ModelStore(
        settings={
            "paths": {
                "models_trained": str(tmp_path / "trained_models"),
                "models_backup": str(tmp_path / "backup_models"),
            }
        }
    )
    tr = Trainer(model_store=ms)

    # Build a tiny synthetic five_min dataset for one coin
    rows = []
    for i in range(20):
        rows.append({"ts_utc_ms": 1000 * i, "coin_id": "bitcoin", "c": float(100 + i)})
    df = pd.DataFrame(rows)

    res = tr.train(df, "bitcoin", horizon_minutes=10)
    assert res["status"] == "OK"
    assert res["rows"] > 0
    assert Path(res["model_path"]).exists()

    # Load back
    model, meta = tr.load("bitcoin", 10)
    assert hasattr(model, "predict")
    assert meta and meta.get("algo") == "LinearRegression"
