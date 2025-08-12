import pandas as pd

from src.prediction.feature_builder import FeatureBuilder


def test_feature_builder_basic_series():
    # Create minimal increasing close series for one coin
    rows = []
    for i in range(6):
        rows.append({"ts_utc_ms": i * 300000, "coin_id": "bitcoin", "c": float(100 + i)})
    df = pd.DataFrame(rows)

    fb = FeatureBuilder(step_minutes=5)
    X, y = fb.build_from_five_min(df, "bitcoin", horizon_minutes=10)

    # With 6 rows, lag1/dropna removes first, and 10min horizon (2 steps) removes last 2 from target
    # Effective rows should be 6 - 1 (lag) - 2 (future horizon) = 3
    assert len(X) == len(y) == 3
    assert list(X.columns) == ["c", "lag1", "diff1"]
    # Basic sanity: no NaNs and finite values
    assert X.isna().sum().sum() == 0
    assert y.isna().sum() == 0
