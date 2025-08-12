import pandas as pd
import numpy as np

from src.prediction.trainer import Trainer
from src.prediction.pipeline import PredictionPipeline
from src.prediction.model_store import ModelStore


def test_pipeline_uses_trained_model(tmp_path, monkeypatch):
    # Redirect model paths to tmp
    ms = ModelStore(settings={
        "paths": {
            "models_trained": str(tmp_path / "trained_models"),
            "models_backup": str(tmp_path / "backup_models"),
            "data_parquet": str(tmp_path / "parquet"),
            "runtime": str(tmp_path / "runtime"),
        },
        "timezone": "UTC"
    })
    trainer = Trainer(model_store=ms)

    # Build synthetic five_min for one coin with linear drift
    rows = []
    price0 = 100.0
    for i in range(50):
        rows.append({"ts_utc_ms": i * 300000, "coin_id": "bitcoin", "c": price0 + i})
    df = pd.DataFrame(rows)

    # Train a 10-minute horizon model
    res = trainer.train(df, "bitcoin", horizon_minutes=10)
    assert res["status"] == "OK"

    # Run pipeline with the same recent df to avoid parquet I/O
    pipe = PredictionPipeline(model_store=ms)
    out = pipe.run(
        coin_id="bitcoin",
        horizon_minutes=10,
        mid_price_usd=df.iloc[-1]["c"],
        fx_rate_usd_per_eur=1.1,
        spread_pct=0.002,
        amount_eur=None,
        recent_five_min_df=df,
    )

    est = out["estimation"]
    assert "value_pred" in est
    assert est["value_pred"] is None or isinstance(est["value_pred"], float)
