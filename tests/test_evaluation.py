import time
import types
import pandas as pd
import pytest

import src.prediction.evaluation as evaluation


def test_save_prediction_uses_writer(monkeypatch):
    captured = {}

    class FakeWriter:
        def write(self, dataset, df, dedup_keys=None, partition_cols=None):
            captured["dataset"] = dataset
            captured["df"] = df.copy()
            return len(df)

    # Patch ParquetWriter in module
    monkeypatch.setattr(evaluation, "ParquetWriter", lambda: FakeWriter())

    pred_id = evaluation.generate_prediction_id()
    ts_pred = int(time.time() * 1000)
    target_ts = ts_pred + 60_000

    evaluation.save_prediction(
        prediction_id=pred_id,
        coin_id="bitcoin",
        horizon_minutes=1,
        ts_pred_utc_ms=ts_pred,
        target_ts_utc_ms=target_ts,
        mid_price_usd=100.0,
        value_pred=None,
        extra={"erreur_attendue_pct": 0.05, "spread_pct": 0.002, "fx_rate_usd_per_eur": 1.10},
    )

    assert captured.get("dataset") == "predictions"
    df = captured.get("df")
    assert isinstance(df, pd.DataFrame)
    assert df.loc[0, "prediction_id"] == pred_id
    assert df.loc[0, "coin_id"] == "bitcoin"
    assert df.loc[0, "horizon_minutes"] == 1
    assert df.loc[0, "target_ts_utc_ms"] == target_ts
    assert df.loc[0, "mid_price_usd"] == 100.0
    assert pytest.approx(df.loc[0, "erreur_attendue_pct"], rel=1e-9) == 0.05


def test_evaluate_pending_computes_abs_error(monkeypatch):
    # Build a fake predictions df with one due record
    now_ms = int(time.time() * 1000)
    pred_mid = 100.0
    due_row = {
        "ts_utc_ms": now_ms - 120_000,
        "prediction_id": "pred-1",
        "coin_id": "bitcoin",
        "horizon_minutes": 1,
        "target_ts_utc_ms": now_ms - 60_000,
        "mid_price_usd": pred_mid,
        "value_pred": None,
    }
    df_predictions = pd.DataFrame([due_row])

    # No previous eval_results
    df_eval_results = None

    # Monkeypatch EvaluationJob internals
    job = evaluation.EvaluationJob()

    def fake_read(dataset: str):
        if dataset == "predictions":
            return df_predictions
        if dataset == "eval_results":
            return df_eval_results
        return None

    monkeypatch.setattr(job, "_read_parquet_dir", fake_read)

    # Realized price 110 → abs_error_pct = |110-100|/100 = 0.10
    monkeypatch.setattr(job, "_realized_mid_usd", lambda coin_id, ts: 110.0)

    captured = {}

    class FakeWriter:
        def write(self, dataset, df, dedup_keys=None, partition_cols=None):
            captured["dataset"] = dataset
            captured["df"] = df.copy()
            return len(df)

    job.writer = FakeWriter()

    n = job.evaluate_pending()
    assert n == 1

    assert captured.get("dataset") == "eval_results"
    out = captured.get("df")
    assert isinstance(out, pd.DataFrame)
    assert out.loc[0, "prediction_id"] == "pred-1"
    assert out.loc[0, "coin_id"] == "bitcoin"
    assert pytest.approx(out.loc[0, "abs_error_pct"], rel=1e-9) == 0.10
