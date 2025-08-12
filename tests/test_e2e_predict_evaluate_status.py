import json
import time
from pathlib import Path

import pandas as pd

import src.prediction.evaluation as evaluation
import src.ops.cli as cli
from src.storage.parquet_writer import ParquetWriter as RealParquetWriter


def test_e2e_predict_evaluate_status(tmp_path, monkeypatch, capsys):
    # Prepare temp settings with parquet root
    data_root = Path(tmp_path) / "parquet"
    runtime_root = Path(tmp_path) / "runtime"
    data_root.mkdir(parents=True, exist_ok=True)
    runtime_root.mkdir(parents=True, exist_ok=True)

    temp_settings = {
        "paths": {
            "data_parquet": str(data_root),
            "runtime": str(runtime_root),
        },
        "parquet": {"compression": "snappy", "retention_days": {}},
    }

    # Patch evaluation settings and its ParquetWriter to use temp paths
    monkeypatch.setattr(evaluation, "_settings", lambda: temp_settings)
    monkeypatch.setattr(evaluation, "ParquetWriter", lambda: RealParquetWriter(settings=temp_settings))

    # Create a prediction in the past so it's due
    now_ms = int(time.time() * 1000)
    pred_id = evaluation.generate_prediction_id()
    coin_id = "bitcoin"

    evaluation.save_prediction(
        prediction_id=pred_id,
        coin_id=coin_id,
        horizon_minutes=10,
        ts_pred_utc_ms=now_ms - 120_000,
        target_ts_utc_ms=now_ms - 60_000,
        mid_price_usd=100.0,
        value_pred=None,
        extra={"erreur_attendue_pct": 0.05, "spread_pct": 0.002, "fx_rate_usd_per_eur": 1.1},
    )

    # Evaluate: mock realized price to avoid needing five_min data
    job = evaluation.EvaluationJob()
    monkeypatch.setattr(job, "_realized_mid_usd", lambda cid, ts: 110.0)
    n = job.evaluate_pending()
    assert n == 1

    # Now run status on the temp parquet root and verify evaluation includes 1 item
    monkeypatch.setattr(cli, "_settings", lambda: temp_settings)
    cli.status(detail=False)
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)

    # Verify evaluation 24h window shows count 1 and mae ~0.10
    eval_stats = payload.get("evaluation")
    assert isinstance(eval_stats, dict)
    w24 = eval_stats.get("window_24h")
    assert w24 and w24.get("count") == 1
    assert abs(float(w24.get("mae_pct")) - 0.10) < 1e-9
