import json
import time
from pathlib import Path

import pandas as pd

import src.ops.cli as cli


def test_status_reads_eval_results_and_reports_mae(tmp_path, capsys, monkeypatch):
    # Prepare a minimal eval_results parquet under a temp data_parquet root
    now_ms = int(time.time() * 1000)
    df = pd.DataFrame(
        [
            {
                "ts_utc_ms": now_ms - 10 * 60 * 1000,  # within 24h
                "prediction_id": "p1",
                "coin_id": "bitcoin",
                "horizon_minutes": 10,
                "pred_mid_price_usd": 100.0,
                "realized_mid_price_usd": 110.0,
                "abs_error_pct": 0.10,
            },
            {
                "ts_utc_ms": now_ms - 20 * 60 * 1000,
                "prediction_id": "p2",
                "coin_id": "ethereum",
                "horizon_minutes": 10,
                "pred_mid_price_usd": 200.0,
                "realized_mid_price_usd": 260.0,
                "abs_error_pct": 0.30,
            },
        ]
    )

    data_root = Path(tmp_path)
    eval_dir = data_root / "eval_results"
    eval_dir.mkdir(parents=True, exist_ok=True)
    # Write a single parquet file
    df.to_parquet(eval_dir / "part-000.parquet", engine="pyarrow")

    # Patch settings to point to our temp data root and runtime
    def fake_settings():
        return {
            "paths": {
                "data_parquet": str(data_root),
                "runtime": str(data_root / "runtime"),
            }
        }

    monkeypatch.setattr(cli, "_settings", fake_settings)

    # Run status without detail (should slim evaluation to global aggregates)
    cli.status(detail=False)
    captured = capsys.readouterr().out.strip()
    assert captured
    payload = json.loads(captured)
    assert "evaluation" in payload
    eval_stats = payload["evaluation"]
    assert isinstance(eval_stats, dict)
    # We only assert 24h window exists and is aggregated over our 2 rows
    win24 = eval_stats.get("window_24h")
    assert isinstance(win24, dict)
    # Non-detailed output has mae_pct/count directly
    assert win24.get("count") == 2
    # Average of 0.10 and 0.30 = 0.20
    assert abs(float(win24.get("mae_pct")) - 0.20) < 1e-9

    # Run status with detail=True (should include by_coin)
    cli.status(detail=True)
    captured2 = capsys.readouterr().out.strip()
    payload2 = json.loads(captured2)
    win24_detailed = payload2["evaluation"]["window_24h"]
    assert "global" in win24_detailed and "by_coin" in win24_detailed
    by_coin = win24_detailed["by_coin"]
    assert set(by_coin.keys()) == {"bitcoin", "ethereum"}
    assert by_coin["bitcoin"]["count"] == 1
    assert abs(by_coin["bitcoin"]["mae_pct"] - 0.10) < 1e-9
    assert abs(by_coin["ethereum"]["mae_pct"] - 0.30) < 1e-9
