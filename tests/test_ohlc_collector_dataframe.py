import pandas as pd
from src.collectors.ohlc_collector import OhlcCollector


def test_ohlc_collector_to_dataframe_mapping():
    payload = {
        "ts_utc_ms": 0,
        "id": "bitcoin",
        "days": 1,
        "data": [
            [100, 9.0, 12.0, 8.5, 11.0],
            [200, 11.0, 12.5, 10.0, 12.0],
        ],
    }
    df = OhlcCollector.to_dataframe("bitcoin", payload)
    assert list(df.columns) == ["ts_utc_ms","coin_id","o","h","l","c","volume","agg_method"]
    assert len(df) == 2
    assert df.iloc[0]["ts_utc_ms"] == 100
    assert df.iloc[0]["o"] == 9.0
    assert df.iloc[0]["h"] == 12.0
    assert df.iloc[0]["l"] == 8.5
    assert df.iloc[0]["c"] == 11.0
    assert df.iloc[0]["agg_method"] == "coingecko_ohlc"
