import pandas as pd
from src.collectors.range_collector import RangeCollector


def test_range_collector_to_dataframe_mapping():
    payload = {
        "ts_utc_ms": 0,
        "id": "bitcoin",
        "from_ms": 0,
        "to_ms": 1000,
        "data": {
            "prices": [[100, 10.0], [200, 12.0]],
            "total_volumes": [[100, 1.5], [200, 1.8]],
        },
    }
    df = RangeCollector.to_dataframe("bitcoin", payload)
    assert list(df.columns) == ["ts_utc_ms","coin_id","o","h","l","c","volume","agg_method"]
    assert len(df) == 2
    assert df.iloc[0]["ts_utc_ms"] == 100
    assert df.iloc[0]["c"] == 10.0
    assert df.iloc[0]["volume"] == 1.5
    assert df.iloc[0]["agg_method"] == "coingecko_range"
