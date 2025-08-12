import json
from pathlib import Path

import src.ops.cli as cli


def test_status_collectors_block_from_cache_json(tmp_path, monkeypatch, capsys):
    # Arrange temp folders
    data_root = Path(tmp_path) / "parquet"
    runtime_root = Path(tmp_path) / "runtime"
    data_root.mkdir(parents=True, exist_ok=True)
    runtime_root.mkdir(parents=True, exist_ok=True)

    # Fake settings with our temp paths
    temp_settings = {
        "paths": {
            "data_parquet": str(data_root),
            "runtime": str(runtime_root),
        }
    }

    # Prepare a fake cache_store.json with one collector entry
    cache_payload = {
        "version": 2,
        "saved_at": 0,
        "entries": {
            "k1": {
                "value": {"ok": True},
                "expires_at": 9999999999.0,
                "created_at": 0.0,
                "meta": {
                    "key_parts": {"endpoint": "simple_price", "ids": ["bitcoin"], "vs": "usd"},
                    "last_latency_ms": 87,
                    "last_status": 200,
                },
            }
        },
        "stats": {
            "k1": {
                "hits": 3,
                "misses": 1,
                "sets": 1,
                "expirations": 0,
                "failures": 0,
                "last_error": None,
                "last_status": 200,
                "last_latency_ms": 87,
                "consecutive_failures": 0,
                "circuit_open_until": 0.0,
            }
        },
    }

    # Path resolution in status uses parent(runtime)/runtime/cache_store.json
    cache_path = Path(runtime_root).parent / "runtime" / "cache_store.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache_payload), encoding="utf-8")

    # Patch settings and run status
    monkeypatch.setattr(cli, "_settings", lambda: temp_settings)
    cli.status(detail=False)

    # Parse output
    out = capsys.readouterr().out.strip()
    assert out
    payload = json.loads(out)

    # Validate collectors block
    collectors = payload.get("collectors")
    assert isinstance(collectors, dict)
    sp = collectors.get("simple_price")
    assert isinstance(sp, dict)
    assert sp.get("last_status") == 200
    assert isinstance(sp.get("last_latency_ms"), int)
    # ttl_remaining_s should be >= 0 or None (depending on now)
    ttl = sp.get("ttl_remaining_s")
    assert ttl is None or isinstance(ttl, int)
    # stats passthrough
    assert sp.get("hits") == 3
    assert sp.get("misses") == 1
    assert sp.get("failures") == 0
