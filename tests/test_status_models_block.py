import json
from types import SimpleNamespace

from src.ops import cli as cli_mod


def test_status_models_block_monkeypatched(capsys, monkeypatch, tmp_path):
    # Prepare fake model files in tmp
    pkl1 = tmp_path / "bitcoin__10m.pkl"
    pkl1.write_bytes(b"fake")
    meta1 = {"saved_at_utc": "2025-08-10T00:00:00+00:00"}

    # Fake store with list_models and load
    class FakeStore:
        def list_models(self):
            return [str(pkl1)]
        def load(self, coin_id, horizon_minutes):
            return object(), meta1

    # Patch ModelStore used inside CLI
    monkeypatch.setattr(cli_mod, "ModelStore", FakeStore)

    # Invoke status and capture output
    cli_mod.status()
    out = capsys.readouterr().out
    data = json.loads(out)

    assert "models" in data
    assert data["models"]["count"] == 1
    by_coin = data["models"].get("by_coin", {})
    assert "bitcoin" in by_coin
    assert "10m" in by_coin["bitcoin"]
    assert by_coin["bitcoin"]["10m"]["saved_at_utc"] == meta1["saved_at_utc"]
