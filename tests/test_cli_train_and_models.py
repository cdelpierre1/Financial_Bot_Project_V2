import json
from typer.testing import CliRunner

from src.ops.cli import app as cli_app


def test_cli_models_command_monkeypatched(monkeypatch, tmp_path):
    class FakeStore:
        def list_models(self):
            return [str(tmp_path / "bitcoin__10m.pkl"), str(tmp_path / "ethereum__10m.pkl")]

    # Patch ModelStore used in CLI
    from src import ops as ops_pkg  # noqa: F401
    import src.ops.cli as cli_mod
    monkeypatch.setattr(cli_mod, "ModelStore", FakeStore)

    runner = CliRunner()
    res = runner.invoke(cli_app, ["models"])  # prints JSON
    assert res.exit_code == 0
    data = json.loads(res.stdout)
    assert data["count"] == 2
    assert "by_coin" in data
    assert "bitcoin" in data["by_coin"] and "ethereum" in data["by_coin"]


def test_cli_train_command_monkeypatched(monkeypatch, tmp_path):
    calls = []

    class FakeTrainer:
        def train(self, df, coin_id, horizon_minutes: int):
            calls.append((coin_id, horizon_minutes, len(df)))
            return {"status": "OK", "rows": 42, "model_path": str(tmp_path / f"{coin_id}__{horizon_minutes}m.pkl")}

    # Patch Trainer and data reader
    import src.ops.cli as cli_mod
    monkeypatch.setattr(cli_mod, "Trainer", FakeTrainer)

    # Provide minimal fake five_min DataFrame
    import pandas as pd
    df = pd.DataFrame([
        {"ts_utc_ms": i * 300000, "coin_id": "bitcoin", "c": 100.0 + i} for i in range(30)
    ])

    def fake_read():
        return df

    monkeypatch.setattr(cli_mod, "_read_recent_five_min", fake_read)

    runner = CliRunner()
    res = runner.invoke(cli_app, ["train", "--coin", "bitcoin", "--horizon", "10"])
    assert res.exit_code == 0
    out = json.loads(res.stdout)
    assert out["horizon_minutes"] == 10
    assert out["results"][0]["coin_id"] == "bitcoin"
    assert out["results"][0]["status"] == "OK"
    assert calls and calls[0][0] == "bitcoin"
