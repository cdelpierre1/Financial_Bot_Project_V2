import json
import pandas as pd
import src.ops.cli as cli


def test_cli_predict_non_interactive_outputs_json_and_saves(monkeypatch, capsys):
    # Patch PriceCollector to avoid network
    monkeypatch.setattr(
        cli.PriceCollector,
        "get_prices",
        lambda self, ids=None: {"prices": {"bitcoin": 100.0}},
    )

    # Patch FxCollector to provide a stable FX dataframe
    monkeypatch.setattr(cli.FxCollector, "get_latest", lambda self: {"ok": True})
    monkeypatch.setattr(
        cli.FxCollector,
        "to_dataframe",
        lambda payload: pd.DataFrame([{"rate_usd_per_eur": 1.1}]),
    )

    # Patch PredictionPipeline.run to return a deterministic result
    def fake_run(self, **kwargs):
        return {
            "coin_id": kwargs.get("coin_id"),
            "horizon_minutes": kwargs.get("horizon_minutes"),
            "mid_price_usd": kwargs.get("mid_price_usd"),
            "fx_rate_usd_per_eur": kwargs.get("fx_rate_usd_per_eur"),
            "spread_pct": 0.002,
            "inputs": {"amount_eur": kwargs.get("amount_eur")},
            "estimation": {
                "value_pred": None,
                "ci_p10": 95.0,
                "ci_p90": 105.0,
                "erreur_attendue_pct": 0.05,
                "target_net_eur": 10.0,
                "expected_profit_net_eur": None,
            },
            "costs": {
                "trading_fee_eur_total": 2.0,
                "spread_cost_eur": 0.2,
                "slippage_cost_eur": 0.05,
                "total_cost_eur": 2.25,
                "effective_buy_price_usd": 100.1,
                "effective_sell_price_usd": 99.9,
            },
            "decision": {
                "status": "OK",
                "seuil_pct": 0.05,
                "cible_profit_eur": 10.0,
                "raisons": [],
            },
        }

    monkeypatch.setattr(cli.PredictionPipeline, "run", fake_run, raising=True)

    # Capture save_prediction invocation
    captured = {}

    def fake_save_prediction(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli, "save_prediction", fake_save_prediction)
    monkeypatch.setattr(cli, "generate_prediction_id", lambda: "test-pred-id")

    # Invoke predict in non-interactive mode
    cli.predict(
        kind="price",
        coin_opt="bitcoin",
        unit_opt="min",
        value_opt=10,
        amount_eur_opt=None,
    )

    # Read printed JSON
    out = capsys.readouterr().out.strip()
    assert out
    payload = json.loads(out)

    # Basic shape assertions
    assert payload["coin"]["id"] == "bitcoin"
    assert payload["requested"]["unit"] == "min"
    assert payload["requested"]["minutes"] == 10
    assert payload["output"]["decision"]["status"] == "OK"

    # Check that save_prediction was called with our pred id and fields
    assert captured.get("prediction_id") == "test-pred-id"
    assert captured.get("coin_id") == "bitcoin"
    assert captured.get("horizon_minutes") == 10
    assert isinstance(captured.get("ts_pred_utc_ms"), int)
    assert isinstance(captured.get("target_ts_utc_ms"), int)
    assert captured.get("mid_price_usd") == 100.0
    extra = captured.get("extra", {})
    assert extra.get("erreur_attendue_pct") == 0.05
    assert extra.get("spread_pct") == 0.002
    assert extra.get("fx_rate_usd_per_eur") == 1.1
