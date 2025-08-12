from src.prediction.cost_model import CostModel


def test_cost_model_basic_roundtrip_costs():
	cm = CostModel()
	costs = cm.estimate_roundtrip_costs(
		mid_price_usd=100.0,
		spread_pct=0.002,  # 0.2%
		amount_eur=100.0,
		fx_rate_usd_per_eur=1.1,
	)
	assert costs["total_cost_eur"] > 0.0
	assert costs["effective_buy_price_usd"] > costs["effective_sell_price_usd"]


def test_cost_model_zero_amount():
	cm = CostModel()
	costs = cm.estimate_roundtrip_costs(
		mid_price_usd=100.0,
		spread_pct=0.002,
		amount_eur=0.0,
		fx_rate_usd_per_eur=1.1,
	)
	assert costs["total_cost_eur"] == 0.0
