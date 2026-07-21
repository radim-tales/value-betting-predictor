from vbp.block_report import aggregate_block


def test_aggregate_block_computes_core_metrics():
    preds = [{"H": 0.5, "D": 0.3, "A": 0.2}, {"H": 0.4, "D": 0.3, "A": 0.3}]
    outcomes = ["H", "A"]
    bets = [{"outcome": "H", "odds": 2.0, "won": True, "clv": 0.02, "model_p": 0.5}]
    rep = aggregate_block(preds, outcomes, bets, n_skipped=1)
    assert rep["n_bets"] == 1
    assert abs(rep["roi"] - 1.0) < 1e-9        # single winning bet at 2.0
    assert "brier" in rep and "clv" in rep
    assert rep["n_skipped"] == 1


def test_empty_block_is_safe():
    rep = aggregate_block([], [], [], n_skipped=0)
    assert rep["n_bets"] == 0
