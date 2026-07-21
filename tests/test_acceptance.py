from vbp.acceptance import evaluate_acceptance


def test_all_pass():
    res = evaluate_acceptance(
        learned={"mean_clv": 0.01, "roi_slip1": 0.02, "n_bets": 150,
                 "roi_ci_lo": -0.05, "brier": 0.60, "roi_drop_top3": -0.02},
        empty={"mean_clv": -0.03}, noise={"mean_clv": -0.04},
        anchor={"brier": 0.62}, market={"brier": 0.61},
        frozen={"mean_clv": 0.005}, static={"mean_clv": 0.0}, no_reflection={"mean_clv": 0.004},
    )
    assert res["passed"] is True
    assert res["criteria"]["clv_positive"] is True


def test_fails_when_clv_not_beating_noise():
    res = evaluate_acceptance(
        learned={"mean_clv": -0.05, "roi_slip1": 0.0, "n_bets": 150,
                 "roi_ci_lo": -0.05, "brier": 0.62, "roi_drop_top3": -0.02},
        empty={"mean_clv": -0.03}, noise={"mean_clv": -0.04},
        anchor={"brier": 0.62}, market={"brier": 0.61},
        frozen={"mean_clv": -0.06}, static={"mean_clv": -0.06}, no_reflection={"mean_clv": -0.06},
    )
    assert res["passed"] is False
