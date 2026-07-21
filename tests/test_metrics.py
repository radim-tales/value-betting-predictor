import numpy as np
from vbp.metrics import clv, roi, bootstrap_roi_ci, brier, apply_slippage

def test_clv_positive_when_open_beats_close():
    # bet at 2.10, closing fair prob 0.52 -> CLV = 0.52*2.10 - 1 = 0.092
    assert abs(clv(o_open=2.10, p_close_fair=0.52) - 0.092) < 1e-9

def test_roi_flat_stake():
    bets = [{"won": True, "odds": 2.0}, {"won": False, "odds": 3.0}]
    # +1.0 and -1.0 -> ROI 0
    assert abs(roi(bets) - 0.0) < 1e-9

def test_roi_all_winners():
    bets = [{"won": True, "odds": 2.0}, {"won": True, "odds": 1.5}]
    # profit 1.0 + 0.5 = 1.5 over 2 stake -> 0.75
    assert abs(roi(bets) - 0.75) < 1e-9

def test_bootstrap_ci_brackets_point_estimate():
    bets = [{"won": True, "odds": 2.0}] * 60 + [{"won": False, "odds": 2.0}] * 40
    lo, hi = bootstrap_roi_ci(bets, n_boot=500, alpha=0.10, seed=42)
    assert lo < roi(bets) < hi

def test_brier_perfect_prediction_is_zero():
    preds = [{"H": 1.0, "D": 0.0, "A": 0.0}]
    outcomes = ["H"]
    assert brier(preds, outcomes) < 1e-12

def test_slippage_reduces_odds():
    assert abs(apply_slippage(2.00, 0.01) - 1.98) < 1e-9

def test_bootstrap_mean_ci_brackets_mean():
    import numpy as np
    from vbp.metrics import bootstrap_mean_ci
    vals = [0.02]*60 + [-0.05]*40      # mean = -0.008
    lo, hi = bootstrap_mean_ci(vals, n_boot=500, alpha=0.10, seed=42)
    assert lo < float(np.mean(vals)) < hi

def test_bootstrap_mean_ci_empty_is_zero():
    from vbp.metrics import bootstrap_mean_ci
    assert bootstrap_mean_ci([], n_boot=100) == (0.0, 0.0)
