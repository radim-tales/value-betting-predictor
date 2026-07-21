from vbp.report import render_learn_report

def test_learn_report_has_verdict_and_variants():
    summary = {
        "learned": {"roi": 0.02, "mean_clv": -0.01, "brier": 0.62, "n_bets": 130},
        "variants": {"empty": {"roi": -0.03, "mean_clv": -0.04},
                     "noise": {"roi": 0.06, "mean_clv": -0.05},
                     "anchor_only": {"roi": 0.016, "mean_clv": -0.06}},
        "acceptance": {"passed": False, "criteria": {"clv_positive": False, "clv_beats_noise": True}},
    }
    md = render_learn_report(summary)
    assert "CLV" in md and "FAIL" in md and "noise" in md.lower()
    assert "clv_positive" in md
