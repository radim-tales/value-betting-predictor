from vbp.report import render_report

def test_report_contains_key_metrics():
    summary = {
        "n_bets": 120, "roi": 0.021, "roi_ci": (-0.03, 0.07),
        "mean_clv": 0.012, "brier": 0.62, "brier_market": 0.61,
        "roi_by_outcome": {"H": 0.03, "D": -0.01, "A": 0.02},
        "baselines": {"noise_roi": -0.04, "always_favorite_roi": -0.05},
    }
    md = render_report(summary)
    assert "ROI" in md and "CLV" in md and "120" in md
    assert "noise" in md.lower()
