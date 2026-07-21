from pathlib import Path
from vbp.data import load_matches
from vbp.backtest import run_backtest

FIX = Path(__file__).parent / "fixtures" / "mini_league.csv"

def test_backtest_runs_and_produces_audit_rows():
    df = load_matches(FIX, odds_source="pinnacle")
    # tiny fixture: train on all-but-last, test last; skip_first_rounds=0 for the fixture
    result = run_backtest(
        train_df=df.iloc[:2], test_df=df.iloc[2:],
        odds_source="pinnacle", devig_method="shin",
        anchor_cfg=dict(k=20, home_adv=70, start_rating=1500),
        value_cfg=dict(min_edge=0.0, odds_min=1.0, odds_max=99.0),
        skip_first_rounds=0,
    )
    assert len(result["audit"]) == len(df.iloc[2:])          # one audit row per test match
    for row in result["audit"]:
        p = row["anchor_p"]
        assert abs(p["H"] + p["D"] + p["A"] - 1.0) < 1e-6    # valid probability
        assert set(row["open_odds"].keys()) == {"H", "D", "A"}   # raw odds stored for baselines
    # settled bets carry the fields metrics need
    for b in result["bets"]:
        assert {"outcome", "odds", "won", "clv"} <= set(b.keys())

def test_backtest_is_deterministic():
    df = load_matches(FIX, odds_source="pinnacle")
    kw = dict(train_df=df.iloc[:2], test_df=df.iloc[2:], odds_source="pinnacle",
              devig_method="shin", anchor_cfg=dict(k=20, home_adv=70, start_rating=1500),
              value_cfg=dict(min_edge=0.0, odds_min=1.0, odds_max=99.0), skip_first_rounds=0)
    assert run_backtest(**kw)["bets"] == run_backtest(**kw)["bets"]
