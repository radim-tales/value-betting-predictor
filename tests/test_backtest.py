from pathlib import Path
import pandas as pd
from vbp.data import load_matches
from vbp.backtest import run_backtest
from vbp.anchor import EloAnchor

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

def test_warmup_advances_ratings_without_refitting_mapping():
    df = load_matches(FIX, odds_source="pinnacle")
    train_df, warmup_df, test_df = df.iloc[:2], df.iloc[2:3], df.iloc[3:]
    kw = dict(odds_source="pinnacle", devig_method="shin",
              anchor_cfg=dict(k=20, home_adv=70, start_rating=1500),
              value_cfg=dict(min_edge=0.0, odds_min=1.0, odds_max=99.0), skip_first_rounds=0)

    result = run_backtest(train_df=train_df, test_df=test_df, warmup_df=warmup_df, **kw)
    assert len(result["audit"]) == len(test_df)  # warmup does not become test/audit rows

    # Reference: fit an anchor on the same train only, to snapshot the mapping coefficients.
    ref_anchor = EloAnchor(**kw["anchor_cfg"])
    train_matches = train_df.to_dict("records")
    deltas = ref_anchor.run_and_collect(train_matches)
    ref_anchor.fit_mapping(deltas, [m["FTR"] for m in train_matches])
    ref_coef = ref_anchor._clf.coef_.copy()

    # Advance the reference anchor through warmup WITHOUT refitting - ratings move, mapping doesn't.
    for m in warmup_df.to_dict("records"):
        ref_anchor.update(m)
    assert (ref_anchor._clf.coef_ == ref_coef).all()  # mapping untouched by warmup
    assert ref_anchor.rating("Beta") != EloAnchor(**kw["anchor_cfg"]).rating("Beta")  # ratings did move

def test_skip_first_rounds_is_round_based():
    # 2 matches on date A, 2 matches on date B; skip_first_rounds=1 should skip
    # BOTH date-A matches (one round), not just the first match.
    train_df = load_matches(FIX, odds_source="pinnacle").iloc[:2]
    test_df = pd.DataFrame([
        {"Date": pd.Timestamp("2024-01-01"), "HomeTeam": "Alpha", "AwayTeam": "Beta",
         "FTHG": 1, "FTAG": 0, "FTR": "H",
         "PSH": 2.0, "PSD": 3.0, "PSA": 4.0, "PSCH": 2.0, "PSCD": 3.0, "PSCA": 4.0},
        {"Date": pd.Timestamp("2024-01-01"), "HomeTeam": "Gamma", "AwayTeam": "Delta",
         "FTHG": 0, "FTAG": 1, "FTR": "A",
         "PSH": 2.0, "PSD": 3.0, "PSA": 4.0, "PSCH": 2.0, "PSCD": 3.0, "PSCA": 4.0},
        {"Date": pd.Timestamp("2024-01-08"), "HomeTeam": "Beta", "AwayTeam": "Alpha",
         "FTHG": 1, "FTAG": 1, "FTR": "D",
         "PSH": 2.0, "PSD": 3.0, "PSA": 4.0, "PSCH": 2.0, "PSCD": 3.0, "PSCA": 4.0},
        {"Date": pd.Timestamp("2024-01-08"), "HomeTeam": "Delta", "AwayTeam": "Gamma",
         "FTHG": 2, "FTAG": 0, "FTR": "H",
         "PSH": 2.0, "PSD": 3.0, "PSA": 4.0, "PSCH": 2.0, "PSCD": 3.0, "PSCA": 4.0},
    ])
    result = run_backtest(
        train_df=train_df, test_df=test_df,
        odds_source="pinnacle", devig_method="shin",
        anchor_cfg=dict(k=20, home_adv=70, start_rating=1500),
        value_cfg=dict(min_edge=0.0, odds_min=1.0, odds_max=99.0),
        skip_first_rounds=1,
    )
    date_a_rows = [r for r in result["audit"] if r["home"] in ("Alpha", "Gamma") and r["i"] < 2]
    assert all(r["bet"] is None for r in date_a_rows)
    assert any(r["bet"] is not None for r in result["audit"] if r["i"] >= 2)

def test_backtest_is_deterministic():
    df = load_matches(FIX, odds_source="pinnacle")
    kw = dict(train_df=df.iloc[:2], test_df=df.iloc[2:], odds_source="pinnacle",
              devig_method="shin", anchor_cfg=dict(k=20, home_adv=70, start_rating=1500),
              value_cfg=dict(min_edge=0.0, odds_min=1.0, odds_max=99.0), skip_first_rounds=0)
    assert run_backtest(**kw)["bets"] == run_backtest(**kw)["bets"]
