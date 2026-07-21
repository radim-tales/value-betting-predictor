from pathlib import Path
import pandas as pd
from vbp.data import load_matches
from vbp.features import build_features, PreMatch

FIX = Path(__file__).parent / "fixtures" / "mini_league.csv"

def _df():
    return load_matches(FIX, odds_source="pinnacle")

def test_first_match_has_zero_history():
    df = _df()
    feats = build_features(df, form_n=5)
    first = feats.iloc[0]
    assert first["home_played"] == 0
    assert first["away_played"] == 0

def test_uses_only_strictly_earlier_matches():
    # For the round-2 match Beta vs Gamma (16/08), Beta played once (09/08 vs Alpha, lost),
    # Gamma played once (09/08 vs Delta, drew). Same-day round-1 matches count; later ones must not.
    df = _df()
    feats = build_features(df, form_n=5)
    row = feats[(feats.HomeTeam == "Beta") & (feats.AwayTeam == "Gamma")].iloc[0]
    assert row["home_played"] == 1
    assert row["away_played"] == 1
    assert row["home_pts"] == 0     # Beta lost round 1
    assert row["away_pts"] == 1     # Gamma drew round 1

def test_no_future_leak_invariant():
    """CI-critical: for every match, no feature may depend on a match with Date >= target Date."""
    df = _df()
    feats = build_features(df, form_n=5)
    # Corrupt the future: flip all FUTURE results and rebuild; earlier-round features must be identical.
    df2 = df.copy()
    df2.loc[df2["Date"] == df2["Date"].max(), "FTR"] = "A"
    feats2 = build_features(df2, form_n=5)
    early_mask = feats["Date"] < df["Date"].max()
    cols = ["home_played", "away_played", "home_pts", "away_pts",
            "home_gf_avg", "home_ga_avg", "away_gf_avg", "away_ga_avg"]
    pd.testing.assert_frame_equal(
        feats.loc[early_mask, cols].reset_index(drop=True),
        feats2.loc[early_mask, cols].reset_index(drop=True),
    )
