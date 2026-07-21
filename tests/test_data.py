from pathlib import Path
import pandas as pd
from vbp.data import load_matches, WHITELIST_POSTFIX

FIX = Path(__file__).parent / "fixtures" / "mini_league.csv"

def test_loads_only_whitelisted_columns():
    df = load_matches(FIX, odds_source="pinnacle")
    # post-match shots must be gone
    assert "HS" not in df.columns and "AS" not in df.columns
    # required whitelisted columns present
    for col in ["Date", "HomeTeam", "AwayTeam", "FTR", "PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA"]:
        assert col in df.columns

def test_parses_dates_and_sorts_chronologically():
    df = load_matches(FIX, odds_source="pinnacle")
    assert str(df["Date"].dtype).startswith("datetime")
    assert df["Date"].is_monotonic_increasing

def test_result_label_is_hda():
    df = load_matches(FIX, odds_source="pinnacle")
    assert set(df["FTR"].unique()).issubset({"H", "D", "A"})
