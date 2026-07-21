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

def test_drops_rows_with_missing_odds(tmp_path):
    csv_text = (
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,PSH,PSD,PSA,PSCH,PSCD,PSCA,HS,AS\n"
        "09/08/2024,Alpha,Beta,2,1,H,1.90,3.50,4.20,1.85,3.60,4.40,12,7\n"
        "09/08/2024,Gamma,Delta,0,0,D,2.60,3.20,2.80,2.55,3.25,2.90,9,10\n"
        "16/08/2024,Beta,Gamma,1,2,A,2.10,3.30,3.60,2.20,3.30,,8,11\n"
        "16/08/2024,Delta,Alpha,1,1,D,3.80,3.60,2.00,3.90,3.55,1.98,6,14\n"
    )
    path = tmp_path / "with_missing_odds.csv"
    path.write_text(csv_text, encoding="utf-8")
    df = load_matches(path, odds_source="pinnacle")
    assert len(df) == 3
    assert not any((df["HomeTeam"] == "Beta") & (df["AwayTeam"] == "Gamma"))
