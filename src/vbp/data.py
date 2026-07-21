from __future__ import annotations
from pathlib import Path
import pandas as pd
import requests

BASE_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
ODDS_PREFIX = {"pinnacle": "PS", "avg": "Avg", "bet365": "B365"}

def _odds_cols(source: str) -> list[str]:
    p = ODDS_PREFIX[source]
    # open = bez suffixu, close = suffix C
    return [f"{p}H", f"{p}D", f"{p}A", f"{p}CH", f"{p}CD", f"{p}CA"]

def load_matches(path: str | Path, odds_source: str = "pinnacle") -> pd.DataFrame:
    """Load football-data CSV keeping ONLY whitelisted pre-match columns.
    Anti-leak: post-match statistics (shots, corners, cards, half-time) are dropped."""
    whitelist = BASE_COLS + _odds_cols(odds_source)
    raw = pd.read_csv(path)
    missing = [c for c in whitelist if c not in raw.columns]
    if missing:
        raise ValueError(f"missing required columns for source {odds_source}: {missing}")
    df = raw[whitelist].copy()
    df = df.dropna(subset=_odds_cols(odds_source)).reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="raise")
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    return df

WHITELIST_POSTFIX = ("H", "D", "A", "CH", "CD", "CA")  # exposed for tests/docs

def download_season(season: str, league: str, dest: str | Path) -> Path:
    """Thin convenience wrapper to download a football-data.co.uk season CSV.

    Real data is fetched manually from https://www.football-data.co.uk/englandm.php
    (file e.g. E1.csv per season) into data/raw/<season>_E1.csv. This helper is
    pure convenience, not exercised by tests against the network.
    """
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest
