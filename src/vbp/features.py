from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass
class PreMatch:
    """Column contract produced by build_features (documented for downstream modules)."""
    cols = [
        "home_played", "away_played", "home_pts", "away_pts",
        "home_gf_avg", "home_ga_avg", "away_gf_avg", "away_ga_avg",
        "home_rest_days", "away_rest_days",
    ]

def _result_points(row, team):
    if row["FTR"] == "D":
        return 1
    winner = row["HomeTeam"] if row["FTR"] == "H" else row["AwayTeam"]
    return 3 if winner == team else 0

def build_features(df: pd.DataFrame, form_n: int = 5) -> pd.DataFrame:
    """For each match, compute pre-match features from STRICTLY earlier matches only
    (Date < target Date). Same-day matches are excluded (no reliable kickoff times)."""
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    out_rows = []
    for i, row in df.iterrows():
        past = df[df["Date"] < row["Date"]]
        feat = {"Date": row["Date"], "HomeTeam": row["HomeTeam"], "AwayTeam": row["AwayTeam"]}
        for side, team in (("home", row["HomeTeam"]), ("away", row["AwayTeam"])):
            th = past[(past["HomeTeam"] == team) | (past["AwayTeam"] == team)]
            recent = th.tail(form_n)
            feat[f"{side}_played"] = len(th)
            feat[f"{side}_pts"] = sum(_result_points(r, team) for _, r in recent.iterrows())
            gf = [(r["FTHG"] if r["HomeTeam"] == team else r["FTAG"]) for _, r in recent.iterrows()]
            ga = [(r["FTAG"] if r["HomeTeam"] == team else r["FTHG"]) for _, r in recent.iterrows()]
            feat[f"{side}_gf_avg"] = (sum(gf) / len(gf)) if gf else 0.0
            feat[f"{side}_ga_avg"] = (sum(ga) / len(ga)) if ga else 0.0
            last_date = th["Date"].max() if len(th) else pd.NaT
            feat[f"{side}_rest_days"] = (row["Date"] - last_date).days if pd.notna(last_date) else -1
        out_rows.append(feat)
    return pd.DataFrame(out_rows)
