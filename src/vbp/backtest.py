from __future__ import annotations
import pandas as pd
from .anchor import EloAnchor
from .devig import devig
from .value_filter import select_bet
from .metrics import clv

def _odds_dicts(row, source):
    p = {"pinnacle": "PS", "avg": "Avg", "bet365": "B365"}[source]
    open_ = {"H": row[f"{p}H"], "D": row[f"{p}D"], "A": row[f"{p}A"]}
    close = {"H": row[f"{p}CH"], "D": row[f"{p}CD"], "A": row[f"{p}CA"]}
    return open_, close

def run_backtest(train_df, test_df, odds_source, devig_method,
                 anchor_cfg, value_cfg, skip_first_rounds):
    anchor = EloAnchor(**anchor_cfg)
    # 1) run Elo through train, collect deltas + labels, fit mapping on TRAIN only
    train_matches = train_df.to_dict("records")
    deltas = anchor.run_and_collect(train_matches)
    anchor.fit_mapping(deltas, [m["FTR"] for m in train_matches])

    audit, bets = [], []
    test_matches = test_df.reset_index(drop=True).to_dict("records")
    for i, m in enumerate(test_matches):
        delta = anchor.delta(m["HomeTeam"], m["AwayTeam"])
        p_model = anchor.predict_proba(delta)
        open_odds, close_odds = _odds_dicts(m, odds_source)
        fair_open = dict(zip(("H", "D", "A"), devig([open_odds["H"], open_odds["D"], open_odds["A"]], devig_method)))
        fair_close = dict(zip(("H", "D", "A"), devig([close_odds["H"], close_odds["D"], close_odds["A"]], devig_method)))
        bet = None
        if i >= skip_first_rounds:
            bet = select_bet(p_model, fair_open, open_odds, **value_cfg)
        row = {"i": i, "home": m["HomeTeam"], "away": m["AwayTeam"], "delta": delta,
               "anchor_p": p_model, "fair_open": fair_open,
               "open_odds": open_odds, "close_odds": close_odds,   # raw odds for baselines/Plan B
               "result": m["FTR"], "bet": bet}
        audit.append(row)
        if bet is not None:
            o = bet["outcome"]
            bets.append({"outcome": o, "odds": open_odds[o], "won": (m["FTR"] == o),
                         "clv": clv(open_odds[o], fair_close[o]),
                         "model_p": p_model[o], "fair_p": fair_open[o]})
        anchor.update(m)   # walk-forward rating update, mapping stays frozen
    return {"audit": audit, "bets": bets}
