from vbp.value_filter import select_bet

CFG = dict(min_edge=0.03, odds_min=1.6, odds_max=4.5)

def test_selects_outcome_with_edge_above_threshold():
    model = {"H": 0.55, "D": 0.25, "A": 0.20}
    fair  = {"H": 0.50, "D": 0.27, "A": 0.23}
    odds  = {"H": 1.90, "D": 3.50, "A": 4.20}
    bet = select_bet(model, fair, odds, **CFG)
    assert bet is not None and bet["outcome"] == "H"
    assert abs(bet["edge"] - 0.05) < 1e-9

def test_no_bet_when_below_threshold():
    model = {"H": 0.51, "D": 0.26, "A": 0.23}
    fair  = {"H": 0.50, "D": 0.27, "A": 0.23}
    odds  = {"H": 1.90, "D": 3.50, "A": 4.20}
    assert select_bet(model, fair, odds, **CFG) is None

def test_odds_out_of_range_excluded():
    model = {"H": 0.90, "D": 0.06, "A": 0.04}
    fair  = {"H": 0.80, "D": 0.12, "A": 0.08}
    odds  = {"H": 1.20, "D": 8.0, "A": 15.0}   # H below odds_min
    assert select_bet(model, fair, odds, **CFG) is None

def test_argmax_edge_picks_single_outcome():
    model = {"H": 0.40, "D": 0.35, "A": 0.25}
    fair  = {"H": 0.33, "D": 0.30, "A": 0.20}
    odds  = {"H": 2.60, "D": 3.10, "A": 3.60}
    bet = select_bet(model, fair, odds, **CFG)
    # edges: H .07, D .05, A .05 -> H wins
    assert bet["outcome"] == "H"
