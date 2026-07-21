import numpy as np
from vbp.anchor import EloAnchor

def _matches():
    # Home team clearly stronger; deterministic-ish results to move Elo
    return [
        {"HomeTeam": "A", "AwayTeam": "B", "FTR": "H"},
        {"HomeTeam": "B", "AwayTeam": "A", "FTR": "A"},
        {"HomeTeam": "A", "AwayTeam": "C", "FTR": "H"},
        {"HomeTeam": "C", "AwayTeam": "B", "FTR": "D"},
        {"HomeTeam": "B", "AwayTeam": "C", "FTR": "H"},
        {"HomeTeam": "C", "AwayTeam": "A", "FTR": "A"},
    ]

def test_elo_diff_grows_for_winner():
    anchor = EloAnchor(k=20, home_adv=70, start_rating=1500)
    diffs = anchor.run_and_collect(_matches())
    assert len(diffs) == len(_matches())
    # after processing, stronger team A should have higher rating than C
    assert anchor.rating("A") > anchor.rating("C")

def test_predict_proba_sums_to_one_and_is_calibrated_shape():
    anchor = EloAnchor(k=20, home_adv=70, start_rating=1500)
    diffs = anchor.run_and_collect(_matches())
    labels = [m["FTR"] for m in _matches()]
    anchor.fit_mapping(diffs, labels)
    p = anchor.predict_proba(delta=200.0)   # strong home edge
    assert abs(p["H"] + p["D"] + p["A"] - 1.0) < 1e-9
    assert p["H"] > p["A"]                   # positive delta -> home favored

def test_mapping_not_refit_on_test_ratings_still_advance():
    """Ratings update walk-forward on any stream, but the H/D/A mapping is frozen after fit."""
    anchor = EloAnchor(k=20, home_adv=70, start_rating=1500)
    diffs = anchor.run_and_collect(_matches())
    anchor.fit_mapping(diffs, [m["FTR"] for m in _matches()])
    coef_before = anchor._clf.coef_.copy()
    anchor.update(_matches()[0])             # advancing ratings must NOT touch the mapping
    assert np.array_equal(anchor._clf.coef_, coef_before)

def test_predict_proba_robust_to_missing_class_in_train():
    """If train split lacks an outcome (e.g. no 'A'), that class must be 0.0, not a crash."""
    anchor = EloAnchor(k=20, home_adv=70, start_rating=1500)
    two = [{"HomeTeam": "A", "AwayTeam": "B", "FTR": "H"},
           {"HomeTeam": "B", "AwayTeam": "A", "FTR": "D"}]   # only H and D
    diffs = anchor.run_and_collect(two)
    anchor.fit_mapping(diffs, [m["FTR"] for m in two])
    p = anchor.predict_proba(delta=50.0)
    assert set(p.keys()) == {"H", "D", "A"}
    assert p["A"] == 0.0
    assert abs(p["H"] + p["D"] + p["A"] - 1.0) < 1e-9
