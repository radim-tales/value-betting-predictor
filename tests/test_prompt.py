from vbp.prompt import build_correction_prompt, build_reflection_prompt


def test_correction_prompt_has_no_odds_and_is_anonymized():
    matches = [{"match_id": "0", "home": "Team_0", "away": "Team_1",
                "anchor_p": {"H": 0.5, "D": 0.3, "A": 0.2},
                "features": {"home_pts": 6, "away_pts": 3}}]
    p = build_correction_prompt(matches, playbook_text="## Priors\n- x\n")
    assert "Team_0" in p and "Team_1" in p
    assert "odds" not in p.lower() and "kurz" not in p.lower()   # anti-leak: no market
    assert "## Priors" in p
    assert "match_id" in p                                       # instructs JSON keying


def test_reflection_prompt_has_metrics_not_stories():
    report = {"n_bets": 20, "roi": -0.05, "clv": -0.04, "brier": 0.63,
              "brier_by_outcome": {"H": 0.2, "D": 0.3, "A": 0.4},
              "overconfidence": 0.08}
    p = build_reflection_prompt(report, playbook_text="## Priors\n- x\n")
    assert "brier" in p.lower() or "kalibr" in p.lower()
    assert "0.63" in p or "-0.05" in p
    assert "## Priors" in p
