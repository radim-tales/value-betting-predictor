from vbp.live.store import Store
from vbp.live.report import summarize, render


def _seed(tmp_path):
    s = Store(tmp_path / "bets.jsonl", tmp_path / "lines.json")
    meta = {"league": "BR", "league_tier": "liquid", "home": "A", "away": "B", "kickoff": "t"}
    s.update_line("m1", meta, {"H": 0.50, "D": 0.30, "A": 0.20})   # open
    s.update_line("m1", {**meta}, {"H": 0.48, "D": 0.30, "A": 0.22})  # close (A more likely)
    s.set_result("m1", "A")
    s.add_bet({"match_id": "m1", "outcome": "A", "book": "williamhill", "price": 5.00,
               "book_type": "soft", "edge": 0.10, "league": "BR", "league_tier": "liquid"})
    return s


def test_summarize_computes_clv_and_groups(tmp_path):
    s = _seed(tmp_path)
    rep = summarize(s)
    # bet on A at 5.00, pin_close_fair A = 0.22 -> CLV = 0.22*5 - 1 = 0.10, won (result A)
    grp = rep["by"][("soft", "liquid")]
    assert abs(grp["mean_clv"] - 0.10) < 1e-9
    assert grp["n"] == 1 and grp["wins"] == 1
    assert "clv_ci" in grp and "roi" in grp


def test_render_includes_caveats(tmp_path):
    s = _seed(tmp_path)
    rep = summarize(s)
    out = render(rep)
    assert "Caveaty" in out
    assert "slippage" in out.lower()
    assert "proxy" in out.lower()


def test_summarize_ignores_unsettled(tmp_path):
    s = Store(tmp_path / "bets.jsonl", tmp_path / "lines.json")
    s.update_line("m9", {"league": "BR", "league_tier": "liquid", "home": "A", "away": "B", "kickoff": "t"},
                  {"H": .5, "D": .3, "A": .2})
    s.add_bet({"match_id": "m9", "outcome": "H", "book": "bwin", "price": 2.0,
               "book_type": "soft", "edge": 0.05, "league": "BR", "league_tier": "liquid"})
    rep = summarize(s)
    assert rep["settled_bets"] == 0
