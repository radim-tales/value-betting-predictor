import json
from vbp.live.store import Store

def test_add_bet_dedups(tmp_path):
    s = Store(tmp_path / "bets.jsonl", tmp_path / "lines.json")
    bet = {"match_id": "m1", "outcome": "H", "book": "williamhill", "price": 1.85,
           "book_type": "soft", "edge": 0.04, "league": "x", "league_tier": "liquid"}
    assert s.add_bet(bet) is True         # new
    assert s.add_bet(bet) is False        # duplicate (match_id+outcome+book) -> not added
    assert len(s.load_bets()) == 1

def test_update_line_sets_open_then_close(tmp_path):
    s = Store(tmp_path / "bets.jsonl", tmp_path / "lines.json")
    fair1 = {"H": 0.52, "D": 0.27, "A": 0.21}
    fair2 = {"H": 0.55, "D": 0.26, "A": 0.19}
    meta = {"league": "x", "league_tier": "liquid", "home": "A", "away": "B", "kickoff": "2026-08-01T20:00:00Z"}
    s.update_line("m1", meta, fair1)      # first -> open and close = fair1
    s.update_line("m1", meta, fair2)      # later -> open stays fair1, close = fair2
    ln = s.load_lines()["m1"]
    assert ln["pin_open"] == fair1 and ln["pin_close"] == fair2

def test_set_result(tmp_path):
    s = Store(tmp_path / "bets.jsonl", tmp_path / "lines.json")
    s.update_line("m1", {"league":"x","league_tier":"liquid","home":"A","away":"B","kickoff":"t"}, {"H":.5,"D":.3,"A":.2})
    s.set_result("m1", "H")
    ln = s.load_lines()["m1"]
    assert ln["result"] == "H" and ln["settled"] is True

def test_settle_date_gate(tmp_path):
    # settle gate se ridi ulozenym datem, aby byl imunni vuci zpozdeni GitHub cronu
    s = Store(tmp_path / "bets.jsonl", tmp_path / "lines.json")
    assert s.last_settle_date() is None          # cerstvy stav -> settle se ma spustit
    s.mark_settled("2026-08-04")
    assert s.last_settle_date() == "2026-08-04"   # tyz den -> uz nesettlovat
    s2 = Store(tmp_path / "bets.jsonl", tmp_path / "lines.json")
    assert s2.last_settle_date() == "2026-08-04"  # prezije napric behy (perzistentni)
