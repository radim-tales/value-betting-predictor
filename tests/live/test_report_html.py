from datetime import datetime, timezone
from vbp.live.store import Store
from vbp.live.report_html import build_context, render_html

NOW = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)

def _seed(tmp_path):
    s = Store(tmp_path / "bets.jsonl", tmp_path / "lines.json")
    meta = {"league": "soccer_x", "league_tier": "neglected", "home": "Alfa",
            "away": "Beta", "kickoff": "2026-08-01T20:00:00Z"}
    s.update_line("m1", meta, {"H": 0.5, "D": 0.3, "A": 0.2})
    s.add_bet({"match_id": "m1", "outcome": "H", "book": "coolbet", "price": 2.3,
               "book_type": "soft", "league_tier": "neglected", "home": "Alfa", "away": "Beta"})
    s.set_result("m1", "H")
    return s

def test_build_context_counts(tmp_path):
    ctx = build_context(_seed(tmp_path), now=NOW)
    assert ctx["total_bets"] == 1
    assert ctx["settled_bets"] == 1
    assert ctx["kicked_off"] == 1          # kickoff 2026-08-01 < NOW
    assert ("soft", "neglected") in ctx["by"]

def test_render_html_is_self_contained(tmp_path):
    html = render_html(build_context(_seed(tmp_path), now=NOW))
    assert html.startswith("<!doctype html>")
    assert "Alfa" in html and "Beta" in html      # settled bet rendered
    assert "http://" not in html and "https://cdn" not in html  # zadne externi zdroje
    assert "{" not in html.split("<style>")[0]    # zadny nevyplneny format placeholder v tele

def test_render_html_empty_state(tmp_path):
    # prazdny store nesmi spadnout
    s = Store(tmp_path / "b.jsonl", tmp_path / "l.json")
    html = render_html(build_context(s, now=NOW))
    assert "Zatím žádná settled" in html
