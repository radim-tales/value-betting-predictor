import json
from pathlib import Path
from vbp.live.store import Store
from vbp.live.run import run_once

FIXDIR = Path(__file__).parent / "fixtures"
ODDS = json.loads((FIXDIR / "odds_sample.json").read_text(encoding="utf-8"))
SCORES = json.loads((FIXDIR / "scores_sample.json").read_text(encoding="utf-8"))

from datetime import datetime, timezone
NOW = datetime(2026, 7, 1, tzinfo=timezone.utc)   # before fixture kickoffs (2026-08-01)

class FakeClient:
    def fetch_odds(self, sport, regions="eu,uk"): return ODDS, {"remaining":"499"}
    def fetch_scores(self, sport, days_from=1): return SCORES, {"remaining":"498"}

def test_run_once_logs_value_and_settles(tmp_path):
    s = Store(tmp_path / "bets.jsonl", tmp_path / "lines.json")
    run_once(FakeClient(), [("soccer_brazil_campeonato","liquid")], s,
             regions="eu", min_edge=0.0, odds_min=1.0, odds_max=99.0, now=NOW)
    bets = s.load_bets(); lines = s.load_lines()
    assert any(b["match_id"] == "m1" for b in bets)       # m1 has pinnacle -> value logged
    assert all(b["match_id"] != "m2" for b in bets)       # m2 no pinnacle -> nothing
    assert lines["m1"]["pin_open"] is not None            # line snapshotted
    assert lines["m1"]["settled"] is True                 # m1 finished in scores -> settled
    for b in bets:
        assert "book_type" in b and "league_tier" in b and "ts_detected" in b

def test_run_once_skips_started_matches(tmp_path):
    s = Store(tmp_path / "bets.jsonl", tmp_path / "lines.json")
    after = datetime(2099, 1, 1, tzinfo=timezone.utc)     # after all fixture kickoffs
    run_once(FakeClient(), [("soccer_brazil_campeonato","liquid")], s,
             regions="eu", min_edge=0.0, odds_min=1.0, odds_max=99.0, now=after)
    # every event already kicked off relative to `after` -> no snapshots, no bets (anti-leak)
    assert s.load_bets() == []
    assert "m1" not in s.load_lines()
