import json
from pathlib import Path
from vbp.live.store import Store
from vbp.live.settle import result_from_scores, settle_finished

SCORES = json.loads((Path(__file__).parent / "fixtures" / "scores_sample.json").read_text(encoding="utf-8"))

def test_result_from_scores():
    assert result_from_scores(SCORES[0]) == "H"     # 2-1 home win
    assert result_from_scores(SCORES[1]) is None    # not completed

def test_settle_writes_result_only_for_finished(tmp_path):
    s = Store(tmp_path / "bets.jsonl", tmp_path / "lines.json")
    for mid in ("m1", "m2"):
        s.update_line(mid, {"league":"x","league_tier":"liquid","home":"A","away":"B","kickoff":"t"}, {"H":.5,"D":.3,"A":.2})
    settle_finished(s, SCORES)
    lines = s.load_lines()
    assert lines["m1"]["result"] == "H" and lines["m1"]["settled"] is True
    assert lines["m2"]["settled"] is False
