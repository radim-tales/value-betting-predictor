import json
from pathlib import Path
from vbp.live.adapter import event_to_books
from vbp.live.value import find_value

FIX = json.loads((Path(__file__).parent / "fixtures" / "odds_sample.json").read_text(encoding="utf-8"))

def test_find_value_needs_pinnacle():
    assert find_value(event_to_books(FIX[1]), min_edge=0.0) == []   # m2 has no pinnacle -> no truth

def test_find_value_returns_candidates_with_book_type():
    cands = find_value(event_to_books(FIX[0]), min_edge=0.0, odds_min=1.0, odds_max=99.0)
    assert cands, "expected value candidates on m1"
    c = cands[0]
    assert set(c) >= {"outcome", "price", "book", "book_type", "edge", "pin_fair"}
    assert c["book_type"] in {"sharp", "soft", "exchange"}
    # best price per outcome is picked
    assert c["price"] == max(event_to_books(FIX[0])[b][c["outcome"]] for b in event_to_books(FIX[0]))
