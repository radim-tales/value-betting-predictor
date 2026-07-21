import json
from pathlib import Path
from vbp.live.adapter import event_to_books, classify_book

FIX = json.loads((Path(__file__).parent / "fixtures" / "odds_sample.json").read_text(encoding="utf-8"))

def test_event_to_books_maps_hda():
    books = event_to_books(FIX[0])
    assert set(books) == {"pinnacle", "williamhill", "betfair_ex_eu"}
    assert books["pinnacle"] == {"H": 1.90, "D": 3.50, "A": 4.20}

def test_event_without_all_outcomes_skipped_book():
    books = event_to_books(FIX[1])
    assert "williamhill" in books

def test_classify_book():
    assert classify_book("pinnacle") == "sharp"
    assert classify_book("betfair_ex_eu") == "exchange"
    assert classify_book("matchbook") == "exchange"
    assert classify_book("williamhill") == "soft"
    assert classify_book("some_unknown_book") == "soft"   # default

def test_book_missing_an_outcome_is_dropped():
    ev = {"home_team": "A", "away_team": "B", "bookmakers": [
        {"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
            {"name": "A", "price": 2.0}, {"name": "B", "price": 4.0}, {"name": "Draw", "price": 3.3}]}]},
        {"key": "partial", "markets": [{"key": "h2h", "outcomes": [   # missing Draw -> dropped
            {"name": "A", "price": 2.1}, {"name": "B", "price": 4.2}]}]},
    ]}
    books = event_to_books(ev)
    assert "pinnacle" in books and "partial" not in books
