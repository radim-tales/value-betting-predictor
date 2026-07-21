from __future__ import annotations

SHARP = {"pinnacle"}
EXCHANGE = {"betfair_ex_eu", "betfair_ex_uk", "matchbook"}

def classify_book(key: str) -> str:
    if key in SHARP: return "sharp"
    if key in EXCHANGE: return "exchange"
    return "soft"

def event_to_books(ev: dict) -> dict:
    """The Odds API event -> {book_key: {'H','D','A'}} decimal odds (drop books missing an outcome)."""
    home, away = ev["home_team"], ev["away_team"]
    books = {}
    for bk in ev.get("bookmakers", []):
        m = next((x for x in bk.get("markets", []) if x["key"] == "h2h"), None)
        if not m:
            continue
        px = {o["name"]: o["price"] for o in m["outcomes"]}
        if home in px and away in px and "Draw" in px:
            books[bk["key"]] = {"H": px[home], "D": px["Draw"], "A": px[away]}
    return books
