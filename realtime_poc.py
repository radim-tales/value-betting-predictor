"""Proof-of-concept: The Odds API -> vbp value detection (LIVE odds).

Reuses vbp.devig (Shin) to turn Pinnacle's line into the "truth", finds the best
price across all books, and flags value bets (positive EV vs Pinnacle).

Run (needs a free key from https://the-odds-api.com):
    ODDS_API_KEY=xxxx  .venv/Scripts/python.exe realtime_poc.py
    .venv/Scripts/python.exe realtime_poc.py --key xxxx --sport soccer_epl
    .venv/Scripts/python.exe realtime_poc.py --key xxxx --list-sports
    .venv/Scripts/python.exe realtime_poc.py --key xxxx --sport soccer_efl_champ --edge 0.03

Nothing is bet. It only reads odds and prints what a value detector would flag now.
"""
from __future__ import annotations
import argparse, os, sys
import requests
from vbp.devig import devig

BASE = "https://api.the-odds-api.com/v4"


def get(path, **params):
    r = requests.get(f"{BASE}{path}", params=params, timeout=30)
    quota = {k: r.headers.get(k) for k in ("x-requests-remaining", "x-requests-used")}
    if r.status_code != 200:
        sys.exit(f"API error {r.status_code}: {r.text[:300]}\nquota={quota}")
    return r.json(), quota


def list_sports(key):
    data, q = get("/sports", apiKey=key)
    soccer = [s for s in data if s["key"].startswith("soccer") and s.get("active")]
    print(f"Active soccer competitions ({len(soccer)}):  quota remaining={q['x-requests-remaining']}")
    for s in sorted(soccer, key=lambda s: s["key"]):
        print(f"  {s['key']:<28} {s['title']}")


def event_to_books(ev):
    """The Odds API event -> {book_key: {'H','D','A'}} decimal odds."""
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


def find_value(ev, min_edge, odds_min, odds_max, truth_key="pinnacle"):
    books = event_to_books(ev)
    if truth_key not in books:
        return None, books  # no sharp reference -> can't judge value
    t = books[truth_key]
    fair = dict(zip("HDA", devig([t["H"], t["D"], t["A"]], "shin")))  # TRUTH
    bets = []
    for o in "HDA":
        price, book = max((books[b][o], b) for b in books)  # best price across books
        edge = fair[o] * price - 1  # EV vs Pinnacle-fair
        if edge >= min_edge and odds_min <= price <= odds_max:
            bets.append(dict(outcome=o, price=price, book=book, edge=edge, fair=fair[o]))
    return bets, books


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=os.environ.get("ODDS_API_KEY"))
    ap.add_argument("--sport", default="soccer_efl_champ")
    ap.add_argument("--regions", default="eu,uk")
    ap.add_argument("--edge", type=float, default=0.03)
    ap.add_argument("--odds-min", type=float, default=1.6)
    ap.add_argument("--odds-max", type=float, default=8.0)
    ap.add_argument("--list-sports", action="store_true")
    a = ap.parse_args(argv)
    try:  # Windows console is often cp1250; force UTF-8 so accented team names don't crash
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if not a.key:
        sys.exit("No API key. Set ODDS_API_KEY or pass --key. Get a free one at https://the-odds-api.com")

    if a.list_sports:
        return list_sports(a.key)

    data, q = get(f"/sports/{a.sport}/odds", apiKey=a.key, regions=a.regions,
                  markets="h2h", oddsFormat="decimal")
    print(f"Sport={a.sport}  events={len(data)}  quota remaining={q['x-requests-remaining']} "
          f"(used {q['x-requests-used']})\n")
    if not data:
        print("No upcoming events (off-season / none listed). Try --list-sports to pick an active one.")
        return

    total_value = 0
    no_pin = 0
    with_pin = 0
    for ev in data:
        bets, books = find_value(ev, a.edge, a.odds_min, a.odds_max)
        head = f"{ev['home_team']} vs {ev['away_team']}  ({ev['commence_time']})  books={len(books)}"
        if bets is None:
            no_pin += 1  # sharp reference not posted yet -> can't judge value
            continue
        with_pin += 1
        if not bets:
            continue
        print(head)
        for b in bets:
            total_value += 1
            print(f"    VALUE  {b['outcome']}  best {b['price']:.2f} @ {b['book']:<12} "
                  f"edge {b['edge']*100:+.1f}%  (pinnacle-fair p={b['fair']:.3f})")
        print()
    print(f"== {total_value} value opportunities at edge>={a.edge*100:.0f}% "
          f"(odds {a.odds_min}-{a.odds_max}), truth=Pinnacle ==")
    print(f"   {with_pin} events had a Pinnacle reference; {no_pin} did not yet "
          f"(sharp line posts closer to kickoff).")


if __name__ == "__main__":
    main()
