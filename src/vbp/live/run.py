from __future__ import annotations
from datetime import datetime, timezone
from vbp.devig import devig
from .adapter import event_to_books
from .value import find_value
from .settle import settle_finished

def _pin_fair(books):
    t = books["pinnacle"]
    return dict(zip("HDA", devig([t["H"], t["D"], t["A"]], "shin")))

def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

def run_once(client, leagues, store, regions="eu,uk",
             min_edge=0.03, odds_min=1.6, odds_max=8.0, now=None,
             do_settle=True, settle_days_from=3):
    """Jeden běh: poll odds + snapshot linií, volitelně settle doběhlých zápasů.

    `do_settle=False` šetří kredity - scores stojí 2 kredity/liga a stačí 1x denně
    (výsledky nikam neutečou), zatímco odds se pollují 3x denně kvůli close snapshotu.
    """
    now = now or datetime.now(timezone.utc)
    for sport, tier in leagues:
        events, quota = client.fetch_odds(sport, regions=regions)
        print(f"[quota] {sport}: remaining={quota.get('remaining')}")
        for ev in events:
            if now >= _parse(ev["commence_time"]):   # ANTI-LEAK: never snapshot/bet a started match
                continue
            books = event_to_books(ev)
            if "pinnacle" not in books:
                continue
            meta = {"league": sport, "league_tier": tier, "home": ev["home_team"],
                    "away": ev["away_team"], "kickoff": ev["commence_time"]}
            store.update_line(ev["id"], meta, _pin_fair(books))
            for c in find_value(books, min_edge, odds_min, odds_max):
                store.add_bet({"match_id": ev["id"], "league": sport, "league_tier": tier,
                               "home": ev["home_team"], "away": ev["away_team"],
                               "kickoff": ev["commence_time"], "ts_detected": now.isoformat(), **c})
        if do_settle:
            scores, _ = client.fetch_scores(sport, days_from=settle_days_from)
            settle_finished(store, scores)   # settle is independent of the kickoff gate

def main():
    from .config import (LEAGUES, REGIONS, MIN_EDGE, ODDS_MIN, ODDS_MAX,
                         BETS_FILE, LINES_FILE, SETTLE_HOUR_UTC)
    from .odds_client import OddsClient
    from .store import Store
    import os
    key = os.environ["ODDS_API_KEY"]
    now = datetime.now(timezone.utc)
    do_settle = now.hour == SETTLE_HOUR_UTC
    print(f"[run] {now.isoformat()} do_settle={do_settle}")
    run_once(OddsClient(key), LEAGUES, Store(BETS_FILE, LINES_FILE),
             regions=REGIONS, min_edge=MIN_EDGE, odds_min=ODDS_MIN, odds_max=ODDS_MAX,
             now=now, do_settle=do_settle)

if __name__ == "__main__":
    main()
