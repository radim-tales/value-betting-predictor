from __future__ import annotations
import requests

BASE = "https://api.the-odds-api.com/v4"

class OddsClient:
    def __init__(self, key: str):
        self.key = key

    def _get(self, path, **params):
        r = requests.get(f"{BASE}{path}", params={"apiKey": self.key, **params}, timeout=30)
        r.raise_for_status()
        quota = {"remaining": r.headers.get("x-requests-remaining"),
                 "used": r.headers.get("x-requests-used")}
        return r.json(), quota

    def fetch_odds(self, sport, regions="eu,uk"):
        return self._get(f"/sports/{sport}/odds", regions=regions, markets="h2h", oddsFormat="decimal")

    def fetch_scores(self, sport, days_from=1):
        return self._get(f"/sports/{sport}/scores", daysFrom=days_from)

    def list_sports(self):
        return self._get("/sports")
