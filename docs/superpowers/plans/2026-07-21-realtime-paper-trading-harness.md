# Real-time Paper-Trading Harness Implementation Plan (v1)

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Postavit headless validační harness, který přes The Odds API sbírá živé fotbalové kurzy, loguje virtuální (paper) line-shopping value sázky proti Pinnacle a měří dopředu CLV rozdělené podle typu knihy a ligy. Žádné reálné sázky, žádný bankroll.

**Architecture:** Stateless Python moduly pod `src/vbp/live/`, znovupoužívají `vbp.devig` a `vbp.metrics`. Jeden běh (`vbp.live.run`) udělá poll → snapshot open/close → settle a zapíše JSON logy do `live_state/`. GitHub Actions cron ho spouští 3×/den a commitne změněné logy zpět do repa. `report` je samostatný (i lokální) skript, který joinem `bets.jsonl × lines.json` spočítá mean CLV s bootstrap CI × typ knihy × liga. Testy jedou na uložených fixture JSON odpovědích - žádná síť ani API kredit.

**Tech Stack:** Python 3.11+, `requests`, `numpy`, `scipy` (přes `vbp.devig`). The Odds API (free tier). GitHub Actions cron. Reuse: `vbp.devig`, `vbp.metrics` (clv, roi, bootstrap_roi_ci) + nový `bootstrap_mean_ci`.

**Spec:** `docs/superpowers/specs/2026-07-21-realtime-paper-trading-harness-design.md`.
**Seed:** `realtime_poc.py` (ověřený PoC v rootu repa) - z něj vychází `adapter` + `value`.

---

## File Structure

```
value-betting-predictor/
  src/vbp/
    metrics.py               # MODIFY: přidat bootstrap_mean_ci(values, ...)
    live/
      __init__.py
      config.py              # LEAGUES [(sport_key, tier)], value prahy, cesty
      odds_client.py         # fetch_odds / fetch_scores / list_sports (+ kredit z hlaviček)
      adapter.py             # event_to_books + classify_book (sharp/soft/exchange)
      value.py               # find_value(books, ...) reuse vbp.devig
      store.py               # bets.jsonl + lines.json I/O; add_bet(dedup), update_line, set_result
      settle.py              # settle_finished(lines, scores) -> result+settled do lines
      report.py              # join bets×lines -> per-bet won/clv/roi -> agregace CLV bootstrap CI
      run.py                 # orchestrace jednoho běhu (injektovatelný klient)
  tests/live/
    __init__.py
    fixtures/odds_sample.json    # uložená odds odpověď (2-3 zápasy, vč. pinnacle + soft + exchange)
    fixtures/scores_sample.json  # uložená scores odpověď (1 doběhlý zápas)
    test_adapter.py
    test_value.py
    test_store.py
    test_settle.py
    test_report.py
    test_run.py
  live_state/                    # persistované logy (NENÍ v .gitignore) - vytvoří se za běhu
  .github/workflows/live-harness.yml   # cron
```

**Použij @superpowers:test-driven-development.** Síťové funkce (`odds_client`) se netestují proti síti; testy krmí uložené fixture JSON přímo do adapter/settle/report a `run` bere injektovatelný fake klient.

---

### Task 0: `bootstrap_mean_ci` do vbp.metrics

Generický bootstrap CI přes seznam hodnot (pro CLV; `bootstrap_roi_ci` bootstrapuje ROI, na CLV nestačí - viz spec §8).

**Files:** Modify `src/vbp/metrics.py`; Test `tests/test_metrics.py` (přidat)

- [ ] **Step 1: Failing test** (přidej do `tests/test_metrics.py`)

```python
def test_bootstrap_mean_ci_brackets_mean():
    import numpy as np
    from vbp.metrics import bootstrap_mean_ci
    vals = [0.02]*60 + [-0.05]*40      # mean = -0.008
    lo, hi = bootstrap_mean_ci(vals, n_boot=500, alpha=0.10, seed=42)
    assert lo < float(np.mean(vals)) < hi

def test_bootstrap_mean_ci_empty_is_zero():
    from vbp.metrics import bootstrap_mean_ci
    assert bootstrap_mean_ci([], n_boot=100) == (0.0, 0.0)
```

- [ ] **Step 2: Run → FAIL.** Run: `.venv/Scripts/python.exe -m pytest tests/test_metrics.py -k bootstrap_mean_ci -v`

- [ ] **Step 3: Implement** (přidej do `src/vbp/metrics.py`)

```python
def bootstrap_mean_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.10, seed: int = 0):
    """Percentile bootstrap CI for the mean of a value list (e.g. per-bet CLV)."""
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))
```

- [ ] **Step 4: Run → PASS.** Full suite green (`pytest -q`).

- [ ] **Step 5: Commit** `feat: bootstrap_mean_ci for CLV confidence intervals`

---

### Task 1: `live` package + config

**Files:** Create `src/vbp/live/__init__.py` (prázdný), `src/vbp/live/config.py`, `tests/live/__init__.py` (prázdný)

- [ ] **Step 1: Implement `config.py`** (žádný test - jen konstanty)

```python
# src/vbp/live/config.py
from __future__ import annotations
from pathlib import Path

# 1 likvidní + 1 zanedbaná liga (finalizovat dle aktivních + Pinnacle pokrytí, spec §10)
LEAGUES = [
    ("soccer_brazil_campeonato", "liquid"),
    ("soccer_sweden_superettan", "neglected"),
]
REGIONS = "eu,uk"
MIN_EDGE = 0.03
ODDS_MIN = 1.6
ODDS_MAX = 8.0
CLOSE_BUFFER_MIN = 0            # aktualizuj pin_close dokud now < kickoff

STATE_DIR = Path("live_state")  # NENÍ v .gitignore, commituje se
BETS_FILE = STATE_DIR / "bets.jsonl"
LINES_FILE = STATE_DIR / "lines.json"
```

- [ ] **Step 2: Ensure `live_state/` isn't gitignored.** Zkontroluj `.gitignore` - nesmí obsahovat `live_state/`. Pokud existuje širší pravidlo, přidej výjimku. (Plan A gitignoroval `runs/` a `data/`, ne `live_state/`, takže OK; jen ověř.)

- [ ] **Step 3: Commit** `chore: vbp.live package + config`

---

### Task 2: Adapter (event → books + klasifikace knihy)

Transformace The Odds API eventu na `{book: {H,D,A}}` a klasifikace typu knihy.

**Files:** Create `src/vbp/live/adapter.py`, `tests/live/fixtures/odds_sample.json`, `tests/live/test_adapter.py`

- [ ] **Step 1: Fixture `odds_sample.json`** - malá reálná struktura The Odds API (2 zápasy). Musí obsahovat: pinnacle, jednu soft knihu (williamhill), jednu burzu (betfair_ex_eu nebo matchbook), a jeden event bez pinnacle (test „no truth").

```json
[
  {"id": "m1", "sport_key": "soccer_brazil_campeonato", "commence_time": "2026-08-01T20:00:00Z",
   "home_team": "Alpha FC", "away_team": "Beta FC",
   "bookmakers": [
     {"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
        {"name": "Alpha FC", "price": 1.90}, {"name": "Beta FC", "price": 4.20}, {"name": "Draw", "price": 3.50}]}]},
     {"key": "williamhill", "markets": [{"key": "h2h", "outcomes": [
        {"name": "Alpha FC", "price": 1.85}, {"name": "Beta FC", "price": 4.60}, {"name": "Draw", "price": 3.40}]}]},
     {"key": "betfair_ex_eu", "markets": [{"key": "h2h", "outcomes": [
        {"name": "Alpha FC", "price": 1.95}, {"name": "Beta FC", "price": 4.80}, {"name": "Draw", "price": 3.55}]}]}
   ]},
  {"id": "m2", "sport_key": "soccer_brazil_campeonato", "commence_time": "2026-08-01T22:00:00Z",
   "home_team": "Gamma FC", "away_team": "Delta FC",
   "bookmakers": [
     {"key": "williamhill", "markets": [{"key": "h2h", "outcomes": [
        {"name": "Gamma FC", "price": 2.10}, {"name": "Delta FC", "price": 3.60}, {"name": "Draw", "price": 3.30}]}]}
   ]}
]
```

- [ ] **Step 2: Failing tests**

```python
# tests/live/test_adapter.py
import json
from pathlib import Path
from vbp.live.adapter import event_to_books, classify_book

FIX = json.loads((Path(__file__).parent / "fixtures" / "odds_sample.json").read_text(encoding="utf-8"))

def test_event_to_books_maps_hda():
    books = event_to_books(FIX[0])
    assert set(books) == {"pinnacle", "williamhill", "betfair_ex_eu"}
    assert books["pinnacle"] == {"H": 1.90, "D": 3.50, "A": 4.20}

def test_event_without_all_outcomes_skipped_book():
    # m2 has only williamhill with all 3 -> fine; a book missing Draw would be dropped
    books = event_to_books(FIX[1])
    assert "williamhill" in books

def test_classify_book():
    assert classify_book("pinnacle") == "sharp"
    assert classify_book("betfair_ex_eu") == "exchange"
    assert classify_book("matchbook") == "exchange"
    assert classify_book("williamhill") == "soft"
    assert classify_book("some_unknown_book") == "soft"   # default
```

- [ ] **Step 3: Run → FAIL. Step 4: Implement**

```python
# src/vbp/live/adapter.py
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
```

- [ ] **Step 5: Run → PASS. Commit** `feat: live adapter (event->books + book classification)`

---

### Task 3: Value detection

Z books najdi value: `devig(Pinnacle)` = pravda, nejlepší cena napříč knihami, EV ≥ práh. Vrací kandidáty s tagem knihy.

**Files:** Create `src/vbp/live/value.py`, `tests/live/test_value.py`

- [ ] **Step 1: Failing tests**

```python
# tests/live/test_value.py
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
```

- [ ] **Step 2: Run → FAIL. Step 3: Implement**

```python
# src/vbp/live/value.py
from __future__ import annotations
from vbp.devig import devig
from .adapter import classify_book

def find_value(books: dict, min_edge: float = 0.03, odds_min: float = 1.6,
               odds_max: float = 8.0, truth: str = "pinnacle") -> list[dict]:
    if truth not in books:
        return []
    t = books[truth]
    fair = dict(zip("HDA", devig([t["H"], t["D"], t["A"]], "shin")))  # TRUTH
    out = []
    for o in "HDA":
        price, book = max((books[b][o], b) for b in books)   # best price across books
        edge = fair[o] * price - 1
        if edge >= min_edge and odds_min <= price <= odds_max:
            out.append({"outcome": o, "price": price, "book": book,
                        "book_type": classify_book(book), "edge": edge, "pin_fair": fair[o]})
    return out
```

- [ ] **Step 4: Run → PASS. Commit** `feat: live value detection (Pinnacle truth, best price, book-tagged)`

---

### Task 4: Store (bets.jsonl + lines.json)

Persistence: append value sázek s dedup, čtení; per-zápas lines s open/close/result.

**Files:** Create `src/vbp/live/store.py`, `tests/live/test_store.py`

- [ ] **Step 1: Failing tests**

```python
# tests/live/test_store.py
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
```

- [ ] **Step 2: Run → FAIL. Step 3: Implement**

```python
# src/vbp/live/store.py
from __future__ import annotations
import json
from pathlib import Path

class Store:
    def __init__(self, bets_file, lines_file):
        self.bets_file = Path(bets_file)
        self.lines_file = Path(lines_file)
        self.bets_file.parent.mkdir(parents=True, exist_ok=True)

    def load_bets(self) -> list[dict]:
        if not self.bets_file.exists():
            return []
        return [json.loads(l) for l in self.bets_file.read_text(encoding="utf-8").splitlines() if l.strip()]

    def _bet_keys(self) -> set:
        return {(b["match_id"], b["outcome"], b["book"]) for b in self.load_bets()}

    def add_bet(self, bet: dict) -> bool:
        """Append if (match_id,outcome,book) not seen. Returns True if added."""
        if (bet["match_id"], bet["outcome"], bet["book"]) in self._bet_keys():
            return False
        with self.bets_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(bet, ensure_ascii=False) + "\n")
        return True

    def load_lines(self) -> dict:
        if not self.lines_file.exists():
            return {}
        return json.loads(self.lines_file.read_text(encoding="utf-8"))

    def _save_lines(self, lines: dict):
        self.lines_file.write_text(json.dumps(lines, ensure_ascii=False, indent=1), encoding="utf-8")

    def update_line(self, match_id: str, meta: dict, pin_fair: dict):
        lines = self.load_lines()
        ln = lines.get(match_id, {**meta, "pin_open": None, "pin_close": None,
                                   "result": None, "settled": False})
        if ln["pin_open"] is None:
            ln["pin_open"] = pin_fair
        ln["pin_close"] = pin_fair                       # last one before kickoff wins
        lines[match_id] = ln
        self._save_lines(lines)

    def set_result(self, match_id: str, result: str):
        lines = self.load_lines()
        if match_id in lines:
            lines[match_id]["result"] = result
            lines[match_id]["settled"] = True
            self._save_lines(lines)
```

- [ ] **Step 4: Run → PASS. Commit** `feat: live store (dedup bets.jsonl + open/close/result lines.json)`

---

### Task 5: Settle (výsledky → lines.json)

Ze scores odpovědi zapíše result do lines. Nepočítá per-sázka metriky.

**Files:** Create `src/vbp/live/settle.py`, `tests/live/fixtures/scores_sample.json`, `tests/live/test_settle.py`

- [ ] **Step 1: Fixture `scores_sample.json`** - The Odds API scores tvar (1 doběhlý zápas m1, Alpha vyhrál):

```json
[
  {"id": "m1", "completed": true, "home_team": "Alpha FC", "away_team": "Beta FC",
   "scores": [{"name": "Alpha FC", "score": "2"}, {"name": "Beta FC", "score": "1"}]},
  {"id": "m2", "completed": false, "home_team": "Gamma FC", "away_team": "Delta FC", "scores": null}
]
```

- [ ] **Step 2: Failing tests**

```python
# tests/live/test_settle.py
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
```

- [ ] **Step 3: Run → FAIL. Step 4: Implement**

```python
# src/vbp/live/settle.py
from __future__ import annotations

def result_from_scores(ev: dict) -> str | None:
    if not ev.get("completed") or not ev.get("scores"):
        return None
    by = {s["name"]: int(s["score"]) for s in ev["scores"]}
    h = by.get(ev["home_team"]); a = by.get(ev["away_team"])
    if h is None or a is None:
        return None
    return "H" if h > a else ("A" if a > h else "D")

def settle_finished(store, scores: list[dict]) -> int:
    """Write result+settled to lines for finished, not-yet-settled matches. Returns #settled."""
    lines = store.load_lines()
    n = 0
    for ev in scores:
        mid = ev.get("id")
        if mid in lines and not lines[mid]["settled"]:
            r = result_from_scores(ev)
            if r:
                store.set_result(mid, r)
                n += 1
    return n
```

- [ ] **Step 5: Run → PASS. Commit** `feat: live settle (scores -> result in lines.json)`

---

### Task 6: Report (join → CLV bootstrap CI × typ knihy × liga)

**Files:** Create `src/vbp/live/report.py`, `tests/live/test_report.py`

- [ ] **Step 1: Failing tests**

```python
# tests/live/test_report.py
from vbp.live.store import Store
from vbp.live.report import summarize

def _seed(tmp_path):
    s = Store(tmp_path / "bets.jsonl", tmp_path / "lines.json")
    meta = {"league":"BR","league_tier":"liquid","home":"A","away":"B","kickoff":"t"}
    s.update_line("m1", meta, {"H":0.50,"D":0.30,"A":0.20})   # open
    s.update_line("m1", {**meta}, {"H":0.48,"D":0.30,"A":0.22})  # close (A more likely)
    s.set_result("m1", "A")
    s.add_bet({"match_id":"m1","outcome":"A","book":"williamhill","price":5.00,
               "book_type":"soft","edge":0.10,"league":"BR","league_tier":"liquid"})
    return s

def test_summarize_computes_clv_and_groups(tmp_path):
    s = _seed(tmp_path)
    rep = summarize(s)
    # bet on A at 5.00, pin_close_fair A = 0.22 -> CLV = 0.22*5 - 1 = 0.10, won (result A)
    grp = rep["by"][("soft","liquid")]
    assert abs(grp["mean_clv"] - 0.10) < 1e-9
    assert grp["n"] == 1 and grp["wins"] == 1
    assert "clv_ci" in grp and "roi" in grp

def test_summarize_ignores_unsettled(tmp_path):
    s = Store(tmp_path / "bets.jsonl", tmp_path / "lines.json")
    s.update_line("m9", {"league":"BR","league_tier":"liquid","home":"A","away":"B","kickoff":"t"}, {"H":.5,"D":.3,"A":.2})
    s.add_bet({"match_id":"m9","outcome":"H","book":"bwin","price":2.0,"book_type":"soft","edge":0.05,"league":"BR","league_tier":"liquid"})
    rep = summarize(s)
    assert rep["settled_bets"] == 0
```

- [ ] **Step 2: Run → FAIL. Step 3: Implement**

```python
# src/vbp/live/report.py
from __future__ import annotations
from collections import defaultdict
from vbp.metrics import clv, roi, bootstrap_roi_ci, bootstrap_mean_ci

def _settled_bet_rows(store):
    lines = store.load_lines()
    rows = []
    for b in store.load_bets():
        ln = lines.get(b["match_id"])
        if not ln or not ln.get("settled") or not ln.get("pin_close"):
            continue
        o = b["outcome"]
        rows.append({**b,
                     "won": ln["result"] == o,
                     "clv": clv(b["price"], ln["pin_close"][o])})   # p_close_fair * price - 1
    return rows

def summarize(store) -> dict:
    rows = _settled_bet_rows(store)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["book_type"], r["league_tier"])].append(r)
    by = {}
    for key, rs in groups.items():
        clvs = [r["clv"] for r in rs]
        bets = [{"won": r["won"], "odds": r["price"]} for r in rs]   # shape-adapt for vbp.metrics
        by[key] = {"n": len(rs), "wins": sum(r["won"] for r in rs),
                   "mean_clv": sum(clvs) / len(clvs), "clv_ci": bootstrap_mean_ci(clvs, seed=0),
                   "roi": roi(bets), "roi_ci": bootstrap_roi_ci(bets, seed=0)}
    return {"settled_bets": len(rows), "total_bets": len(store.load_bets()), "by": by}

def render(rep: dict) -> str:
    lines = [f"# Live harness report  (settled {rep['settled_bets']}/{rep['total_bets']} bets)", ""]
    for (bt, tier), g in sorted(rep["by"].items()):
        lo, hi = g["clv_ci"]
        verdict = "EDGE" if (bt == "soft" and lo > 0) else ""
        lines.append(f"- {bt:<8} {tier:<9} n={g['n']:>3} wins={g['wins']:>3} "
                     f"CLV={g['mean_clv']:+.4f} [{lo:+.4f},{hi:+.4f}] ROI={g['roi']:+.3f} {verdict}")
    return "\n".join(lines)

if __name__ == "__main__":
    from .config import BETS_FILE, LINES_FILE
    from .store import Store
    print(render(summarize(Store(BETS_FILE, LINES_FILE))))
```

- [ ] **Step 4: Run → PASS. Commit** `feat: live report (join, per-bet CLV, bootstrap CI by book-type x league)`

---

### Task 7: Odds client (tenký API klient)

**Files:** Create `src/vbp/live/odds_client.py` (bez síťového testu)

- [ ] **Step 1: Implement** (síťové funkce - netestují se proti síti; ověří se v Tasku 8 dry-run)

```python
# src/vbp/live/odds_client.py
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
```

- [ ] **Step 2: Import sanity** `.venv/Scripts/python.exe -c "from vbp.live.odds_client import OddsClient; print('ok')"`

- [ ] **Step 3: Commit** `feat: thin The Odds API client`

---

### Task 8: Run orchestrátor (injektovatelný klient)

Jeden běh: pro každou ligu poll → per event: pokud má Pinnacle, update_line + add_bet value; pak scores → settle. Klient je injektovatelný (fake v testech).

**Files:** Create `src/vbp/live/run.py`, `tests/live/test_run.py`

- [ ] **Step 1: Failing test (fake klient, žádná síť)**

```python
# tests/live/test_run.py
import json
from pathlib import Path
from vbp.live.store import Store
from vbp.live.run import run_once

FIXDIR = Path(__file__).parent / "fixtures"
ODDS = json.loads((FIXDIR / "odds_sample.json").read_text(encoding="utf-8"))
SCORES = json.loads((FIXDIR / "scores_sample.json").read_text(encoding="utf-8"))

class FakeClient:
    def fetch_odds(self, sport, regions="eu,uk"): return ODDS, {"remaining":"499"}
    def fetch_scores(self, sport, days_from=1): return SCORES, {"remaining":"498"}

def test_run_once_logs_value_and_settles(tmp_path):
    s = Store(tmp_path / "bets.jsonl", tmp_path / "lines.json")
    run_once(FakeClient(), [("soccer_brazil_campeonato","liquid")], s,
             regions="eu", min_edge=0.0, odds_min=1.0, odds_max=99.0)
    bets = s.load_bets(); lines = s.load_lines()
    assert any(b["match_id"] == "m1" for b in bets)       # m1 has pinnacle -> value logged
    assert all(b["match_id"] != "m2" for b in bets)       # m2 no pinnacle -> nothing
    assert lines["m1"]["pin_open"] is not None            # line snapshotted
    assert lines["m1"]["settled"] is True                 # m1 finished in scores -> settled
    for b in bets:
        assert "book_type" in b and "league_tier" in b
```

- [ ] **Step 2: Run → FAIL. Step 3: Implement**

```python
# src/vbp/live/run.py
from __future__ import annotations
from vbp.devig import devig
from .adapter import event_to_books
from .value import find_value
from .settle import settle_finished

def _pin_fair(books):
    t = books["pinnacle"]
    return dict(zip("HDA", devig([t["H"], t["D"], t["A"]], "shin")))

def run_once(client, leagues, store, regions="eu,uk",
             min_edge=0.03, odds_min=1.6, odds_max=8.0):
    for sport, tier in leagues:
        events, quota = client.fetch_odds(sport, regions=regions)
        for ev in events:
            books = event_to_books(ev)
            if "pinnacle" not in books:
                continue
            meta = {"league": sport, "league_tier": tier, "home": ev["home_team"],
                    "away": ev["away_team"], "kickoff": ev["commence_time"]}
            store.update_line(ev["id"], meta, _pin_fair(books))
            for c in find_value(books, min_edge, odds_min, odds_max):
                store.add_bet({"match_id": ev["id"], "league": sport, "league_tier": tier,
                               "home": ev["home_team"], "away": ev["away_team"],
                               "kickoff": ev["commence_time"], **c})
        scores, _ = client.fetch_scores(sport)
        settle_finished(store, scores)

def main():
    from .config import LEAGUES, REGIONS, MIN_EDGE, ODDS_MIN, ODDS_MAX, BETS_FILE, LINES_FILE
    from .odds_client import OddsClient
    from .store import Store
    import os
    key = os.environ["ODDS_API_KEY"]
    run_once(OddsClient(key), LEAGUES, Store(BETS_FILE, LINES_FILE),
             regions=REGIONS, min_edge=MIN_EDGE, odds_min=ODDS_MIN, odds_max=ODDS_MAX)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run → PASS. Run FULL suite. Commit** `feat: live run orchestrator (poll -> value -> snapshot -> settle)`

---

### Task 9: GitHub Actions cron workflow

**Files:** Create `.github/workflows/live-harness.yml`

- [ ] **Step 1: Implement workflow**

```yaml
name: live-harness
on:
  schedule:
    - cron: "0 8 * * *"
    - cron: "0 14 * * *"
    - cron: "0 20 * * *"
  workflow_dispatch: {}
permissions:
  contents: write
jobs:
  poll:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - name: Install minimal deps
        run: pip install numpy scipy requests && pip install -e . --no-deps
      - name: Run harness
        env:
          ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
        run: python -m vbp.live.run
      - name: Commit state
        run: |
          git config user.name "live-harness"
          git config user.email "bot@users.noreply.github.com"
          git add live_state/
          git diff --cached --quiet || git commit -m "live: update state $(date -u +%FT%TZ)"
          git push
```

- [ ] **Step 2: Ověření nastavení (ruční, mimo CI)**
  - V GitHubu přidat repo secret `ODDS_API_KEY`.
  - Ověřit, že repo má remote (Plán A/B byl lokální bez remotu - **workflow potřebuje GitHub remote**; pokud není, push na GitHub napřed).
  - Finalizovat `LEAGUES` v `config.py` dle aktuálně aktivních lig (spec §10) - ověřit přes `list_sports` a rychlý `fetch_odds`, že mají Pinnacle pokrytí.

- [ ] **Step 3: Dry-run lokálně** (potřebuje ODDS_API_KEY): `ODDS_API_KEY=xxx .venv/Scripts/python.exe -m vbp.live.run` → zkontroluj, že `live_state/bets.jsonl` + `lines.json` vzniknou a `python -m vbp.live.report` něco vypíše. Nekomitovat live_state z dry-runu do plánu-branch (uklidit).

- [ ] **Step 4: Commit** `feat: GitHub Actions cron for live harness`

---

## Definition of Done (v1)

- [ ] `pytest -q` zelené vč. nových `tests/live/*` a `bootstrap_mean_ci`; **žádný test nevolá síť** (vše přes fixture JSON + FakeClient).
- [ ] `python -m vbp.live.run` s reálným klíčem naplní `live_state/` (ověřeno dry-runem).
- [ ] `python -m vbp.live.report` vypíše mean CLV + bootstrap CI × typ knihy × liga.
- [ ] GitHub Actions workflow existuje, `ODDS_API_KEY` secret nastaven, repo má GitHub remote, `live_state/` NENÍ gitignorováno.
- [ ] Anti-leak/poctivost: report obsahuje caveat řádek (close = proxy; paper ignoruje slippage+bany).

## Navazuje (mimo v1)

Po týdnech běhu: vyhodnotit report. Když CLV soft-bookmakerů má CI nad nulou → edge potvrzen dopředu → teprve pak zvažovat v2 (bankroll/Kelly doporučování, víc lig, placený tier, přesnější close, případně scraping/LLM vrstva pro zanedbané ligy).
