# Deterministic Backtest Harness Implementation Plan (Plán A)

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Postavit plně funkční, deterministický value-betting backtester pro fotbalový trh 1X2 (data → pre-match features bez leaku → Shin devig → Elo anchor → value filtr → metriky CLV/ROI/kalibrace → baseliny → CLI report), bez jakéhokoli LLM.

**Architecture:** Čistá Python pipeline. Každý modul má jednu odpovědnost a je testovaný v izolaci. Walk-forward orchestrátor projde locked-test sezonu, pro každé kolo postaví features jen z dřívějších zápasů, anchor (Elo + na train fitnuté mapování) vyrobí P(H/D/A), value filtr vybere sázky proti otevíracímu kurzu, oracle spočítá CLV/ROI/kalibraci a uloží audit log. Výstupem je report anchor-only strategie + baseliny - přesně baseline, proti kterému bude Plán B (LLM korektor) měřen.

**Tech Stack:** Python 3.11+, pandas, numpy, scipy (root-finding pro Shin), scikit-learn (multinomiální mapování Ela), pyyaml, pytest. Data z `football-data.co.uk` (CSV).

**Spec:** `docs/superpowers/specs/2026-07-21-value-betting-predictor-design.md` (defaulty viz §10).

**Rozsah tohoto plánu:** deterministické jádro (§4.1, §4.2, §4.4, §4.5 části bez LLM, §6 baseliny/splity, §7 metriky). **Mimo tento plán (→ Plán B):** LLM korektor (§4.3), playbook, bloková reflexe, ablace, akceptační vyhodnocení.

---

## File Structure

```
value-betting-predictor/
  pyproject.toml
  config.yaml                     # už existuje v specu, sem se zkopíruje
  src/vbp/
    __init__.py
    config.py          # načtení + validace config.yaml -> dataclass
    data.py            # stažení + načtení football-data CSV přes whitelist sloupců
    anonymize.py       # mapování týmů -> Team_A/B, strip data/ligy
    features.py        # pre-match feature builder, tvrdá časová hranice
    devig.py           # Shin devig (+ power a proportional pro srovnání)
    anchor.py          # Elo rating + multinomiální mapování Δ -> P(H/D/A)
    value_filter.py    # výběr sázek (edge, rozsah kurzu, argmax, 1/zápas)
    metrics.py         # CLV, ROI, bootstrap CI, Brier, kalibrace, slippage, stratifikace
    baselines.py       # market / anchor-only / noise / always-favorite / strong model
    backtest.py        # deterministický walk-forward orchestrátor + audit log
    report.py          # textový/markdown souhrn běhu
    cli.py             # entrypoint: vbp-backtest
  tests/
    conftest.py
    fixtures/mini_league.csv       # malý ručně sestavený golden dataset
    test_config.py
    test_data.py
    test_anonymize.py
    test_features.py               # vč. leak testů
    test_devig.py
    test_anchor.py
    test_value_filter.py
    test_metrics.py
    test_baselines.py
    test_backtest.py
```

**Použij @superpowers:test-driven-development pro každý task.** Pořadí: napiš test → spusť (FAIL) → minimální implementace → spusť (PASS) → commit.

---

### Task 0: Project scaffold

**Files:**
- Create: `pyproject.toml`, `src/vbp/__init__.py`, `tests/conftest.py`

- [ ] **Step 1: Vytvoř `pyproject.toml`**

```toml
[project]
name = "vbp"
version = "0.1.0"
description = "Value-betting predictor - deterministic backtest harness"
requires-python = ">=3.11"
dependencies = [
  "pandas>=2.0",
  "numpy>=1.26",
  "scipy>=1.11",
  "scikit-learn>=1.4",
  "pyyaml>=6.0",
  "requests>=2.31",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
vbp-backtest = "vbp.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: Vytvoř `src/vbp/__init__.py`** (prázdný) a `tests/conftest.py`

```python
# tests/conftest.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
```

- [ ] **Step 3: Nainstaluj a ověř**

Run: `pip install -e ".[dev]"` a `pytest -q`
Expected: `no tests ran` (0 testů) bez import chyb.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml src/vbp/__init__.py tests/conftest.py
git commit -m "chore: project scaffold for backtest harness"
```

---

### Task 1: Config loader

Načte `config.yaml` do typované dataclass a zvaliduje klíčové invarianty (splity se nepřekrývají, prahy v rozsahu).

**Files:**
- Create: `src/vbp/config.py`, `tests/test_config.py`

- [ ] **Step 1: Napiš failing test**

```python
# tests/test_config.py
import pytest
from vbp.config import load_config, Config

def test_loads_defaults_from_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "league: E1\n"
        "seasons:\n  train: ['2122','2223']\n  validation: ['2324']\n  locked_test: ['2425']\n"
        "odds_source: pinnacle\ndevig: shin\n"
        "anchor: {type: elo_softmax_map, k: 20, home_adv: 70, start_rating: 1500}\n"
        "value: {min_edge: 0.03, odds_min: 1.6, odds_max: 4.5, skip_first_rounds: 4, stake: 1.0, max_bets_per_match: 1}\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert isinstance(cfg, Config)
    assert cfg.league == "E1"
    assert cfg.seasons.train == ["2122", "2223"]
    assert cfg.value.min_edge == 0.03
    assert cfg.value.odds_max == 4.5

def test_rejects_overlapping_splits(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "league: E1\n"
        "seasons:\n  train: ['2122','2324']\n  validation: ['2324']\n  locked_test: ['2425']\n"
        "odds_source: pinnacle\ndevig: shin\n"
        "anchor: {type: elo_softmax_map, k: 20, home_adv: 70, start_rating: 1500}\n"
        "value: {min_edge: 0.03, odds_min: 1.6, odds_max: 4.5, skip_first_rounds: 4, stake: 1.0, max_bets_per_match: 1}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overlap"):
        load_config(p)
```

- [ ] **Step 2: Spusť test → FAIL** (`ModuleNotFoundError: vbp.config`)

Run: `pytest tests/test_config.py -v`

- [ ] **Step 3: Implementace**

```python
# src/vbp/config.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Seasons:
    train: list[str]
    validation: list[str]
    locked_test: list[str]

@dataclass
class AnchorCfg:
    type: str = "elo_softmax_map"
    k: float = 20.0
    home_adv: float = 70.0
    start_rating: float = 1500.0

@dataclass
class ValueCfg:
    min_edge: float = 0.03
    odds_min: float = 1.6
    odds_max: float = 4.5
    skip_first_rounds: int = 4
    stake: float = 1.0
    max_bets_per_match: int = 1

@dataclass
class Config:
    league: str
    seasons: Seasons
    odds_source: str
    devig: str
    anchor: AnchorCfg
    value: ValueCfg

def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    seasons = Seasons(**raw["seasons"])
    all_splits = seasons.train + seasons.validation + seasons.locked_test
    if len(all_splits) != len(set(all_splits)):
        raise ValueError("season splits overlap - train/validation/locked_test must be disjoint")
    cfg = Config(
        league=raw["league"],
        seasons=seasons,
        odds_source=raw["odds_source"],
        devig=raw["devig"],
        anchor=AnchorCfg(**raw.get("anchor", {})),
        value=ValueCfg(**raw.get("value", {})),
    )
    if not (0 < cfg.value.min_edge < 0.5):
        raise ValueError("min_edge out of range")
    if cfg.value.odds_min >= cfg.value.odds_max:
        raise ValueError("odds_min must be < odds_max")
    return cfg
```

- [ ] **Step 4: Spusť test → PASS.** Run: `pytest tests/test_config.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/vbp/config.py tests/test_config.py
git commit -m "feat: config loader with split-overlap validation"
```

---

### Task 2: Data layer (whitelist sloupců)

Načte football-data CSV a **propustí jen whitelistované sloupce** (datum, týmy, výsledek, otevírací + zavírací 1X2 kurzy zvoleného zdroje). Nikdy pozápasové statistiky. Whitelist je klíčová anti-leak pojistka (§2).

**Files:**
- Create: `src/vbp/data.py`, `tests/test_data.py`, `tests/fixtures/mini_league.csv`

- [ ] **Step 1: Vytvoř fixture `tests/fixtures/mini_league.csv`**

Malý ručně sestavený dataset s pozápasovými sloupci navíc (HS, AS = střely), které MUSÍ být odfiltrovány. Sloupce: `Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,PSH,PSD,PSA,PSCH,PSCD,PSCA,HS,AS`.

```csv
Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,PSH,PSD,PSA,PSCH,PSCD,PSCA,HS,AS
09/08/2024,Alpha,Beta,2,1,H,1.90,3.50,4.20,1.85,3.60,4.40,12,7
09/08/2024,Gamma,Delta,0,0,D,2.60,3.20,2.80,2.55,3.25,2.90,9,10
16/08/2024,Beta,Gamma,1,2,A,2.10,3.30,3.60,2.20,3.30,3.40,8,11
16/08/2024,Delta,Alpha,1,1,D,3.80,3.60,2.00,3.90,3.55,1.98,6,14
```

- [ ] **Step 2: Napiš failing test**

```python
# tests/test_data.py
from pathlib import Path
import pandas as pd
from vbp.data import load_matches, WHITELIST_POSTFIX

FIX = Path(__file__).parent / "fixtures" / "mini_league.csv"

def test_loads_only_whitelisted_columns():
    df = load_matches(FIX, odds_source="pinnacle")
    # post-match shots must be gone
    assert "HS" not in df.columns and "AS" not in df.columns
    # required whitelisted columns present
    for col in ["Date", "HomeTeam", "AwayTeam", "FTR", "PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA"]:
        assert col in df.columns

def test_parses_dates_and_sorts_chronologically():
    df = load_matches(FIX, odds_source="pinnacle")
    assert str(df["Date"].dtype).startswith("datetime")
    assert df["Date"].is_monotonic_increasing

def test_result_label_is_hda():
    df = load_matches(FIX, odds_source="pinnacle")
    assert set(df["FTR"].unique()).issubset({"H", "D", "A"})
```

- [ ] **Step 3: Spusť → FAIL.** Run: `pytest tests/test_data.py -v`

- [ ] **Step 4: Implementace**

```python
# src/vbp/data.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

BASE_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
ODDS_PREFIX = {"pinnacle": "PS", "avg": "Avg", "bet365": "B365"}

def _odds_cols(source: str) -> list[str]:
    p = ODDS_PREFIX[source]
    # open = bez suffixu, close = suffix C
    return [f"{p}H", f"{p}D", f"{p}A", f"{p}CH", f"{p}CD", f"{p}CA"]

def load_matches(path: str | Path, odds_source: str = "pinnacle") -> pd.DataFrame:
    """Load football-data CSV keeping ONLY whitelisted pre-match columns.
    Anti-leak: post-match statistics (shots, corners, cards, half-time) are dropped."""
    whitelist = BASE_COLS + _odds_cols(odds_source)
    raw = pd.read_csv(path)
    missing = [c for c in whitelist if c not in raw.columns]
    if missing:
        raise ValueError(f"missing required columns for source {odds_source}: {missing}")
    df = raw[whitelist].copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="raise")
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    return df

WHITELIST_POSTFIX = ("H", "D", "A", "CH", "CD", "CA")  # exposed for tests/docs
```

Poznámka pro implementátora: reálná data se stahují ručně z `https://www.football-data.co.uk/englandm.php` (soubor `E1.csv` per sezona) do `data/raw/<season>_E1.csv`. Downloader je čistě pohodlí; testy jedou na fixture. Přidej `download_season(season, league, dest)` přes `requests` jako tenký helper BEZ testu proti síti.

- [ ] **Step 5: Spusť → PASS. Commit.**

```bash
git add src/vbp/data.py tests/test_data.py tests/fixtures/mini_league.csv
git commit -m "feat: data loader with column whitelist (anti-leak)"
```

---

### Task 3: Feature builder + LEAK testy

Ke každému zápasu postaví pre-match balíček **jen ze zápasů s dřívějším datem** (`date < target_date`; zápasy ze stejného dne se ignorují). Toto je nejdůležitější anti-leak invariant projektu - leak testy jsou povinné.

**Files:**
- Create: `src/vbp/features.py`, `tests/test_features.py`

- [ ] **Step 1: Napiš failing testy (vč. leak testu)**

```python
# tests/test_features.py
from pathlib import Path
import pandas as pd
from vbp.data import load_matches
from vbp.features import build_features, PreMatch

FIX = Path(__file__).parent / "fixtures" / "mini_league.csv"

def _df():
    return load_matches(FIX, odds_source="pinnacle")

def test_first_match_has_zero_history():
    df = _df()
    feats = build_features(df, form_n=5)
    first = feats.iloc[0]
    assert first["home_played"] == 0
    assert first["away_played"] == 0

def test_uses_only_strictly_earlier_matches():
    # For the round-2 match Beta vs Gamma (16/08), Beta played once (09/08 vs Alpha, lost),
    # Gamma played once (09/08 vs Delta, drew). Same-day round-1 matches count; later ones must not.
    df = _df()
    feats = build_features(df, form_n=5)
    row = feats[(feats.HomeTeam == "Beta") & (feats.AwayTeam == "Gamma")].iloc[0]
    assert row["home_played"] == 1
    assert row["away_played"] == 1
    assert row["home_pts"] == 0     # Beta lost round 1
    assert row["away_pts"] == 1     # Gamma drew round 1

def test_no_future_leak_invariant():
    """CI-critical: for every match, no feature may depend on a match with Date >= target Date."""
    df = _df()
    feats = build_features(df, form_n=5)
    # Corrupt the future: flip all FUTURE results and rebuild; earlier-round features must be identical.
    df2 = df.copy()
    df2.loc[df2["Date"] == df2["Date"].max(), "FTR"] = "A"
    feats2 = build_features(df2, form_n=5)
    early_mask = feats["Date"] < df["Date"].max()
    cols = ["home_played", "away_played", "home_pts", "away_pts",
            "home_gf_avg", "home_ga_avg", "away_gf_avg", "away_ga_avg"]
    pd.testing.assert_frame_equal(
        feats.loc[early_mask, cols].reset_index(drop=True),
        feats2.loc[early_mask, cols].reset_index(drop=True),
    )
```

- [ ] **Step 2: Spusť → FAIL.**

- [ ] **Step 3: Implementace**

```python
# src/vbp/features.py
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass
class PreMatch:
    """Column contract produced by build_features (documented for downstream modules)."""
    cols = [
        "home_played", "away_played", "home_pts", "away_pts",
        "home_gf_avg", "home_ga_avg", "away_gf_avg", "away_ga_avg",
        "home_rest_days", "away_rest_days",
    ]

def _result_points(row, team):
    if row["FTR"] == "D":
        return 1
    winner = row["HomeTeam"] if row["FTR"] == "H" else row["AwayTeam"]
    return 3 if winner == team else 0

def build_features(df: pd.DataFrame, form_n: int = 5) -> pd.DataFrame:
    """For each match, compute pre-match features from STRICTLY earlier matches only
    (Date < target Date). Same-day matches are excluded (no reliable kickoff times)."""
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    out_rows = []
    for i, row in df.iterrows():
        past = df[df["Date"] < row["Date"]]
        feat = {"Date": row["Date"], "HomeTeam": row["HomeTeam"], "AwayTeam": row["AwayTeam"]}
        for side, team in (("home", row["HomeTeam"]), ("away", row["AwayTeam"])):
            th = past[(past["HomeTeam"] == team) | (past["AwayTeam"] == team)]
            recent = th.tail(form_n)
            feat[f"{side}_played"] = len(th)
            feat[f"{side}_pts"] = sum(_result_points(r, team) for _, r in recent.iterrows())
            gf = [(r["FTHG"] if r["HomeTeam"] == team else r["FTAG"]) for _, r in recent.iterrows()]
            ga = [(r["FTAG"] if r["HomeTeam"] == team else r["FTHG"]) for _, r in recent.iterrows()]
            feat[f"{side}_gf_avg"] = (sum(gf) / len(gf)) if gf else 0.0
            feat[f"{side}_ga_avg"] = (sum(ga) / len(ga)) if ga else 0.0
            last_date = th["Date"].max() if len(th) else pd.NaT
            feat[f"{side}_rest_days"] = (row["Date"] - last_date).days if pd.notna(last_date) else -1
        out_rows.append(feat)
    return pd.DataFrame(out_rows)
```

- [ ] **Step 4: Spusť → PASS.** Zvlášť ověř `test_no_future_leak_invariant`.

- [ ] **Step 5: Commit**

```bash
git add src/vbp/features.py tests/test_features.py
git commit -m "feat: pre-match feature builder with future-leak invariant test"
```

---

### Task 4: Anonymizace

Nahradí jména týmů konzistentně `Team_<id>` a zajistí, že do promptu (pro Plán B) nepůjde kalendářní datum ani jméno ligy - jen relativní čas. V Plánu A anchor pracuje s reálnými týmy interně (Elo), anonymizace se aplikuje jen na výstupní balíček určený LLM. Implementujeme čistou funkci, ať je připravená pro Plán B a testovatelná už teď.

**Files:**
- Create: `src/vbp/anonymize.py`, `tests/test_anonymize.py`

- [ ] **Step 1: Napiš failing test**

```python
# tests/test_anonymize.py
from vbp.anonymize import anonymize_teams

def test_consistent_mapping_within_season():
    m = ["Alpha", "Beta", "Alpha", "Gamma"]
    ids, mapping = anonymize_teams(m)
    assert ids[0] == ids[2]            # Alpha -> same id both times
    assert ids[0] != ids[1]
    assert set(mapping.values()) == set(ids)
    assert all(v.startswith("Team_") for v in mapping.values())

def test_mapping_is_deterministic_given_order():
    a, _ = anonymize_teams(["X", "Y", "Z"])
    b, _ = anonymize_teams(["X", "Y", "Z"])
    assert a == b
```

- [ ] **Step 2: Spusť → FAIL. Step 3: Implementace**

```python
# src/vbp/anonymize.py
from __future__ import annotations

def anonymize_teams(team_sequence: list[str]) -> tuple[list[str], dict[str, str]]:
    """Map team names to Team_N ids, first-seen order = stable & deterministic.
    Returns (anonymized_sequence, {original: anon})."""
    mapping: dict[str, str] = {}
    out: list[str] = []
    for t in team_sequence:
        if t not in mapping:
            mapping[t] = f"Team_{len(mapping)}"
        out.append(mapping[t])
    return out, mapping
```

- [ ] **Step 4: Spusť → PASS. Step 5: Commit**

```bash
git add src/vbp/anonymize.py tests/test_anonymize.py
git commit -m "feat: deterministic team anonymization"
```

---

### Task 5: Devig (Shin) + srovnávací metody

Z desítkových kurzů spočítá férové pravděpodobnosti očištěné o marži. Default = **Shin** (řeší favorite-longshot bias líp než proporční normalizace). `power` a `proportional` jsou k dispozici pro sensitivity log.

**Files:**
- Create: `src/vbp/devig.py`, `tests/test_devig.py`

- [ ] **Step 1: Napiš failing testy**

```python
# tests/test_devig.py
import numpy as np
import pytest
from vbp.devig import devig, shin, proportional

def test_output_sums_to_one():
    p = devig([1.90, 3.50, 4.20], method="shin")
    assert abs(sum(p) - 1.0) < 1e-9

def test_fair_book_returns_input():
    # odds with zero margin: probabilities 0.5/0.3/0.2 -> odds 2.0/3.333/5.0
    odds = [2.0, 1/0.3, 5.0]
    p = devig(odds, method="shin")
    assert np.allclose(p, [0.5, 0.3, 0.2], atol=1e-3)

def test_preserves_ordering():
    p = devig([1.90, 3.50, 4.20], method="shin")
    assert p[0] > p[1] > p[2]

def test_shin_removes_more_margin_from_longshot_than_proportional():
    odds = [1.90, 3.50, 6.50]
    ps = shin(odds)
    pp = proportional(odds)
    # favorite-longshot bias: Shin assigns lower fair prob to the longshot than naive proportional
    assert ps[2] < pp[2]

def test_unknown_method_raises():
    with pytest.raises(ValueError):
        devig([2.0, 3.0, 4.0], method="nope")
```

- [ ] **Step 2: Spusť → FAIL. Step 3: Implementace**

```python
# src/vbp/devig.py
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq

def proportional(odds: list[float]) -> np.ndarray:
    r = 1.0 / np.asarray(odds, dtype=float)
    return r / r.sum()

def power(odds: list[float]) -> np.ndarray:
    """Fair probs p_i = (1/odds_i)**k, k chosen so sum(p)=1."""
    r = 1.0 / np.asarray(odds, dtype=float)
    f = lambda k: (r ** k).sum() - 1.0
    k = brentq(f, 0.5, 5.0)
    p = r ** k
    return p / p.sum()

def shin(odds: list[float]) -> np.ndarray:
    """Shin (1992) inversion: recover fair probabilities assuming a proportion z of
    insider money. Solve z so fair probs sum to 1."""
    r = 1.0 / np.asarray(odds, dtype=float)
    s = r.sum()
    def probs(z):
        return (np.sqrt(z * z + 4.0 * (1.0 - z) * r * r / s) - z) / (2.0 * (1.0 - z))
    if s <= 1.0 + 1e-12:          # no margin -> normalized implied
        return r / s
    z = brentq(lambda z: probs(z).sum() - 1.0, 1e-9, 0.5)
    p = probs(z)
    return p / p.sum()

_METHODS = {"shin": shin, "power": power, "proportional": proportional}

def devig(odds: list[float], method: str = "shin") -> np.ndarray:
    if method not in _METHODS:
        raise ValueError(f"unknown devig method: {method}")
    return _METHODS[method](odds)
```

- [ ] **Step 4: Spusť → PASS.** Pokud `test_fair_book_returns_input` selže kvůli numerice v Shin u nulové marže, ověř větev `s <= 1`.

- [ ] **Step 5: Commit**

```bash
git add src/vbp/devig.py tests/test_devig.py
git commit -m "feat: Shin/power/proportional devig with favorite-longshot test"
```

---

### Task 6: Elo anchor + mapování na P(H/D/A)

Sekvenční Elo (walk-forward, bez leaku z principu). Rozdíl ratingů `Δ = elo_H − elo_A + home_adv` se **na train** namapuje multinomiální logistickou regresí na (P_H, P_D, P_A). Mapování se fituje jen na train a zmrazí.

**Files:**
- Create: `src/vbp/anchor.py`, `tests/test_anchor.py`

- [ ] **Step 1: Napiš failing testy**

```python
# tests/test_anchor.py
import numpy as np
from vbp.anchor import EloAnchor

def _matches():
    # Home team clearly stronger; deterministic-ish results to move Elo
    return [
        {"HomeTeam": "A", "AwayTeam": "B", "FTR": "H"},
        {"HomeTeam": "B", "AwayTeam": "A", "FTR": "A"},
        {"HomeTeam": "A", "AwayTeam": "C", "FTR": "H"},
        {"HomeTeam": "C", "AwayTeam": "B", "FTR": "D"},
        {"HomeTeam": "B", "AwayTeam": "C", "FTR": "H"},
        {"HomeTeam": "C", "AwayTeam": "A", "FTR": "A"},
    ]

def test_elo_diff_grows_for_winner():
    anchor = EloAnchor(k=20, home_adv=70, start_rating=1500)
    diffs = anchor.run_and_collect(_matches())
    assert len(diffs) == len(_matches())
    # after processing, stronger team A should have higher rating than C
    assert anchor.rating("A") > anchor.rating("C")

def test_predict_proba_sums_to_one_and_is_calibrated_shape():
    anchor = EloAnchor(k=20, home_adv=70, start_rating=1500)
    diffs = anchor.run_and_collect(_matches())
    labels = [m["FTR"] for m in _matches()]
    anchor.fit_mapping(diffs, labels)
    p = anchor.predict_proba(delta=200.0)   # strong home edge
    assert abs(p["H"] + p["D"] + p["A"] - 1.0) < 1e-9
    assert p["H"] > p["A"]                   # positive delta -> home favored

def test_mapping_not_refit_on_test_ratings_still_advance():
    """Ratings update walk-forward on any stream, but the H/D/A mapping is frozen after fit."""
    anchor = EloAnchor(k=20, home_adv=70, start_rating=1500)
    diffs = anchor.run_and_collect(_matches())
    anchor.fit_mapping(diffs, [m["FTR"] for m in _matches()])
    coef_before = anchor._clf.coef_.copy()
    anchor.update(_matches()[0])             # advancing ratings must NOT touch the mapping
    assert np.array_equal(anchor._clf.coef_, coef_before)
```

- [ ] **Step 2: Spusť → FAIL. Step 3: Implementace**

```python
# src/vbp/anchor.py
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression

class EloAnchor:
    def __init__(self, k=20.0, home_adv=70.0, start_rating=1500.0):
        self.k = k
        self.home_adv = home_adv
        self.start = start_rating
        self._r: dict[str, float] = {}
        self._clf: LogisticRegression | None = None
        self._classes = ["H", "D", "A"]

    def rating(self, team: str) -> float:
        return self._r.get(team, self.start)

    def delta(self, home: str, away: str) -> float:
        return self.rating(home) - self.rating(away) + self.home_adv

    def update(self, match: dict) -> None:
        h, a, res = match["HomeTeam"], match["AwayTeam"], match["FTR"]
        exp_h = 1.0 / (1.0 + 10 ** (-(self.delta(h, a)) / 400.0))
        s_h = 1.0 if res == "H" else (0.5 if res == "D" else 0.0)
        change = self.k * (s_h - exp_h)
        self._r[h] = self.rating(h) + change
        self._r[a] = self.rating(a) - change

    def run_and_collect(self, matches: list[dict]) -> list[float]:
        """Walk-forward: record PRE-match delta, then update ratings. Returns deltas."""
        diffs = []
        for m in matches:
            diffs.append(self.delta(m["HomeTeam"], m["AwayTeam"]))
            self.update(m)
        return diffs

    def fit_mapping(self, deltas: list[float], labels: list[str]) -> None:
        X = np.asarray(deltas, dtype=float).reshape(-1, 1)
        y = np.asarray(labels)
        self._clf = LogisticRegression(max_iter=1000)
        self._clf.fit(X, y)

    def predict_proba(self, delta: float) -> dict[str, float]:
        if self._clf is None:
            raise RuntimeError("mapping not fit - call fit_mapping on train first")
        probs = self._clf.predict_proba([[delta]])[0]
        out = {c: float(probs[list(self._clf.classes_).index(c)]) for c in self._classes}
        tot = sum(out.values())
        return {k: v / tot for k, v in out.items()}
```

- [ ] **Step 4: Spusť → PASS.** Pokud sklearn u malého fixture nemá všechny 3 třídy, doplň fixture o alespoň jeden H, D i A (výše splněno).

- [ ] **Step 5: Commit**

```bash
git add src/vbp/anchor.py tests/test_anchor.py
git commit -m "feat: Elo anchor with train-only H/D/A mapping"
```

---

### Task 7: Value filter

Z pravděpodobností modelu a férového (devig) otevíracího kurzu vybere sázky: edge = `p_model − p_fair`, práh, rozsah kurzu, max 1 sázka/zápas (argmax edge).

**Files:**
- Create: `src/vbp/value_filter.py`, `tests/test_value_filter.py`

- [ ] **Step 1: Napiš failing testy**

```python
# tests/test_value_filter.py
from vbp.value_filter import select_bet

CFG = dict(min_edge=0.03, odds_min=1.6, odds_max=4.5)

def test_selects_outcome_with_edge_above_threshold():
    model = {"H": 0.55, "D": 0.25, "A": 0.20}
    fair  = {"H": 0.50, "D": 0.27, "A": 0.23}
    odds  = {"H": 1.90, "D": 3.50, "A": 4.20}
    bet = select_bet(model, fair, odds, **CFG)
    assert bet is not None and bet["outcome"] == "H"
    assert abs(bet["edge"] - 0.05) < 1e-9

def test_no_bet_when_below_threshold():
    model = {"H": 0.51, "D": 0.26, "A": 0.23}
    fair  = {"H": 0.50, "D": 0.27, "A": 0.23}
    odds  = {"H": 1.90, "D": 3.50, "A": 4.20}
    assert select_bet(model, fair, odds, **CFG) is None

def test_odds_out_of_range_excluded():
    model = {"H": 0.90, "D": 0.06, "A": 0.04}
    fair  = {"H": 0.80, "D": 0.12, "A": 0.08}
    odds  = {"H": 1.20, "D": 8.0, "A": 15.0}   # H below odds_min
    assert select_bet(model, fair, odds, **CFG) is None

def test_argmax_edge_picks_single_outcome():
    model = {"H": 0.40, "D": 0.35, "A": 0.25}
    fair  = {"H": 0.33, "D": 0.30, "A": 0.20}
    odds  = {"H": 2.60, "D": 3.10, "A": 3.60}
    bet = select_bet(model, fair, odds, **CFG)
    # edges: H .07, D .05, A .05 -> H wins
    assert bet["outcome"] == "H"
```

- [ ] **Step 2: Spusť → FAIL. Step 3: Implementace**

```python
# src/vbp/value_filter.py
from __future__ import annotations

def select_bet(model_p: dict, fair_p: dict, odds: dict,
               min_edge: float, odds_min: float, odds_max: float) -> dict | None:
    """Return the single best value bet for a match, or None. edge = model_p - fair_p."""
    candidates = []
    for o in ("H", "D", "A"):
        if not (odds_min <= odds[o] <= odds_max):
            continue
        edge = model_p[o] - fair_p[o]
        if edge >= min_edge:
            candidates.append({"outcome": o, "edge": edge, "odds": odds[o],
                               "model_p": model_p[o], "fair_p": fair_p[o]})
    if not candidates:
        return None
    return max(candidates, key=lambda c: c["edge"])
```

- [ ] **Step 4: Spusť → PASS. Step 5: Commit**

```bash
git add src/vbp/value_filter.py tests/test_value_filter.py
git commit -m "feat: value filter (edge threshold, odds range, argmax)"
```

---

### Task 8: Metriky (CLV, ROI, bootstrap CI, Brier, kalibrace, slippage, stratifikace)

**Fixní definice CLV** (zamčená): sázíme na otevírací kurz `o_open` na vybraný výsledek; férová zavírací pravděpodobnost toho výsledku `p_close_fair` (Shin na zavíracích kurzech). `CLV = p_close_fair * o_open - 1`. Kladné CLV = porazili jsme zavírací linii.

**Files:**
- Create: `src/vbp/metrics.py`, `tests/test_metrics.py`

- [ ] **Step 1: Napiš failing testy**

```python
# tests/test_metrics.py
import numpy as np
from vbp.metrics import clv, roi, bootstrap_roi_ci, brier, apply_slippage

def test_clv_positive_when_open_beats_close():
    # bet at 2.10, closing fair prob 0.52 -> CLV = 0.52*2.10 - 1 = 0.092
    assert abs(clv(o_open=2.10, p_close_fair=0.52) - 0.092) < 1e-9

def test_roi_flat_stake():
    bets = [{"won": True, "odds": 2.0}, {"won": False, "odds": 3.0}]
    # +1.0 and -1.0 -> ROI 0
    assert abs(roi(bets) - 0.0) < 1e-9

def test_roi_all_winners():
    bets = [{"won": True, "odds": 2.0}, {"won": True, "odds": 1.5}]
    # profit 1.0 + 0.5 = 1.5 over 2 stake -> 0.75
    assert abs(roi(bets) - 0.75) < 1e-9

def test_bootstrap_ci_brackets_point_estimate():
    bets = [{"won": True, "odds": 2.0}] * 60 + [{"won": False, "odds": 2.0}] * 40
    lo, hi = bootstrap_roi_ci(bets, n_boot=500, alpha=0.10, seed=42)
    assert lo < roi(bets) < hi

def test_brier_perfect_prediction_is_zero():
    preds = [{"H": 1.0, "D": 0.0, "A": 0.0}]
    outcomes = ["H"]
    assert brier(preds, outcomes) < 1e-12

def test_slippage_reduces_odds():
    assert abs(apply_slippage(2.00, 0.01) - 1.98) < 1e-9
```

- [ ] **Step 2: Spusť → FAIL. Step 3: Implementace**

```python
# src/vbp/metrics.py
from __future__ import annotations
import numpy as np

def clv(o_open: float, p_close_fair: float) -> float:
    """Closing Line Value: EV of a bet placed at opening odds under the closing fair line."""
    return p_close_fair * o_open - 1.0

def roi(bets: list[dict], stake: float = 1.0) -> float:
    if not bets:
        return 0.0
    profit = sum((b["odds"] - 1.0) * stake if b["won"] else -stake for b in bets)
    return profit / (len(bets) * stake)

def bootstrap_roi_ci(bets: list[dict], n_boot: int = 2000, alpha: float = 0.10, seed: int = 0):
    if not bets:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(bets))
    rois = []
    for _ in range(n_boot):
        sample = rng.choice(idx, size=len(bets), replace=True)
        rois.append(roi([bets[i] for i in sample]))
    lo = float(np.percentile(rois, 100 * alpha / 2))
    hi = float(np.percentile(rois, 100 * (1 - alpha / 2)))
    return lo, hi

def brier(preds: list[dict], outcomes: list[str]) -> float:
    """Multiclass Brier over ALL predictions (not just bets)."""
    total = 0.0
    for p, y in zip(preds, outcomes):
        total += sum((p[o] - (1.0 if o == y else 0.0)) ** 2 for o in ("H", "D", "A"))
    return total / len(preds)

def apply_slippage(odds: float, pct: float) -> float:
    """Worsen the price you actually get by pct (e.g. 0.01 -> 1% lower odds)."""
    return odds * (1.0 - pct)

def roi_after_slippage(bets: list[dict], pct: float, stake: float = 1.0) -> float:
    worsened = [{"won": b["won"], "odds": apply_slippage(b["odds"], pct)} for b in bets]
    return roi(worsened, stake)

def roi_by_outcome(bets: list[dict]) -> dict:
    out = {}
    for o in ("H", "D", "A"):
        sub = [b for b in bets if b["outcome"] == o]
        out[o] = roi(sub) if sub else None
    return out

def roi_drop_top(bets: list[dict], k: int = 3) -> float:
    """ROI after removing the k biggest winners (concentration check)."""
    profits = sorted(((b["odds"] - 1.0) if b["won"] else -1.0, i) for i, b in enumerate(bets))
    drop = set(i for _, i in profits[-k:])
    kept = [b for i, b in enumerate(bets) if i not in drop]
    return roi(kept)
```

- [ ] **Step 4: Spusť → PASS. Step 5: Commit**

```bash
git add src/vbp/metrics.py tests/test_metrics.py
git commit -m "feat: metrics - CLV, ROI, bootstrap CI, Brier, slippage, stratification"
```

---

### Task 9: Baseliny

Deterministické baseliny, proti kterým se poměřuje (§6): `market` (sázej podle nejvyšší tržní pravděpodobnosti), `always_favorite` (nejnižší kurz), `anchor_only` (Elo mapování + value filtr - to je hlavní strategie Plánu A), `noise` (tržní P + gaussovský šum → value filtr). Silný model odlišný od anchoru (Dixon-Coles) je volitelný stretch v Plánu A; pokud se nestihne, zapíše se do reportu jako „N/A (Plán B)".

**Files:**
- Create: `src/vbp/baselines.py`, `tests/test_baselines.py`

- [ ] **Step 1: Napiš failing testy**

```python
# tests/test_baselines.py
from vbp.baselines import always_favorite_pick, noise_probs

def test_always_favorite_picks_lowest_odds():
    odds = {"H": 1.80, "D": 3.40, "A": 4.50}
    assert always_favorite_pick(odds) == "H"

def test_noise_probs_sum_to_one_and_are_reproducible():
    fair = {"H": 0.5, "D": 0.3, "A": 0.2}
    a = noise_probs(fair, sigma=0.03, seed=1)
    b = noise_probs(fair, sigma=0.03, seed=1)
    assert a == b
    assert abs(sum(a.values()) - 1.0) < 1e-9
```

- [ ] **Step 2: Spusť → FAIL. Step 3: Implementace**

```python
# src/vbp/baselines.py
from __future__ import annotations
import numpy as np

def always_favorite_pick(odds: dict) -> str:
    return min(odds, key=lambda o: odds[o])

def market_pick(fair_p: dict) -> str:
    return max(fair_p, key=lambda o: fair_p[o])

def noise_probs(fair_p: dict, sigma: float = 0.03, seed: int = 0) -> dict:
    """Market fair probs + gaussian noise, clipped & renormalized. The null 'edge from noise' baseline."""
    rng = np.random.default_rng(seed)
    vals = {o: max(1e-3, fair_p[o] + rng.normal(0, sigma)) for o in ("H", "D", "A")}
    tot = sum(vals.values())
    return {o: vals[o] / tot for o in ("H", "D", "A")}
```

Poznámka: `noise` baseline v backtestu volá `noise_probs(fair, seed=match_index)` a pak stejný `select_bet` jako anchor - liší se jen zdroj `model_p`.

- [ ] **Step 4: Spusť → PASS. Step 5: Commit**

```bash
git add src/vbp/baselines.py tests/test_baselines.py
git commit -m "feat: deterministic baselines (favorite, market, noise)"
```

---

### Task 10: Backtest orchestrátor (walk-forward, bez LLM) + audit log

Spojí vše: natrénuje Elo přes train+validation, fitne mapování na train, pak projde locked-test sezonu kolo po kole. Pro každý zápas: features (jen z minulosti), Δ z Ela, anchor P, devig otevíracích kurzů, value filtr, uložení sázky + audit řádku. Elo se v testu dál dopředně updatuje (ne refit mapování). Vrací strukturu s audit logem a settled sázkami pro metriky.

**Files:**
- Create: `src/vbp/backtest.py`, `tests/test_backtest.py`

- [ ] **Step 1: Napiš failing test (integrace na fixture)**

```python
# tests/test_backtest.py
from pathlib import Path
from vbp.data import load_matches
from vbp.backtest import run_backtest

FIX = Path(__file__).parent / "fixtures" / "mini_league.csv"

def test_backtest_runs_and_produces_audit_rows():
    df = load_matches(FIX, odds_source="pinnacle")
    # tiny fixture: train on all-but-last, test last; skip_first_rounds=0 for the fixture
    result = run_backtest(
        train_df=df.iloc[:2], test_df=df.iloc[2:],
        odds_source="pinnacle", devig_method="shin",
        anchor_cfg=dict(k=20, home_adv=70, start_rating=1500),
        value_cfg=dict(min_edge=0.0, odds_min=1.0, odds_max=99.0),
        skip_first_rounds=0,
    )
    assert len(result["audit"]) == len(df.iloc[2:])          # one audit row per test match
    for row in result["audit"]:
        p = row["anchor_p"]
        assert abs(p["H"] + p["D"] + p["A"] - 1.0) < 1e-6    # valid probability
    # settled bets carry the fields metrics need
    for b in result["bets"]:
        assert {"outcome", "odds", "won", "clv"} <= set(b.keys())

def test_backtest_is_deterministic():
    df = load_matches(FIX, odds_source="pinnacle")
    kw = dict(train_df=df.iloc[:2], test_df=df.iloc[2:], odds_source="pinnacle",
              devig_method="shin", anchor_cfg=dict(k=20, home_adv=70, start_rating=1500),
              value_cfg=dict(min_edge=0.0, odds_min=1.0, odds_max=99.0), skip_first_rounds=0)
    assert run_backtest(**kw)["bets"] == run_backtest(**kw)["bets"]
```

- [ ] **Step 2: Spusť → FAIL. Step 3: Implementace**

```python
# src/vbp/backtest.py
from __future__ import annotations
import pandas as pd
from .anchor import EloAnchor
from .devig import devig
from .value_filter import select_bet
from .metrics import clv

def _odds_dicts(row, source):
    p = {"pinnacle": "PS", "avg": "Avg", "bet365": "B365"}[source]
    open_ = {"H": row[f"{p}H"], "D": row[f"{p}D"], "A": row[f"{p}A"]}
    close = {"H": row[f"{p}CH"], "D": row[f"{p}CD"], "A": row[f"{p}CA"]}
    return open_, close

def run_backtest(train_df, test_df, odds_source, devig_method,
                 anchor_cfg, value_cfg, skip_first_rounds):
    anchor = EloAnchor(**anchor_cfg)
    # 1) run Elo through train, collect deltas + labels, fit mapping on TRAIN only
    train_matches = train_df.to_dict("records")
    deltas = anchor.run_and_collect(train_matches)
    anchor.fit_mapping(deltas, [m["FTR"] for m in train_matches])

    audit, bets = [], []
    test_matches = test_df.reset_index(drop=True).to_dict("records")
    for i, m in enumerate(test_matches):
        delta = anchor.delta(m["HomeTeam"], m["AwayTeam"])
        p_model = anchor.predict_proba(delta)
        open_odds, close_odds = _odds_dicts(m, odds_source)
        fair_open = dict(zip(("H", "D", "A"), devig([open_odds["H"], open_odds["D"], open_odds["A"]], devig_method)))
        fair_close = dict(zip(("H", "D", "A"), devig([close_odds["H"], close_odds["D"], close_odds["A"]], devig_method)))
        bet = None
        if i >= skip_first_rounds:
            bet = select_bet(p_model, fair_open, open_odds, **value_cfg)
        row = {"i": i, "home": m["HomeTeam"], "away": m["AwayTeam"], "delta": delta,
               "anchor_p": p_model, "fair_open": fair_open, "result": m["FTR"], "bet": bet}
        audit.append(row)
        if bet is not None:
            o = bet["outcome"]
            bets.append({"outcome": o, "odds": open_odds[o], "won": (m["FTR"] == o),
                         "clv": clv(open_odds[o], fair_close[o]),
                         "model_p": p_model[o], "fair_p": fair_open[o]})
        anchor.update(m)   # walk-forward rating update, mapping stays frozen
    return {"audit": audit, "bets": bets}
```

- [ ] **Step 4: Spusť → PASS. Step 5: Commit**

```bash
git add src/vbp/backtest.py tests/test_backtest.py
git commit -m "feat: deterministic walk-forward backtest orchestrator + audit log"
```

---

### Task 11: CLI + report

Entrypoint `vbp-backtest --config config.yaml --data-dir data/raw`: načte data zvolené ligy/sezon, spustí backtest na locked-test, spočítá metriky (anchor strategie + baseliny) a vypíše markdown report + uloží audit log a bets do `runs/<timestamp>/`.

**Files:**
- Create: `src/vbp/report.py`, `src/vbp/cli.py`; Modify: (žádný)
- Test: `tests/test_report.py`

- [ ] **Step 1: Napiš failing test (report renderuje metriky)**

```python
# tests/test_report.py
from vbp.report import render_report

def test_report_contains_key_metrics():
    summary = {
        "n_bets": 120, "roi": 0.021, "roi_ci": (-0.03, 0.07),
        "mean_clv": 0.012, "brier": 0.62, "brier_market": 0.61,
        "roi_by_outcome": {"H": 0.03, "D": -0.01, "A": 0.02},
        "baselines": {"noise_roi": -0.04, "always_favorite_roi": -0.05},
    }
    md = render_report(summary)
    assert "ROI" in md and "CLV" in md and "120" in md
    assert "noise" in md.lower()
```

- [ ] **Step 2: Spusť → FAIL. Step 3: Implementace `report.py`**

```python
# src/vbp/report.py
from __future__ import annotations

def render_report(s: dict) -> str:
    lo, hi = s["roi_ci"]
    lines = [
        "# Backtest report (anchor-only, Plán A)",
        "",
        f"- Sázek: **{s['n_bets']}**",
        f"- ROI: **{s['roi']:.3f}** (90% CI [{lo:.3f}, {hi:.3f}])",
        f"- Průměrné CLV: **{s['mean_clv']:.4f}**",
        f"- Brier: **{s['brier']:.4f}** (trh {s['brier_market']:.4f})",
        "",
        "## ROI podle výsledku",
        *[f"- {k}: {('N/A' if v is None else f'{v:.3f}')}" for k, v in s["roi_by_outcome"].items()],
        "",
        "## Baseliny",
        f"- noise: {s['baselines']['noise_roi']:.3f}",
        f"- always_favorite: {s['baselines']['always_favorite_roi']:.3f}",
    ]
    return "\n".join(lines)
```

- [ ] **Step 4: Spusť → PASS.**

- [ ] **Step 5: Implementuj `cli.py`** (bez samostatného unit testu; ověří se ručním během)

```python
# src/vbp/cli.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from .config import load_config
from .data import load_matches
from .backtest import run_backtest
from .baselines import noise_probs, always_favorite_pick
from .value_filter import select_bet
from .devig import devig
from .metrics import (roi, bootstrap_roi_ci, brier, roi_by_outcome, roi_after_slippage)
from .report import render_report

def _load_split(data_dir, league, seasons, source):
    frames = [load_matches(Path(data_dir) / f"{s}_{league}.csv", source) for s in seasons]
    return pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)

def main(argv=None):
    ap = argparse.ArgumentParser(prog="vbp-backtest")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out-dir", default="runs")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    train = _load_split(args.data_dir, cfg.league, cfg.seasons.train + cfg.seasons.validation, cfg.odds_source)
    test = _load_split(args.data_dir, cfg.league, cfg.seasons.locked_test, cfg.odds_source)

    result = run_backtest(
        train_df=train, test_df=test, odds_source=cfg.odds_source, devig_method=cfg.devig,
        anchor_cfg=dict(k=cfg.anchor.k, home_adv=cfg.anchor.home_adv, start_rating=cfg.anchor.start_rating),
        value_cfg=dict(min_edge=cfg.value.min_edge, odds_min=cfg.value.odds_min, odds_max=cfg.value.odds_max),
        skip_first_rounds=cfg.value.skip_first_rounds,
    )
    bets = result["bets"]
    all_preds = [r["anchor_p"] for r in result["audit"]]
    all_out = [r["result"] for r in result["audit"]]
    market_preds = [r["fair_open"] for r in result["audit"]]

    summary = {
        "n_bets": len(bets),
        "roi": roi(bets),
        "roi_ci": bootstrap_roi_ci(bets, seed=0),
        "mean_clv": (sum(b["clv"] for b in bets) / len(bets)) if bets else 0.0,
        "brier": brier(all_preds, all_out),
        "brier_market": brier(market_preds, all_out),
        "roi_by_outcome": roi_by_outcome(bets),
        "roi_slippage_1pct": roi_after_slippage(bets, 0.01),
        "baselines": _baselines(result["audit"], cfg),
    }
    out = Path(args.out_dir) / pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(render_report(summary), encoding="utf-8")
    (out / "audit.json").write_text(json.dumps(result["audit"], default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    print(render_report(summary))
    print(f"\nSaved to {out}")

def _baselines(audit, cfg):
    noise_bets, fav_bets = [], []
    vc = dict(min_edge=cfg.value.min_edge, odds_min=cfg.value.odds_min, odds_max=cfg.value.odds_max)
    for r in audit:
        # noise baseline reuses fair_open as market prob source
        np_ = noise_probs(r["fair_open"], seed=r["i"])
        nb = select_bet(np_, r["fair_open"], _reconstruct_odds(r), **vc)
        if nb:
            o = nb["outcome"]
            noise_bets.append({"outcome": o, "odds": _reconstruct_odds(r)[o], "won": r["result"] == o})
    return {
        "noise_roi": roi(noise_bets),
        "always_favorite_roi": _favorite_roi(audit),
    }

def _reconstruct_odds(r):
    # fair_open -> approximate odds back is lossy; store raw odds in audit instead (see note)
    raise NotImplementedError

if __name__ == "__main__":
    main()
```

> **Implementační poznámka (důležitá):** `_reconstruct_odds` je záměrně nedokončená - ukazuje chybu v návrhu audit řádku. Uprav `backtest.run_backtest`, aby do každého audit řádku uložil i **raw `open_odds` a `close_odds`** (ne jen `fair_open`). Pak `_baselines` čte skutečné kurzy přímo. Přidej k Tasku 10 test, že audit řádek obsahuje `open_odds`. Toto je jediné místo, kde plán vědomě nechává implementátora dorovnat kontrakt - udělej to jako první krok Tasku 11.

- [ ] **Step 6: Ruční ověření (potřebuje stažená data)**

Stáhni `E1` sezony do `data/raw/<season>_E1.csv`, zkopíruj `config.yaml` do rootu, spusť:

Run: `vbp-backtest --config config.yaml --data-dir data/raw`
Expected: vytiskne report s `n_bets`, ROI + CI, CLV, Brier vs trh, baseliny; uloží `runs/<ts>/report.md` + `audit.json`.

- [ ] **Step 7: Commit**

```bash
git add src/vbp/report.py src/vbp/cli.py tests/test_report.py
git commit -m "feat: CLI entrypoint + markdown report for anchor backtest"
```

---

## Definition of Done (Plán A)

- [ ] `pytest -q` zelené (config, data+whitelist, features+leak invariant, anonymize, devig, anchor, value_filter, metrics, baselines, backtest, report).
- [ ] `vbp-backtest` proběhne na reálných `E1` datech a vytiskne report anchor strategie + baseliny + CLV + Brier vs trh.
- [ ] Leak invariant test (`test_no_future_leak_invariant`) a whitelist test prochází - anti-leak pojistky jsou v CI.
- [ ] Audit log ukládá per-zápas raw open/close kurzy, anchor P, fair P, sázku, výsledek, CLV → připravený vstup pro Plán B.

## Navazuje (Plán B, samostatný plán)

LLM korektor nad anchor P (§4.3), strukturovaný playbook + lifecycle, bloková reflexe, orchestrátor učení, ablace (empty/frozen/static/no-reflection) a vyhodnocení akceptačních kritérií. Plán B se napíše, až Plán A poběží - reálné výstupy anchoru upřesní formát promptu a rozsah korekcí.
```
