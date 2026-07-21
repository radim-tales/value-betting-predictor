# LLM Corrector Learning Layer Implementation Plan (Plán B)

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Postavit učící se LLM vrstvu nad hotový deterministický harness (Plán A): Claude korektor upravuje anchor pravděpodobnosti podle sebe-psaného strukturovaného playbooku, playbook se blokově přepisuje reflexí, a celé se změří proti anchor-only baselinu + ablacím + akceptačním kritériím.

**Architecture:** LLM volání jsou schovaná za injektovatelné rozhraní (`LLMClient`), takže testy jedou na `FakeLLM` s naskriptovanými JSON odpověďmi a **žádná síť/token se v testech neutratí**. Reálný `AnthropicClient` je tenký a ověřuje se manuálně/živě. Deterministické části (aplikace korekcí, playbook lifecycle, agregace bloku, ablace, akceptace) jsou plně TDD testované. Orchestrátor učení znovupoužívá `EloAnchor`, `devig`, `select_bit`, `metrics` a `build_features` z Plánu A. Každé LLM volání se loguje kompletně (prompt hash + raw odpověď) a běh je **přehratelný z uložených odpovědí** (Claude není bitově deterministický).

**Tech Stack:** Python 3.11+, `anthropic` SDK (Claude), pydantic (structured output validace), + vše z Plánu A (pandas, numpy, scipy, scikit-learn). Modely: korekce = **Haiku 4.5** (`claude-haiku-4-5`, přijímá temperature, structured output), reflexe = **Sonnet 5** (`claude-sonnet-5`, adaptivní thinking, **temperature NELZE** - 400, řídí se promptem/effort).

**Spec:** `docs/superpowers/specs/2026-07-21-value-betting-predictor-design.md` (§4.3 korektor, §4.6 loop, §6 ablace/akceptace, §10 zamčené defaulty).
**Baseline k poražení (běh 21.7.2026):** anchor-only na locked-test 24/25 = 342 sázek, ROI +1,6 % (CI [-11,4;+14,3]), **CLV -6,05 %**, Brier 0,625 (trh 0,619), noise +6,6 %.

**Předpoklady:** Plán A je hotový a zmergovaný (anchor, devig, value_filter, metrics, features, backtest, cli). `ANTHROPIC_API_KEY` v prostředí (nebo `ant auth login`) pro živý běh; testy ho nepotřebují.

---

## Důležité API poznámky (z claude-api skillu - implementátor je MUSÍ dodržet)

- **Haiku 4.5 korekce:** `client.messages.parse(model="claude-haiku-4-5", max_tokens=..., output_format=CorrectionBatch, messages=[...])` s pydantic modelem. `temperature=0.0` je OK. **NEposílat `effort`** (Haiku 4.5 ho odmítá). **NEposílat `thinking`** (netřeba).
- **Sonnet 5 reflexe:** `client.messages.create(model="claude-sonnet-5", max_tokens=..., output_config={"effort":"medium"}, thinking={"type":"adaptive"}, messages=[...])`. **NEposílat `temperature`/`top_p`/`top_k`** (400). Variabilitu neřešit teplotou, ale promptem.
- **Structured output:** `messages.parse()` vrací `.parsed_output` (validovaná pydantic instance). Když schéma nesedí → SDK/retry; my navíc validujeme rozsahy a v krajním případě zápas přeskočíme (spec §10).
- **Reprodukovatelnost:** logovat `model`, `prompt` (nebo jeho sha256), `response._request_id`, raw `.to_dict()`. Přehrání = číst z logu, ne volat znovu.
- **Prompt caching:** playbook + anchor prefix drž na začátku promptu, `cache_control` na posledním stabilním bloku (levnější opakované korekce). Volitelné, ne blokující.

---

## File Structure

```
src/vbp/
  llm/
    __init__.py
    client.py        # LLMClient protokol + AnthropicClient (reálný) + FakeLLM (testy)
    schemas.py       # pydantic: CorrectionBatch, Correction
  prompt.py          # build_correction_prompt(), build_reflection_prompt()
  corrections.py     # apply_corrections(): validace -> zero-sum -> clip -> renorm -> skip
  playbook.py        # Playbook: parse/serialize, sekce, rule lifecycle, délkový strop
  block_report.py    # aggregate_block(): metriky + kalibrační summary bloku pro reflexi
  learn.py           # run_learning(): LLM-in-the-loop walk-forward + bloková reflexe + audit
  ablations.py       # run_ablations(): learned/empty/frozen/static/no-reflection + noise/anchor
  acceptance.py      # evaluate_acceptance(): locked-test výsledky vs zamčená kritéria
  learn_cli.py       # vbp-learn entrypoint
tests/
  test_llm_schemas.py
  test_corrections.py
  test_playbook.py
  test_block_report.py
  test_learn.py           # s FakeLLM
  test_ablations.py       # s FakeLLM
  test_acceptance.py
  fixtures/seed_playbook.md
```

**Použij @superpowers:test-driven-development.** LLM je vždy `FakeLLM` v testech.

---

### Task 0: LLM schémata (pydantic) + package

**Files:** Create `src/vbp/llm/__init__.py`, `src/vbp/llm/schemas.py`, `tests/test_llm_schemas.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_llm_schemas.py
import pytest
from vbp.llm.schemas import Correction, CorrectionBatch

def test_correction_batch_parses():
    b = CorrectionBatch(corrections=[
        Correction(match_id="0", dH=0.03, dD=-0.01, dA=-0.02, rationale="home form"),
    ])
    assert b.corrections[0].match_id == "0"
    assert b.corrections[0].dH == 0.03

def test_delta_hard_cap_enforced():
    with pytest.raises(Exception):
        Correction(match_id="1", dH=0.5, dD=-0.25, dA=-0.25)  # |dH| > 0.15 hard cap
```

- [ ] **Step 2: Run → FAIL. Step 3: Implement**

```python
# src/vbp/llm/schemas.py
from __future__ import annotations
from pydantic import BaseModel, Field

class Correction(BaseModel):
    match_id: str
    dH: float = Field(ge=-0.15, le=0.15)   # hard cap per spec §10
    dD: float = Field(ge=-0.15, le=0.15)
    dA: float = Field(ge=-0.15, le=0.15)
    rationale: str = Field(default="", max_length=200)

class CorrectionBatch(BaseModel):
    corrections: list[Correction]
```

`src/vbp/llm/__init__.py` is empty.

- [ ] **Step 4: Run → PASS. Step 5: Commit** `feat: LLM correction pydantic schemas`

---

### Task 1: Apply corrections (deterministic)

Aplikuje delty na anchor P s pravidly ze spec §10: validace rozsahu (řeší schéma) → **zero-sum** kontrola → `P = anchor + delta` → clip `[0.01, 0.98]` → renormalizace. Když raw součet delt mimo toleranci nebo P silně mimo → **skip** (vrať anchor beze změny + flag).

**Files:** Create `src/vbp/corrections.py`, `tests/test_corrections.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_corrections.py
from vbp.corrections import apply_correction

ANCHOR = {"H": 0.50, "D": 0.30, "A": 0.20}

def test_applies_zero_sum_delta():
    p, skipped = apply_correction(ANCHOR, {"dH": 0.03, "dD": -0.01, "dA": -0.02})
    assert not skipped
    assert abs(sum(p.values()) - 1.0) < 1e-9
    assert p["H"] > ANCHOR["H"]

def test_skips_when_delta_not_zero_sum():
    # deltas sum to +0.10 -> outside tolerance -> skip, return anchor
    p, skipped = apply_correction(ANCHOR, {"dH": 0.10, "dD": 0.0, "dA": 0.0}, zero_sum_tol=0.02)
    assert skipped
    assert p == ANCHOR

def test_clips_and_renormalizes():
    # push A negative -> clipped to 0.01 then renormalized
    p, skipped = apply_correction({"H": 0.60, "D": 0.35, "A": 0.05},
                                  {"dH": 0.0, "dD": 0.10, "dA": -0.10})
    assert not skipped
    assert p["A"] >= 0.01
    assert abs(sum(p.values()) - 1.0) < 1e-9
```

- [ ] **Step 2: Run → FAIL. Step 3: Implement**

```python
# src/vbp/corrections.py
from __future__ import annotations

def apply_correction(anchor_p: dict, delta: dict,
                     zero_sum_tol: float = 0.02,
                     clip_lo: float = 0.01, clip_hi: float = 0.98) -> tuple[dict, bool]:
    """Apply LLM delta to anchor probs. Returns (probs, skipped).
    Skip (return anchor unchanged) if deltas are not ~zero-sum (a broken/incoherent
    correction) - per spec §10 we do NOT silently normalize large errors."""
    d = {o: float(delta.get(f"d{o}", 0.0)) for o in ("H", "D", "A")}
    if abs(sum(d.values())) > zero_sum_tol:
        return dict(anchor_p), True
    raw = {o: anchor_p[o] + d[o] for o in ("H", "D", "A")}
    clipped = {o: min(clip_hi, max(clip_lo, raw[o])) for o in ("H", "D", "A")}
    tot = sum(clipped.values())
    return {o: clipped[o] / tot for o in ("H", "D", "A")}, False
```

- [ ] **Step 4: Run → PASS. Step 5: Commit** `feat: deterministic correction application (zero-sum, clip, renorm, skip)`

---

### Task 2: Playbook (strukturovaný, lifecycle)

Playbook = markdown se sekcemi Priors / Rules (s počítadlem support-oppose) / Hypotheses (TTL) / Banned / Notes. Parse ↔ serialize round-trip, délkový strop (znaky), max počet pravidel. Lifecycle je řízený **kódem** (ne LLM): reflexe vrací navržený nový playbook text; kód ho zvaliduje (ořízne na max_rules/max_chars, zahodí prošlé hypotézy). Poznámka: reflexe primárně přepisuje text; počítadla support/oppose a promote/drop udržuje kód podle výsledků (Task 5 je propojí).

**Files:** Create `src/vbp/playbook.py`, `tests/test_playbook.py`, `tests/fixtures/seed_playbook.md`

- [ ] **Step 1: Fixture `seed_playbook.md`**

```markdown
## Priors
- Domácí výhoda je reálná, ale trh ji už započítává.

## Rules

## Hypotheses

## Banned

## Notes
```

- [ ] **Step 2: Failing tests**

```python
# tests/test_playbook.py
from pathlib import Path
from vbp.playbook import Playbook

SEED = (Path(__file__).parent / "fixtures" / "seed_playbook.md").read_text(encoding="utf-8")

def test_roundtrip_parse_serialize():
    pb = Playbook.parse(SEED)
    assert "Priors" in pb.sections
    out = pb.serialize()
    assert Playbook.parse(out).sections["Priors"] == pb.sections["Priors"]

def test_enforce_max_chars_trims_notes_first():
    pb = Playbook.parse(SEED)
    pb.sections["Notes"] = ["x" * 5000]
    pb.sections["Rules"] = ["important rule"]
    pb.enforce_limits(max_chars=200, max_rules=12)
    out = pb.serialize()
    assert len(out) <= 400            # notes trimmed; rules kept
    assert "important rule" in out

def test_enforce_max_rules_keeps_first_n():
    pb = Playbook.parse(SEED)
    pb.sections["Rules"] = [f"rule {i}" for i in range(20)]
    pb.enforce_limits(max_chars=10000, max_rules=12)
    assert len(pb.sections["Rules"]) == 12
```

- [ ] **Step 3: Run → FAIL. Step 4: Implement**

```python
# src/vbp/playbook.py
from __future__ import annotations
from dataclasses import dataclass, field

SECTION_ORDER = ["Priors", "Rules", "Hypotheses", "Banned", "Notes"]

@dataclass
class Playbook:
    sections: dict[str, list[str]] = field(default_factory=lambda: {s: [] for s in SECTION_ORDER})

    @classmethod
    def parse(cls, text: str) -> "Playbook":
        sections = {s: [] for s in SECTION_ORDER}
        current = None
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                name = stripped[3:].strip()
                current = name if name in sections else None
            elif stripped.startswith("- ") and current:
                sections[current].append(stripped[2:].strip())
        return cls(sections=sections)

    def serialize(self) -> str:
        parts = []
        for s in SECTION_ORDER:
            parts.append(f"## {s}")
            parts.extend(f"- {item}" for item in self.sections.get(s, []))
            parts.append("")
        return "\n".join(parts).rstrip() + "\n"

    def enforce_limits(self, max_chars: int, max_rules: int) -> None:
        """Trim to fit: cap Rules count, then drop Notes/Banned/Hypotheses until under max_chars."""
        self.sections["Rules"] = self.sections["Rules"][:max_rules]
        trim_order = ["Notes", "Banned", "Hypotheses"]
        while len(self.serialize()) > max_chars:
            for s in trim_order:
                if self.sections[s]:
                    self.sections[s].pop()
                    break
            else:
                # nothing left to trim in soft sections; trim rules as last resort
                if self.sections["Rules"]:
                    self.sections["Rules"].pop()
                else:
                    break
```

- [ ] **Step 5: Run → PASS. Step 6: Commit** `feat: structured playbook parse/serialize + limit enforcement`

---

### Task 3: LLM client (protokol + reálný + fake)

`LLMClient` protokol s `correct()` a `reflect()`. `FakeLLM` vrací naskriptované odpovědi (pro testy). `AnthropicClient` volá Claude přesně podle claude-api skillu. **Testy jedou jen na FakeLLM.**

**Files:** Create `src/vbp/llm/client.py`, `tests/test_llm_client.py`

- [ ] **Step 1: Failing test (jen FakeLLM)**

```python
# tests/test_llm_client.py
from vbp.llm.client import FakeLLM
from vbp.llm.schemas import CorrectionBatch, Correction

def test_fake_llm_returns_scripted_corrections():
    fake = FakeLLM(corrections=[CorrectionBatch(corrections=[
        Correction(match_id="0", dH=0.02, dD=-0.01, dA=-0.01)])],
        reflections=["## Priors\n- updated\n"])
    batch = fake.correct(prompt="ignored")
    assert isinstance(batch, CorrectionBatch)
    assert batch.corrections[0].match_id == "0"
    assert fake.reflect(prompt="ignored") == "## Priors\n- updated\n"

def test_fake_llm_logs_calls():
    fake = FakeLLM(corrections=[CorrectionBatch(corrections=[])], reflections=["x"])
    fake.correct(prompt="P1")
    fake.reflect(prompt="P2")
    assert [c["kind"] for c in fake.calls] == ["correct", "reflect"]
```

- [ ] **Step 2: Run → FAIL. Step 3: Implement**

```python
# src/vbp/llm/client.py
from __future__ import annotations
from typing import Protocol
from .schemas import CorrectionBatch

class LLMClient(Protocol):
    def correct(self, prompt: str) -> CorrectionBatch: ...
    def reflect(self, prompt: str) -> str: ...

class FakeLLM:
    """Deterministic stand-in for tests. Returns scripted outputs in order."""
    def __init__(self, corrections: list[CorrectionBatch], reflections: list[str]):
        self._corrections = list(corrections)
        self._reflections = list(reflections)
        self.calls: list[dict] = []
    def correct(self, prompt: str) -> CorrectionBatch:
        self.calls.append({"kind": "correct", "prompt": prompt})
        return self._corrections.pop(0) if self._corrections else CorrectionBatch(corrections=[])
    def reflect(self, prompt: str) -> str:
        self.calls.append({"kind": "reflect", "prompt": prompt})
        return self._reflections.pop(0) if self._reflections else ""

class AnthropicClient:
    """Real Claude client. NOT exercised by the test suite (needs ANTHROPIC_API_KEY).
    Correction = Haiku 4.5 (accepts temperature, structured output, NO effort/thinking).
    Reflection = Sonnet 5 (adaptive thinking, NO temperature - it 400s)."""
    def __init__(self, correct_model="claude-haiku-4-5", reflect_model="claude-sonnet-5",
                 temp_correct=0.0, log=None):
        import anthropic
        self._client = anthropic.Anthropic()
        self.correct_model = correct_model
        self.reflect_model = reflect_model
        self.temp_correct = temp_correct
        self.log = log  # callable(dict) for audit, optional

    def correct(self, prompt: str) -> CorrectionBatch:
        resp = self._client.messages.parse(
            model=self.correct_model, max_tokens=4000,
            temperature=self.temp_correct,          # OK on Haiku 4.5
            output_format=CorrectionBatch,
            messages=[{"role": "user", "content": prompt}],
        )
        if self.log:
            self.log({"kind": "correct", "model": self.correct_model,
                      "request_id": resp._request_id, "raw": resp.to_dict()})
        return resp.parsed_output

    def reflect(self, prompt: str) -> str:
        resp = self._client.messages.create(
            model=self.reflect_model, max_tokens=4000,
            output_config={"effort": "medium"},     # NO temperature on Sonnet 5
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
        if self.log:
            self.log({"kind": "reflect", "model": self.reflect_model,
                      "request_id": resp._request_id, "raw": resp.to_dict()})
        return text
```

- [ ] **Step 4: Run → PASS (2 tests, FakeLLM only). Step 5: Commit** `feat: LLMClient protocol + FakeLLM + AnthropicClient (Haiku correct / Sonnet reflect)`

---

### Task 4: Prompt builders

Sestaví prompt pro korekci (anonymizovaný pre-match balíček + anchor P + playbook; **žádný kurz**) a pro reflexi (agregovaný block report + kalibrační summary + aktuální playbook, s instrukcí přepsat playbook a uvést metrický důvod u změn).

**Files:** Create `src/vbp/prompt.py`, `tests/test_prompt.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_prompt.py
from vbp.prompt import build_correction_prompt, build_reflection_prompt

def test_correction_prompt_has_no_odds_and_is_anonymized():
    matches = [{"match_id": "0", "home": "Team_0", "away": "Team_1",
                "anchor_p": {"H": 0.5, "D": 0.3, "A": 0.2},
                "features": {"home_pts": 6, "away_pts": 3}}]
    p = build_correction_prompt(matches, playbook_text="## Priors\n- x\n")
    assert "Team_0" in p and "Team_1" in p
    assert "odds" not in p.lower() and "kurz" not in p.lower()   # anti-leak: no market
    assert "## Priors" in p
    assert "match_id" in p                                       # instructs JSON keying

def test_reflection_prompt_has_metrics_not_stories():
    report = {"n_bets": 20, "roi": -0.05, "clv": -0.04, "brier": 0.63,
              "brier_by_outcome": {"H": 0.2, "D": 0.3, "A": 0.4},
              "overconfidence": 0.08}
    p = build_reflection_prompt(report, playbook_text="## Priors\n- x\n")
    assert "brier" in p.lower() or "kalibr" in p.lower()
    assert "0.63" in p or "-0.05" in p
    assert "## Priors" in p
```

- [ ] **Step 2: Run → FAIL. Step 3: Implement** (`build_correction_prompt`, `build_reflection_prompt` - čisté string builders; playbook prefix na začátku kvůli cachingu; explicitně instruovat "vrať JSON korekce, delty ~ sečteny na 0, cap ±0.10", resp. "přepiš playbook, u každé změny metrický důvod"). Plné znění promptů dolaď v implementaci; testy zamykají invarianty (bez kurzu, anonymizace, metriky ne příběhy).

- [ ] **Step 4: Run → PASS. Step 5: Commit** `feat: correction + reflection prompt builders (odds-free, calibration-fed)`

---

### Task 5: Block report (agregace pro reflexi)

Z bloku vyhodnocených kol spočítá agregát pro reflexi: n sázek, ROI, CLV, Brier (celkově + po H/D/A), overconfidence (průměr `model_p - realized` na vsazených), kolik korekcí bylo skipnuto. **Žádné příběhy jednotlivých zápasů.**

**Files:** Create `src/vbp/block_report.py`, `tests/test_block_report.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_block_report.py
from vbp.block_report import aggregate_block

def test_aggregate_block_computes_core_metrics():
    preds = [{"H": 0.5, "D": 0.3, "A": 0.2}, {"H": 0.4, "D": 0.3, "A": 0.3}]
    outcomes = ["H", "A"]
    bets = [{"outcome": "H", "odds": 2.0, "won": True, "clv": 0.02, "model_p": 0.5}]
    rep = aggregate_block(preds, outcomes, bets, n_skipped=1)
    assert rep["n_bets"] == 1
    assert abs(rep["roi"] - 1.0) < 1e-9        # single winning bet at 2.0
    assert "brier" in rep and "clv" in rep
    assert rep["n_skipped"] == 1

def test_empty_block_is_safe():
    rep = aggregate_block([], [], [], n_skipped=0)
    assert rep["n_bets"] == 0
```

- [ ] **Step 2: Run → FAIL. Step 3: Implement** (reuse `metrics.roi/brier`; overconfidence = mean(model_p − won) na sázkách; guard prázdný blok). 

- [ ] **Step 4: Run → PASS. Step 5: Commit** `feat: block aggregation for reflection (calibration summary)`

---

### Task 6: Learning orchestrator (LLM-in-the-loop, bloková reflexe)

Walk-forward jako Plán A, ale mezi anchor a value filtr vstoupí LLM korektor; po každém **bloku** (config `block_every_rounds` / `block_min_bets`) proběhne reflexe a playbook se přepíše. Znovupoužívá anchor/devig/value_filter/metrics. Přijímá `LLMClient` (v testech `FakeLLM`). Vrací audit + settled sázky + finální playbook + snapshoty playbooku po blocích.

**Files:** Create `src/vbp/learn.py`, `tests/test_learn.py`

- [ ] **Step 1: Failing test (FakeLLM, na fixture)**

```python
# tests/test_learn.py
from pathlib import Path
from vbp.data import load_matches
from vbp.llm.client import FakeLLM
from vbp.llm.schemas import CorrectionBatch, Correction
from vbp.learn import run_learning

FIX = Path(__file__).parent / "fixtures" / "mini_league.csv"

def _fake(n):
    # one empty correction batch per round, a couple reflections
    return FakeLLM(corrections=[CorrectionBatch(corrections=[]) for _ in range(n)],
                   reflections=["## Priors\n- learned something\n" for _ in range(n)])

def test_learning_runs_and_updates_playbook():
    df = load_matches(FIX, odds_source="pinnacle")
    result = run_learning(
        train_df=df.iloc[:2], test_df=df.iloc[2:], warmup_df=None,
        llm=_fake(10), seed_playbook="## Priors\n- start\n",
        odds_source="pinnacle", devig_method="shin",
        anchor_cfg=dict(k=20, home_adv=70, start_rating=1500),
        value_cfg=dict(min_edge=0.0, odds_min=1.0, odds_max=99.0),
        skip_first_rounds=0, block_every_rounds=1, block_min_bets=0,
        playbook_limits=dict(max_chars=10000, max_rules=12),
    )
    assert len(result["audit"]) == len(df.iloc[2:])
    for r in result["audit"]:
        p = r["corrected_p"]
        assert abs(p["H"] + p["D"] + p["A"] - 1.0) < 1e-6
    # empty corrections => corrected_p equals anchor_p
    assert result["audit"][0]["corrected_p"] == result["audit"][0]["anchor_p"]
    assert "final_playbook" in result and result["final_playbook"]

def test_learning_deterministic_given_fake():
    df = load_matches(FIX, odds_source="pinnacle")
    kw = dict(train_df=df.iloc[:2], test_df=df.iloc[2:], warmup_df=None,
              seed_playbook="## Priors\n- s\n", odds_source="pinnacle", devig_method="shin",
              anchor_cfg=dict(k=20, home_adv=70, start_rating=1500),
              value_cfg=dict(min_edge=0.0, odds_min=1.0, odds_max=99.0),
              skip_first_rounds=0, block_every_rounds=1, block_min_bets=0,
              playbook_limits=dict(max_chars=10000, max_rules=12))
    a = run_learning(llm=_fake(10), **kw)["bets"]
    b = run_learning(llm=_fake(10), **kw)["bets"]
    assert a == b
```

- [ ] **Step 2: Run → FAIL. Step 3: Implement** `run_learning(...)`. Skeleton:

```python
# src/vbp/learn.py  (structure — full impl in this task)
from __future__ import annotations
from .anchor import EloAnchor
from .devig import devig
from .value_filter import select_bet
from .metrics import clv
from .features import build_features
from .anonymize import anonymize_teams
from .playbook import Playbook
from .corrections import apply_correction
from .prompt import build_correction_prompt, build_reflection_prompt
from .block_report import aggregate_block

def run_learning(train_df, test_df, warmup_df, llm, seed_playbook,
                 odds_source, devig_method, anchor_cfg, value_cfg,
                 skip_first_rounds, block_every_rounds, block_min_bets, playbook_limits):
    anchor = EloAnchor(**anchor_cfg)
    tm = train_df.to_dict("records")
    anchor.fit_mapping(anchor.run_and_collect(tm), [m["FTR"] for m in tm])
    if warmup_df is not None:
        for m in warmup_df.to_dict("records"):
            anchor.update(m)

    playbook = Playbook.parse(seed_playbook)
    audit, bets, snapshots = [], [], []
    block_preds, block_out, block_bets, block_skipped = [], [], [], 0
    rounds_since_block = 0

    # group test matches into rounds by date (a "round" = matches sharing a date)
    test = test_df.reset_index(drop=True)
    for round_idx, (date, group) in enumerate(test.groupby("Date", sort=True)):
        # build anonymized packet for the round (features from strictly-earlier matches)
        # NOTE: features computed on the full history-to-date; odds NEVER in the packet
        round_matches = group.to_dict("records")
        packet = []
        for j, m in enumerate(round_matches):
            delta = anchor.delta(m["HomeTeam"], m["AwayTeam"])
            anchor_p = anchor.predict_proba(delta)
            packet.append({"match_id": f"{round_idx}:{j}", "home": m["HomeTeam"],
                           "away": m["AwayTeam"], "anchor_p": anchor_p})
        # anonymize team names for the prompt only
        # (packet carries anchor_p; corrector returns deltas keyed by match_id)
        corr_prompt = build_correction_prompt(packet, playbook.serialize())
        batch = llm.correct(corr_prompt) if round_idx >= skip_first_rounds else None
        deltas = {c.match_id: c for c in (batch.corrections if batch else [])}

        for j, m in enumerate(round_matches):
            mid = f"{round_idx}:{j}"
            anchor_p = packet[j]["anchor_p"]
            c = deltas.get(mid)
            corrected_p, skipped = (apply_correction(anchor_p,
                {"dH": c.dH, "dD": c.dD, "dA": c.dA}) if c else (anchor_p, False))
            block_skipped += int(skipped)
            open_odds, close_odds = _odds(m, odds_source)
            fair_open = _devig(open_odds, devig_method)
            fair_close = _devig(close_odds, devig_method)
            bet = None
            if round_idx >= skip_first_rounds:
                bet = select_bet(corrected_p, fair_open, open_odds, **value_cfg)
            audit.append({"round": round_idx, "match_id": mid, "anchor_p": anchor_p,
                          "corrected_p": corrected_p, "skipped": skipped,
                          "open_odds": open_odds, "result": m["FTR"], "bet": bet})
            block_preds.append(corrected_p); block_out.append(m["FTR"])
            if bet:
                o = bet["outcome"]
                rec = {"outcome": o, "odds": open_odds[o], "won": m["FTR"] == o,
                       "clv": clv(open_odds[o], fair_close[o]), "model_p": corrected_p[o]}
                bets.append(rec); block_bets.append(rec)
            anchor.update(m)

        rounds_since_block += 1
        if rounds_since_block >= block_every_rounds and len(block_bets) >= block_min_bets:
            report = aggregate_block(block_preds, block_out, block_bets, block_skipped)
            new_text = llm.reflect(build_reflection_prompt(report, playbook.serialize()))
            if new_text.strip():
                playbook = Playbook.parse(new_text)
                playbook.enforce_limits(**playbook_limits)
            snapshots.append(playbook.serialize())
            block_preds, block_out, block_bets, block_skipped = [], [], [], 0
            rounds_since_block = 0

    return {"audit": audit, "bets": bets, "final_playbook": playbook.serialize(),
            "snapshots": snapshots}
```

Doplň privátní `_odds`/`_devig` (viz `backtest.py` z Plánu A - lze importovat `backtest._odds_dicts` a `devig`). Ošetři LLM failure (schéma/timeout) → skip zápasu s logem (FakeLLM to netriggeruje; reálný ano).

- [ ] **Step 4: Run → PASS. Step 5: Commit** `feat: LLM-in-the-loop learning orchestrator with block reflection`

---

### Task 7: Ablace + baseliny

Spustí tutéž locked-test sadu v režimech: **learned** (plný `run_learning`), **empty** (prázdný playbook, bez reflexe), **frozen** (seed playbook, bez reflexe), **static** (ručně psaný playbook, bez reflexe), **no-reflection** (korektor jede, ale playbook se nepřepisuje). Plus deterministické baseliny z Plánu A (**anchor-only**, **noise**, **always-favorite**) přes `run_backtest` + `baselines`. "Bez reflexe" = `run_learning` s `block_every_rounds=∞` (nikdy nereflektuje) nebo přepínač.

**Files:** Create `src/vbp/ablations.py`, `tests/test_ablations.py`

- [ ] **Step 1: Failing test (FakeLLM)** - ověří, že `run_ablations` vrátí dict s klíči `learned/empty/frozen/no_reflection/anchor_only/noise`, každý se svými `bets` a souhrnnými metrikami; a že `no_reflection` nezmění playbook (snapshots prázdné nebo seed). Použij FakeLLM s prázdnými korekcemi.

- [ ] **Step 2: Run → FAIL. Step 3: Implement** `run_ablations(...)` - orchestruje `run_learning` s různými seed playbooky a přepínačem reflexe + volá `run_backtest`/baseliny z Plánu A pro deterministické větve. Vrací per-variant `{bets, roi, mean_clv, brier, n_bets}` přes `metrics`.

- [ ] **Step 4: Run → PASS. Step 5: Commit** `feat: ablation battery (learned/empty/frozen/static/no-reflection + det. baselines)`

---

### Task 8: Akceptační vyhodnocení

Z výsledků ablací (na locked testu) spočítá pass/fail proti **předem zamčeným** kritériím (spec §6/§10): CLV > 0 (primárně) a > empty & > noise; ROI > 0 po 1% slippage; N ≥ 120; bootstrap 90% CI dolní mez > −8 p.b.; Brier ≤ anchor i ≤ trh; learned ≥ frozen/static/no-reflection; P/L po odečtení top-3 výher ROI > −10 %. Vrací strukturovaný verdikt (každé kritérium true/false + hodnota).

**Files:** Create `src/vbp/acceptance.py`, `tests/test_acceptance.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_acceptance.py
from vbp.acceptance import evaluate_acceptance

def test_all_pass():
    res = evaluate_acceptance(
        learned={"mean_clv": 0.01, "roi_slip1": 0.02, "n_bets": 150,
                 "roi_ci_lo": -0.05, "brier": 0.60, "roi_drop_top3": -0.02},
        empty={"mean_clv": -0.03}, noise={"mean_clv": -0.04},
        anchor={"brier": 0.62}, market={"brier": 0.61},
        frozen={"mean_clv": 0.005}, static={"mean_clv": 0.0}, no_reflection={"mean_clv": 0.004},
    )
    assert res["passed"] is True
    assert res["criteria"]["clv_positive"] is True

def test_fails_when_clv_not_beating_noise():
    res = evaluate_acceptance(
        learned={"mean_clv": -0.05, "roi_slip1": 0.0, "n_bets": 150,
                 "roi_ci_lo": -0.05, "brier": 0.62, "roi_drop_top3": -0.02},
        empty={"mean_clv": -0.03}, noise={"mean_clv": -0.04},
        anchor={"brier": 0.62}, market={"brier": 0.61},
        frozen={"mean_clv": -0.06}, static={"mean_clv": -0.06}, no_reflection={"mean_clv": -0.06},
    )
    assert res["passed"] is False
```

- [ ] **Step 2: Run → FAIL. Step 3: Implement** `evaluate_acceptance(...)` - čistá funkce vracející `{"passed": bool, "criteria": {name: bool}, "values": {...}}`. Konkrétní prahy zafixované jako konstanty (z §10).

- [ ] **Step 4: Run → PASS. Step 5: Commit** `feat: acceptance-criteria evaluation vs locked thresholds`

---

### Task 9: CLI `vbp-learn` + report

Entrypoint: načte config + data (jako Plán A CLI), postaví `AnthropicClient` s audit logem, spustí `run_ablations` na locked-test, spočítá `evaluate_acceptance`, uloží report (learned vs baseliny vs ablace + verdikt), audit log LLM volání, finální playbook a snapshoty do `runs/learn_<ts>/`. Podporuje `--dry-run` (použije FakeLLM s prázdnými korekcemi = jen anchor, bez nákladů) a `--replay <dir>` (přehraje z uložených odpovědí). Vypíše řádový odhad nákladů před spuštěním živého běhu.

**Files:** Create `src/vbp/learn_cli.py`, `tests/test_learn_report.py` (jen render report); Modify: `pyproject.toml` (přidat `vbp-learn` entrypoint)

- [ ] **Step 1: Failing test** - `render_learn_report(summary)` obsahuje learned ROI/CLV, řádek per ablace, a verdikt PASS/FAIL.
- [ ] **Step 2: Run → FAIL. Step 3: Implement report + CLI.** Přidej do `pyproject.toml`: `vbp-learn = "vbp.learn_cli:main"`.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: `--dry-run` smoke test** na fixture (FakeLLM, žádná síť): ověř, že CLI doběhne a vytiskne report bez volání Anthropic. Nezakomituj throwaway skript.
- [ ] **Step 6: Commit** `feat: vbp-learn CLI (ablations + acceptance report, dry-run + replay)`

---

## Config change (zapoj do implementace Tasku 9)

Sekce `llm` a `corrections` v `config.yaml` z Plánu A se upraví (Sonnet 5 nesmí mít temperature):

```yaml
llm:
  correct_model: claude-haiku-4-5      # přijímá temperature + structured output
  reflect_model: claude-sonnet-5       # adaptivní thinking, BEZ temperature
  temp_correct: 0.0
  reflect_effort: medium               # místo temp_reflect (Sonnet 5 temp = 400)
```

(Odstraň `temp_reflect` z configu; `config.py` z Plánu A rozšiř o pole `reflect_effort` a odeber `temp_reflect`, s testem.)

---

## Definition of Done (Plán B)

- [ ] `pytest -q` zelené včetně nových testů; **žádný test nevolá Anthropic** (vše přes FakeLLM).
- [ ] `vbp-learn --dry-run` doběhne na fixture bez sítě.
- [ ] Živý běh `vbp-learn` na reálných E1 datech: proběhne learned + ablace + baseliny, vytvoří acceptance verdikt, uloží LLM audit log (přehratelný přes `--replay`) + finální playbook + snapshoty.
- [ ] Anti-leak dodržen: korekční prompt neobsahuje kurz ani reálná jména (anonymizováno); ověřeno v `test_prompt.py`.
- [ ] Reprodukovatelnost: `--replay <dir>` dá identický výsledek jako uložený běh (LLM odpovědi z logu, ne nové volání).

## Otevřené k rozhodnutí při živém běhu (ne blokuje implementaci)

- Finální znění promptů (korekce/reflexe) - ladit na validační sezoně 23/24, ne na locked testu.
- Zda `static` (ručně psaný) playbook napsat teď, nebo ablaci `static` vynechat v prvním běhu (frozen/empty/no-reflection stačí na první verdikt).
- Konkrétní odhad nákladů před živým během (řádově jednotky až desítky Kč na plnou baterii vč. ablací; potvrdit `count_tokens` na jednom kole).
