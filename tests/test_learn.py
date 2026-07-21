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


class _AlwaysFailsLLM:
    """Stub whose correct/reflect always raise - exercises the retry-then-fallback
    path in run_learning without needing a real network call."""
    def correct(self, prompt):
        raise RuntimeError("boom: correct")

    def reflect(self, prompt):
        raise RuntimeError("boom: reflect")


def test_learning_survives_llm_failure():
    df = load_matches(FIX, odds_source="pinnacle")
    result = run_learning(
        train_df=df.iloc[:2], test_df=df.iloc[2:], warmup_df=None,
        llm=_AlwaysFailsLLM(), seed_playbook="## Priors\n- start\n",
        odds_source="pinnacle", devig_method="shin",
        anchor_cfg=dict(k=20, home_adv=70, start_rating=1500),
        value_cfg=dict(min_edge=0.0, odds_min=1.0, odds_max=99.0),
        skip_first_rounds=0, block_every_rounds=1, block_min_bets=0,
        playbook_limits=dict(max_chars=10000, max_rules=12),
    )
    # run completes and produces audit rows despite every LLM call failing
    assert len(result["audit"]) == len(df.iloc[2:])
    # correct() always fails -> no corrections applied -> corrected_p == anchor_p
    for r in result["audit"]:
        assert r["corrected_p"] == r["anchor_p"]
    # reflect() always fails -> playbook rewrite skipped, seed playbook kept
    from vbp.playbook import Playbook
    assert result["final_playbook"] == Playbook.parse("## Priors\n- start\n").serialize()


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
