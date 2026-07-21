from pathlib import Path

from vbp.data import load_matches
from vbp.llm.client import FakeLLM
from vbp.llm.schemas import CorrectionBatch
from vbp.ablations import run_ablations
from vbp.playbook import Playbook

FIX = Path(__file__).parent / "fixtures" / "mini_league.csv"


def _llm_factory():
    def make():
        return FakeLLM(
            corrections=[CorrectionBatch(corrections=[]) for _ in range(10)],
            reflections=["## Priors\n- learned something\n" for _ in range(10)],
        )
    return make


def _common_kwargs(df):
    return dict(
        train_df=df.iloc[:2], test_df=df.iloc[2:], warmup_df=None,
        llm_factory=_llm_factory(),
        seed_playbook="## Priors\n- start\n",
        odds_source="pinnacle", devig_method="shin",
        anchor_cfg=dict(k=20, home_adv=70, start_rating=1500),
        value_cfg=dict(min_edge=0.0, odds_min=1.0, odds_max=99.0),
        skip_first_rounds=0, block_every_rounds=1, block_min_bets=0,
        playbook_limits=dict(max_chars=10000, max_rules=12),
    )


def test_run_ablations_returns_all_variants():
    df = load_matches(FIX, odds_source="pinnacle")
    result = run_ablations(**_common_kwargs(df))

    expected_keys = {"learned", "empty", "frozen", "no_reflection", "anchor_only", "noise"}
    assert set(result.keys()) == expected_keys

    for name, summary in result.items():
        assert set(summary.keys()) == {"bets", "roi", "mean_clv", "brier", "n_bets", "final_playbook"}
        assert summary["n_bets"] == len(summary["bets"])
        assert isinstance(summary["roi"], float)
        assert isinstance(summary["brier"], float)

    # no_reflection: playbook must never have been rewritten - reflect never fires because
    # block_every_rounds is forced huge internally, so the final playbook equals the
    # normalized seed playbook.
    seed_normalized = Playbook.parse("## Priors\n- start\n").serialize()

    from vbp.learn import run_learning
    kw = _common_kwargs(df)
    no_reflection_result = run_learning(
        train_df=kw["train_df"], test_df=kw["test_df"], warmup_df=kw["warmup_df"],
        llm=kw["llm_factory"](), seed_playbook=kw["seed_playbook"],
        odds_source=kw["odds_source"], devig_method=kw["devig_method"],
        anchor_cfg=kw["anchor_cfg"], value_cfg=kw["value_cfg"],
        skip_first_rounds=kw["skip_first_rounds"], block_every_rounds=10**9,
        block_min_bets=kw["block_min_bets"], playbook_limits=kw["playbook_limits"],
    )
    assert no_reflection_result["snapshots"] == []
    assert no_reflection_result["final_playbook"] == seed_normalized


def test_llm_factory_gives_fresh_client_per_variant():
    df = load_matches(FIX, odds_source="pinnacle")
    calls = []

    def factory():
        client = FakeLLM(
            corrections=[CorrectionBatch(corrections=[]) for _ in range(10)],
            reflections=["## Priors\n- learned something\n" for _ in range(10)],
        )
        calls.append(client)
        return client

    kw = _common_kwargs(df)
    kw["llm_factory"] = factory
    run_ablations(**kw)

    # 4 LLM-driven variants: learned, empty, frozen, no_reflection
    assert len(calls) == 4
    # each got its own instance
    assert len(set(id(c) for c in calls)) == 4
