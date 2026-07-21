from __future__ import annotations

from .backtest import run_backtest
from .baselines import noise_probs
from .devig import devig
from .learn import run_learning
from .metrics import brier, clv as _clv, roi
from .value_filter import select_bet

# Reflection never fires: forces block_every_rounds past any realistic test length.
_NO_REFLECTION_ROUNDS = 10**9


def _summary(bets: list[dict], preds: list[dict], outcomes: list[str], final_playbook: str | None = None) -> dict:
    mean_clv = (sum(b["clv"] for b in bets) / len(bets)) if bets else 0.0
    return {
        "bets": bets,
        "roi": roi(bets),
        "mean_clv": mean_clv,
        "brier": brier(preds, outcomes) if preds else 0.0,
        "n_bets": len(bets),
        "final_playbook": final_playbook,
    }


def _learning_variant(llm, seed_playbook, block_every_rounds, *, train_df, test_df, warmup_df,
                       odds_source, devig_method, anchor_cfg, value_cfg, skip_first_rounds,
                       block_min_bets, playbook_limits):
    result = run_learning(
        train_df=train_df, test_df=test_df, warmup_df=warmup_df, llm=llm,
        seed_playbook=seed_playbook, odds_source=odds_source, devig_method=devig_method,
        anchor_cfg=anchor_cfg, value_cfg=value_cfg, skip_first_rounds=skip_first_rounds,
        block_every_rounds=block_every_rounds, block_min_bets=block_min_bets,
        playbook_limits=playbook_limits,
    )
    preds = [r["corrected_p"] for r in result["audit"]]
    outcomes = [r["result"] for r in result["audit"]]
    return _summary(result["bets"], preds, outcomes, final_playbook=result["final_playbook"])


def run_ablations(train_df, test_df, warmup_df, llm_factory, seed_playbook, odds_source,
                   devig_method, anchor_cfg, value_cfg, skip_first_rounds, block_every_rounds,
                   block_min_bets, playbook_limits) -> dict:
    empty_playbook = "## Priors\n\n## Rules\n\n## Hypotheses\n\n## Banned\n\n## Notes\n"

    common = dict(train_df=train_df, test_df=test_df, warmup_df=warmup_df,
                  odds_source=odds_source, devig_method=devig_method, anchor_cfg=anchor_cfg,
                  value_cfg=value_cfg, skip_first_rounds=skip_first_rounds,
                  block_min_bets=block_min_bets, playbook_limits=playbook_limits)

    results = {
        "learned": _learning_variant(llm_factory(), seed_playbook, block_every_rounds, **common),
        "empty": _learning_variant(llm_factory(), empty_playbook, _NO_REFLECTION_ROUNDS, **common),
        "frozen": _learning_variant(llm_factory(), seed_playbook, _NO_REFLECTION_ROUNDS, **common),
        "no_reflection": _learning_variant(llm_factory(), seed_playbook, _NO_REFLECTION_ROUNDS, **common),
    }

    # Deterministic baselines (Plan A).
    bt = run_backtest(train_df, test_df, odds_source, devig_method, anchor_cfg, value_cfg,
                       skip_first_rounds, warmup_df=warmup_df)
    anchor_preds = [r["anchor_p"] for r in bt["audit"]]
    anchor_outcomes = [r["result"] for r in bt["audit"]]
    results["anchor_only"] = _summary(bt["bets"], anchor_preds, anchor_outcomes)

    noise_preds = [noise_probs(r["fair_open"], seed=r["i"]) for r in bt["audit"]]
    noise_outcomes = [r["result"] for r in bt["audit"]]
    noise_bets = []
    for r, np_ in zip(bt["audit"], noise_preds):
        nb = select_bet(np_, r["fair_open"], r["open_odds"], **value_cfg)
        if nb:
            o = nb["outcome"]
            fair_close = dict(zip(("H", "D", "A"),
                                   devig([r["close_odds"]["H"], r["close_odds"]["D"], r["close_odds"]["A"]],
                                         devig_method)))
            noise_bets.append({
                "outcome": o, "odds": r["open_odds"][o], "won": r["result"] == o,
                "clv": _clv(r["open_odds"][o], fair_close[o]),
            })
    results["noise"] = _summary(noise_bets, noise_preds, noise_outcomes)

    return results
