"""Acceptance-criteria evaluation against locked thresholds (spec §6/§10).

Consumes per-variant summaries produced by the ablation study (Task 7) plus
ROI/CI/concentration figures computed by the CLI (Task 9) and returns a
structured pass/fail verdict.
"""

from __future__ import annotations

from typing import Any

MIN_N_BETS = 120
MIN_ROI_CI_LO = -0.08
MIN_ROI_DROP_TOP3 = -0.10


def evaluate_acceptance(
    learned: dict[str, Any],
    empty: dict[str, Any],
    noise: dict[str, Any],
    anchor: dict[str, Any],
    market: dict[str, Any],
    frozen: dict[str, Any],
    static: dict[str, Any] | None,
    no_reflection: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate locked acceptance criteria (spec §6/§10) for one run.

    Returns {"passed": bool, "criteria": {name: bool}, "values": {...}}.
    """
    learned_clv = learned["mean_clv"]
    static_clv = static["mean_clv"] if static is not None else float("-inf")

    criteria = {
        "clv_positive": learned_clv > 0,
        "clv_beats_empty": learned_clv > empty["mean_clv"],
        "clv_beats_noise": learned_clv > noise["mean_clv"],
        "roi_positive_after_slippage": learned["roi_slip1"] > 0,
        "enough_bets": learned["n_bets"] >= MIN_N_BETS,
        "ci_lower_ok": learned["roi_ci_lo"] > MIN_ROI_CI_LO,
        "brier_beats_anchor": learned["brier"] <= anchor["brier"],
        "brier_beats_market": learned["brier"] <= market["brier"],
        "learned_beats_ablations": learned_clv >= max(
            frozen["mean_clv"], static_clv, no_reflection["mean_clv"]
        ),
        "not_concentrated": learned["roi_drop_top3"] > MIN_ROI_DROP_TOP3,
    }

    values = {
        "learned_clv": learned_clv,
        "empty_clv": empty["mean_clv"],
        "noise_clv": noise["mean_clv"],
        "n_bets": learned["n_bets"],
        "roi_ci_lo": learned["roi_ci_lo"],
        "learned_brier": learned["brier"],
        "anchor_brier": anchor["brier"],
        "market_brier": market["brier"],
    }

    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "values": values,
    }
