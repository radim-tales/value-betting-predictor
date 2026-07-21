from __future__ import annotations

from vbp.metrics import brier, roi


def aggregate_block(
    preds: list[dict],
    outcomes: list[str],
    bets: list[dict],
    n_skipped: int,
) -> dict:
    """Aggregate a block of evaluated rounds into a metrics-only reflection summary.

    Deliberately excludes any per-match data (no match stories) so the
    reflection step reasons about signal, not anecdotes.
    """
    n_bets = len(bets)

    block_roi = roi(bets) if bets else 0.0

    mean_clv = sum(b["clv"] for b in bets) / n_bets if n_bets else 0.0

    block_brier = brier(preds, outcomes) if preds else 0.0

    brier_by_outcome: dict[str, float] = {}
    if preds:
        for cls in ("H", "D", "A"):
            idx = [i for i, y in enumerate(outcomes) if y == cls]
            if idx:
                sub_preds = [preds[i] for i in idx]
                sub_outcomes = [outcomes[i] for i in idx]
                brier_by_outcome[cls] = brier(sub_preds, sub_outcomes)
            else:
                brier_by_outcome[cls] = 0.0

    overconfidence = (
        sum(b["model_p"] - (1.0 if b["won"] else 0.0) for b in bets) / n_bets
        if n_bets
        else 0.0
    )

    return {
        "n_bets": n_bets,
        "n_skipped": n_skipped,
        "roi": block_roi,
        "clv": mean_clv,
        "brier": block_brier,
        "brier_by_outcome": brier_by_outcome,
        "overconfidence": overconfidence,
    }
