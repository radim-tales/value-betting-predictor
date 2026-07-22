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

def bootstrap_mean_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.10, seed: int = 0):
    """Percentile bootstrap CI for the mean of a value list (e.g. per-bet CLV)."""
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))

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
