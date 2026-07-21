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
