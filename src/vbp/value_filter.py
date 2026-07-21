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
