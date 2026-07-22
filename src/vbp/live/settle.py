from __future__ import annotations

def result_from_scores(ev: dict) -> str | None:
    if not ev.get("completed") or not ev.get("scores"):
        return None
    by = {s["name"]: int(s["score"]) for s in ev["scores"]}
    h = by.get(ev["home_team"]); a = by.get(ev["away_team"])
    if h is None or a is None:
        return None
    return "H" if h > a else ("A" if a > h else "D")

def settle_finished(store, scores: list[dict]) -> int:
    """Write result+settled to lines for finished, not-yet-settled matches. Returns #settled."""
    lines = store.load_lines()
    n = 0
    for ev in scores:
        mid = ev.get("id")
        if mid in lines and not lines[mid]["settled"]:
            r = result_from_scores(ev)
            if r:
                store.set_result(mid, r)
                n += 1
    return n
