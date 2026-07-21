from __future__ import annotations
from collections import defaultdict
from vbp.metrics import clv, roi, bootstrap_roi_ci, bootstrap_mean_ci


def _settled_bet_rows(store):
    lines = store.load_lines()
    rows = []
    for b in store.load_bets():
        ln = lines.get(b["match_id"])
        if not ln or not ln.get("settled") or not ln.get("pin_close"):
            continue
        o = b["outcome"]
        rows.append({**b,
                     "won": ln["result"] == o,
                     "clv": clv(b["price"], ln["pin_close"][o])})   # p_close_fair * price - 1
    return rows


def summarize(store) -> dict:
    rows = _settled_bet_rows(store)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["book_type"], r["league_tier"])].append(r)
    by = {}
    for key, rs in groups.items():
        clvs = [r["clv"] for r in rs]
        bets = [{"won": r["won"], "odds": r["price"]} for r in rs]   # shape-adapt for vbp.metrics
        by[key] = {"n": len(rs), "wins": sum(r["won"] for r in rs),
                   "mean_clv": sum(clvs) / len(clvs), "clv_ci": bootstrap_mean_ci(clvs, seed=0),
                   "roi": roi(bets), "roi_ci": bootstrap_roi_ci(bets, seed=0)}
    return {"settled_bets": len(rows), "total_bets": len(store.load_bets()), "by": by}


def render(rep: dict) -> str:
    lines = [f"# Live harness report  (settled {rep['settled_bets']}/{rep['total_bets']} bets)", ""]
    for (bt, tier), g in sorted(rep["by"].items()):
        lo, hi = g["clv_ci"]
        verdict = "EDGE" if (bt == "soft" and lo > 0) else ""
        lines.append(f"- {bt:<8} {tier:<9} n={g['n']:>3} wins={g['wins']:>3} "
                     f"CLV={g['mean_clv']:+.4f} [{lo:+.4f},{hi:+.4f}] ROI={g['roi']:+.3f} {verdict}")
    lines.extend([
        "",
        "Caveaty:",
        "- 'close' = poslední snapshot pred vykopem (proxy, ne -5 min).",
        "- Paper-trading ignoruje slippage a limity/bany knih -> potvrzuje EXISTENCI edge, ne skalovatelnost.",
        "- Bootstrap CI bere sazky jako nezavisle, coz nejsou (H/D/A tehoz zapasu + stejny match+outcome "
        "relogovany u ruznych knih pri zmene nejlepsi ceny) -> CI je optimisticky uzsi; ber CLV verdikt s rezervou.",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    from .config import BETS_FILE, LINES_FILE
    from .store import Store
    print(render(summarize(Store(BETS_FILE, LINES_FILE))))
