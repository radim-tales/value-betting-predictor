from __future__ import annotations

def render_report(s: dict) -> str:
    lo, hi = s["roi_ci"]
    lines = [
        "# Backtest report (anchor-only, Plán A)",
        "",
        f"- Sázek: **{s['n_bets']}**",
        f"- ROI: **{s['roi']:.3f}** (90% CI [{lo:.3f}, {hi:.3f}])",
        f"- Průměrné CLV: **{s['mean_clv']:.4f}**",
        f"- Brier: **{s['brier']:.4f}** (trh {s['brier_market']:.4f})",
        "",
        "## ROI podle výsledku",
        *[f"- {k}: {('N/A' if v is None else f'{v:.3f}')}" for k, v in s["roi_by_outcome"].items()],
        "",
        "## Baseliny",
        f"- noise: {s['baselines']['noise_roi']:.3f}",
        f"- always_favorite: {s['baselines']['always_favorite_roi']:.3f}",
    ]
    return "\n".join(lines)

def render_learn_report(summary: dict) -> str:
    learned = summary["learned"]
    acceptance = summary["acceptance"]
    verdict = "PASS" if acceptance["passed"] else "FAIL"
    lines = [
        "# Learning report (Plán B - LLM corrector + reflection)",
        "",
        f"- Sázek: **{learned.get('n_bets', 0)}**",
        f"- ROI: **{learned['roi']:.3f}**",
        f"- Průměrné CLV: **{learned['mean_clv']:.4f}**",
        f"- Brier: **{learned['brier']:.4f}**",
        "",
        "## Ablace",
    ]
    for name, v in summary["variants"].items():
        lines.append(f"- {name}: ROI {v['roi']:.3f}, CLV {v['mean_clv']:.4f}")
    lines += [
        "",
        f"## Verdikt: **{verdict}**",
        "",
    ]
    for crit, ok in acceptance["criteria"].items():
        lines.append(f"- {crit}: {'OK' if ok else 'FAIL'}")
    return "\n".join(lines)
