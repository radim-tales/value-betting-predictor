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
