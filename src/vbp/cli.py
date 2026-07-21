from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from .config import load_config
from .data import load_matches
from .backtest import run_backtest
from .baselines import noise_probs, always_favorite_pick
from .value_filter import select_bet
from .devig import devig
from .metrics import (roi, bootstrap_roi_ci, brier, roi_by_outcome, roi_after_slippage, roi_drop_top)
from .report import render_report

def _load_split(data_dir, league, seasons, source):
    frames = [load_matches(Path(data_dir) / f"{s}_{league}.csv", source) for s in seasons]
    return pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)

def main(argv=None):
    ap = argparse.ArgumentParser(prog="vbp-backtest")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out-dir", default="runs")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    train = _load_split(args.data_dir, cfg.league, cfg.seasons.train + cfg.seasons.validation, cfg.odds_source)
    test = _load_split(args.data_dir, cfg.league, cfg.seasons.locked_test, cfg.odds_source)

    result = run_backtest(
        train_df=train, test_df=test, odds_source=cfg.odds_source, devig_method=cfg.devig,
        anchor_cfg=dict(k=cfg.anchor.k, home_adv=cfg.anchor.home_adv, start_rating=cfg.anchor.start_rating),
        value_cfg=dict(min_edge=cfg.value.min_edge, odds_min=cfg.value.odds_min, odds_max=cfg.value.odds_max),
        skip_first_rounds=cfg.value.skip_first_rounds,
    )
    bets = result["bets"]
    all_preds = [r["anchor_p"] for r in result["audit"]]
    all_out = [r["result"] for r in result["audit"]]
    market_preds = [r["fair_open"] for r in result["audit"]]

    summary = {
        "n_bets": len(bets),
        "roi": roi(bets),
        "roi_ci": bootstrap_roi_ci(bets, seed=0),
        "mean_clv": (sum(b["clv"] for b in bets) / len(bets)) if bets else 0.0,
        "brier": brier(all_preds, all_out),
        "brier_market": brier(market_preds, all_out),
        "roi_by_outcome": roi_by_outcome(bets),
        "roi_slippage_1pct": roi_after_slippage(bets, 0.01),
        "roi_drop_top3": roi_drop_top(bets, 3) if bets else 0.0,   # concentration check (Plán B accept. crit.)
        "baselines": _baselines(result["audit"], cfg),
    }
    out = Path(args.out_dir) / pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(render_report(summary), encoding="utf-8")
    (out / "audit.json").write_text(json.dumps(result["audit"], default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    print(render_report(summary))
    print(f"\nSaved to {out}")

def _baselines(audit, cfg):
    vc = dict(min_edge=cfg.value.min_edge, odds_min=cfg.value.odds_min, odds_max=cfg.value.odds_max)
    noise_bets = []
    for r in audit:
        odds = r["open_odds"]                       # raw odds stored by backtest
        np_ = noise_probs(r["fair_open"], seed=r["i"])   # noise baseline: market fair + noise
        nb = select_bet(np_, r["fair_open"], odds, **vc)
        if nb:
            o = nb["outcome"]
            noise_bets.append({"outcome": o, "odds": odds[o], "won": r["result"] == o})
    return {
        "noise_roi": roi(noise_bets),
        "always_favorite_roi": _favorite_roi(audit),
    }

def _favorite_roi(audit):
    """Always bet the lowest-odds outcome (the market favorite), flat stake."""
    bets = []
    for r in audit:
        odds = r["open_odds"]
        o = always_favorite_pick(odds)
        bets.append({"outcome": o, "odds": odds[o], "won": r["result"] == o})
    return roi(bets)

if __name__ == "__main__":
    main()
