from __future__ import annotations
import argparse, json
from pathlib import Path

import pandas as pd

from .config import load_config
from .data import load_matches
from .backtest import run_backtest
from .ablations import run_ablations
from .acceptance import evaluate_acceptance
from .metrics import brier, roi_after_slippage, bootstrap_roi_ci, roi_drop_top
from .report import render_learn_report
from .llm.client import AnthropicClient, FakeLLM, ReplayLLM
from .llm.schemas import CorrectionBatch

_EMPTY_SEED_PLAYBOOK = "## Priors\n\n## Rules\n\n## Hypotheses\n\n## Banned\n\n## Notes\n"
_BLOCK_EVERY_ROUNDS_DEFAULT = 4
_BLOCK_MIN_BETS_DEFAULT = 25
_PLAYBOOK_LIMITS_DEFAULT = dict(max_chars=10000, max_rules=12)


def _load_split(data_dir, league, seasons, source):
    frames = [load_matches(Path(data_dir) / f"{s}_{league}.csv", source) for s in seasons]
    return pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)


def _make_log_appender(path: Path):
    def _append(entry: dict) -> None:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")
    return _append


def _read_log_entries(path: Path) -> list[dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vbp-learn")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--replay", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    train = _load_split(args.data_dir, cfg.league, cfg.seasons.train, cfg.odds_source)
    warmup = _load_split(args.data_dir, cfg.league, cfg.seasons.validation, cfg.odds_source)
    test = _load_split(args.data_dir, cfg.league, cfg.seasons.locked_test, cfg.odds_source)

    out = Path(args.out_dir) / f"learn_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    llm_log_path = out / "llm_log.jsonl"

    if args.dry_run:
        llm_factory = lambda: FakeLLM(
            corrections=[CorrectionBatch(corrections=[]) for _ in range(500)],
            reflections=["" for _ in range(500)],
        )
    elif args.replay:
        log_entries = _read_log_entries(Path(args.replay) / "llm_log.jsonl")
        llm_factory = lambda: ReplayLLM(log_entries)
    else:
        print("Živý běh volá Claude API - odhad jednotky až desítky Kč.")
        log_append = _make_log_appender(llm_log_path)
        llm_factory = lambda: AnthropicClient(
            correct_model=cfg.llm.correct_model,
            reflect_model=cfg.llm.reflect_model,
            temp_correct=cfg.llm.temp_correct,
            reflect_effort=cfg.llm.reflect_effort,
            log=log_append,
        )

    anchor_cfg = dict(k=cfg.anchor.k, home_adv=cfg.anchor.home_adv, start_rating=cfg.anchor.start_rating)
    value_cfg = dict(min_edge=cfg.value.min_edge, odds_min=cfg.value.odds_min, odds_max=cfg.value.odds_max)

    ablations = run_ablations(
        train_df=train, test_df=test, warmup_df=warmup, llm_factory=llm_factory,
        seed_playbook=_EMPTY_SEED_PLAYBOOK, odds_source=cfg.odds_source, devig_method=cfg.devig,
        anchor_cfg=anchor_cfg, value_cfg=value_cfg, skip_first_rounds=cfg.value.skip_first_rounds,
        block_every_rounds=_BLOCK_EVERY_ROUNDS_DEFAULT, block_min_bets=_BLOCK_MIN_BETS_DEFAULT,
        playbook_limits=_PLAYBOOK_LIMITS_DEFAULT,
    )

    learned_bets = ablations["learned"]["bets"]
    learned = dict(ablations["learned"])
    learned["roi_slip1"] = roi_after_slippage(learned_bets, 0.01) if learned_bets else 0.0
    learned["roi_ci_lo"] = bootstrap_roi_ci(learned_bets, seed=0)[0] if learned_bets else 0.0
    learned["roi_drop_top3"] = roi_drop_top(learned_bets, 3) if learned_bets else 0.0

    anchor = ablations["anchor_only"]

    bt = run_backtest(train, test, cfg.odds_source, cfg.devig, anchor_cfg, value_cfg,
                       cfg.value.skip_first_rounds, warmup_df=warmup)
    market_brier = brier([r["fair_open"] for r in bt["audit"]], [r["result"] for r in bt["audit"]])
    market = {"brier": market_brier}

    acceptance = evaluate_acceptance(
        learned=learned, empty=ablations["empty"], noise=ablations["noise"],
        anchor=anchor, market=market, frozen=ablations["frozen"], static=None,
        no_reflection=ablations["no_reflection"],
    )

    summary = {
        "learned": learned,
        "variants": {
            "empty": ablations["empty"],
            "frozen": ablations["frozen"],
            "no_reflection": ablations["no_reflection"],
            "anchor_only": ablations["anchor_only"],
            "noise": ablations["noise"],
        },
        "acceptance": acceptance,
    }

    report_md = render_learn_report(summary)
    (out / "report.md").write_text(report_md, encoding="utf-8")
    (out / "final_playbook.md").write_text(learned.get("final_playbook") or "", encoding="utf-8")
    acceptance_clean = {
        "passed": bool(acceptance["passed"]),
        "criteria": {k: bool(v) for k, v in acceptance["criteria"].items()},
        "values": acceptance["values"],
    }
    (out / "acceptance.json").write_text(
        json.dumps(acceptance_clean, default=str, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(report_md)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
