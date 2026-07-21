from __future__ import annotations

import pandas as pd

from .anchor import EloAnchor
from .anonymize import anonymize_teams
from .backtest import _odds_dicts
from .block_report import aggregate_block
from .corrections import apply_correction
from .devig import devig
from .features import build_features
from .metrics import clv
from .playbook import Playbook
from .prompt import build_correction_prompt, build_reflection_prompt
from .value_filter import select_bet


def _devig_dict(odds: dict, method: str) -> dict:
    fair = devig([odds["H"], odds["D"], odds["A"]], method)
    return dict(zip(("H", "D", "A"), fair))


def run_learning(train_df, test_df, warmup_df, llm, seed_playbook,
                 odds_source, devig_method, anchor_cfg, value_cfg,
                 skip_first_rounds, block_every_rounds, block_min_bets, playbook_limits):
    anchor = EloAnchor(**anchor_cfg)
    tm = train_df.to_dict("records")
    anchor.fit_mapping(anchor.run_and_collect(tm), [m["FTR"] for m in tm])
    if warmup_df is not None:
        for m in warmup_df.to_dict("records"):
            anchor.update(m)

    playbook = Playbook.parse(seed_playbook)
    audit, bets, snapshots = [], [], []
    block_preds, block_out, block_bets, block_skipped = [], [], [], 0
    rounds_since_block = 0

    # Pre-match features for every test match, computed leak-safe (build_features uses
    # only strictly-earlier matches). Concat all history so a test match sees train+warmup+
    # earlier-test form. Index by (Date, HomeTeam, AwayTeam).
    hist = pd.concat([df for df in (train_df, warmup_df, test_df) if df is not None], ignore_index=True)
    feats = build_features(hist)                       # from Plan A
    feat_by_key = {(r["Date"], r["HomeTeam"], r["AwayTeam"]):
                   {c: r[c] for c in feats.columns if c not in ("Date", "HomeTeam", "AwayTeam")}
                   for _, r in feats.iterrows()}
    # One consistent anonymization mapping over ALL teams (prompt-only; never leaks real names).
    all_teams = list(hist["HomeTeam"]) + list(hist["AwayTeam"])
    _, anon = anonymize_teams(all_teams)               # from Plan A: {real: "Team_N"}

    # group test matches into rounds by date (a "round" = matches sharing a date)
    test = test_df.reset_index(drop=True)
    for round_idx, (date, group) in enumerate(test.groupby("Date", sort=True)):
        # build anonymized packet for the round (features from strictly-earlier matches)
        # NOTE: features computed on the full history-to-date; odds NEVER in the packet
        round_matches = group.to_dict("records")
        packet = []
        for j, m in enumerate(round_matches):
            delta = anchor.delta(m["HomeTeam"], m["AwayTeam"])
            anchor_p = anchor.predict_proba(delta)
            key = (m["Date"], m["HomeTeam"], m["AwayTeam"])
            packet.append({"match_id": f"{round_idx}:{j}",
                           "home": anon[m["HomeTeam"]], "away": anon[m["AwayTeam"]],  # anonymized
                           "anchor_p": anchor_p,
                           "features": feat_by_key.get(key, {})})                    # pre-match, no odds
        # packet carries anonymized names + features + anchor_p (NO odds); corrector
        # returns deltas keyed by match_id. build_correction_prompt must not add odds.
        corr_prompt = build_correction_prompt(packet, playbook.serialize())
        batch = llm.correct(corr_prompt) if round_idx >= skip_first_rounds else None
        deltas = {c.match_id: c for c in (batch.corrections if batch else [])}

        for j, m in enumerate(round_matches):
            mid = f"{round_idx}:{j}"
            anchor_p = packet[j]["anchor_p"]
            c = deltas.get(mid)
            corrected_p, skipped = (apply_correction(anchor_p,
                {"dH": c.dH, "dD": c.dD, "dA": c.dA}) if c else (anchor_p, False))
            block_skipped += int(skipped)
            open_odds, close_odds = _odds_dicts(m, odds_source)
            fair_open = _devig_dict(open_odds, devig_method)
            fair_close = _devig_dict(close_odds, devig_method)
            bet = None
            if round_idx >= skip_first_rounds:
                bet = select_bet(corrected_p, fair_open, open_odds, **value_cfg)
            audit.append({"round": round_idx, "match_id": mid, "anchor_p": anchor_p,
                          "corrected_p": corrected_p, "skipped": skipped,
                          "open_odds": open_odds, "result": m["FTR"], "bet": bet})
            block_preds.append(corrected_p); block_out.append(m["FTR"])
            if bet:
                o = bet["outcome"]
                rec = {"outcome": o, "odds": open_odds[o], "won": m["FTR"] == o,
                       "clv": clv(open_odds[o], fair_close[o]), "model_p": corrected_p[o]}
                bets.append(rec); block_bets.append(rec)
            anchor.update(m)

        rounds_since_block += 1
        if rounds_since_block >= block_every_rounds and len(block_bets) >= block_min_bets:
            report = aggregate_block(block_preds, block_out, block_bets, block_skipped)
            new_text = llm.reflect(build_reflection_prompt(report, playbook.serialize()))
            if new_text.strip():
                playbook = Playbook.parse(new_text)
                playbook.enforce_limits(**playbook_limits)
            snapshots.append(playbook.serialize())
            block_preds, block_out, block_bets, block_skipped = [], [], [], 0
            rounds_since_block = 0

    return {"audit": audit, "bets": bets, "final_playbook": playbook.serialize(),
            "snapshots": snapshots}
