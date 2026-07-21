from __future__ import annotations

import json

DELTA_CAP = 0.10


def build_correction_prompt(matches: list[dict], playbook_text: str) -> str:
    """Build the per-block correction prompt for the LLM corrector.

    The corrector only ever sees: the playbook, anonymized team names,
    pre-match features (form, table points, goals, rest days, etc.) and the
    anchor probabilities. It never sees betting odds - the correction must
    be based purely on the playbook and the match context, not on market
    signal. This function must never emit the words "odds" or "kurz".
    """
    parts: list[str] = []

    # Playbook goes first so it sits in the stable, cacheable prefix of the
    # prompt (prompt caching keys off a shared prefix across calls).
    parts.append("# Playbook\n")
    parts.append(playbook_text.strip())
    parts.append("\n")

    parts.append(
        "\n# Role\n"
        "You are a value-betting probability corrector. For each match below "
        "you are given an anchor probability distribution over the outcomes "
        "H (home win), D (draw), A (away win), together with anonymized "
        "pre-match features (team form, league table points, goal statistics, "
        "rest days). Team names have been anonymized and no market signal is "
        "provided - base your correction only on the playbook and the "
        "features.\n"
    )

    parts.append(
        "\n# Task\n"
        "For each match, propose a small correction (delta) to the anchor "
        "probabilities, guided by the playbook's priors, rules and "
        "hypotheses. Requirements:\n"
        "- Output deltas dH, dD, dA that are zero-sum (dH + dD + dA ~ 0).\n"
        f"- Each delta must be capped at +/-{DELTA_CAP:.2f}.\n"
        "- Give a short rationale (one sentence) that references which "
        "playbook item motivated the correction.\n"
        "- If nothing in the playbook applies to a match, return all deltas "
        "as 0.0 with rationale \"no applicable playbook item\".\n"
    )

    parts.append(
        "\n# Output format\n"
        "Return a single JSON object keyed by match_id. Each value must be "
        "an object with keys dH, dD, dA (floats) and rationale (string). "
        "Example:\n"
        '{"<match_id>": {"dH": 0.02, "dD": -0.01, "dA": -0.01, '
        '"rationale": "..."}}\n'
        "Return JSON only, no surrounding prose.\n"
    )

    parts.append("\n# Matches\n")
    for m in matches:
        entry = {
            "match_id": m.get("match_id"),
            "home": m.get("home"),
            "away": m.get("away"),
            "anchor_p": m.get("anchor_p"),
            "features": m.get("features"),
        }
        parts.append(json.dumps(entry, ensure_ascii=False))
        parts.append("\n")

    return "".join(parts)


def build_reflection_prompt(report: dict, playbook_text: str) -> str:
    """Build the block-level reflection prompt that asks the LLM to rewrite
    the playbook based on aggregated calibration/performance metrics for the
    block - never individual match stories (that would invite overfitting
    to anecdotes instead of learning from calibration).
    """
    parts: list[str] = []

    parts.append("# Current playbook\n")
    parts.append(playbook_text.strip())
    parts.append("\n")

    parts.append(
        "\n# Role\n"
        "You are reflecting on a completed block of value bets to improve "
        "the playbook that guides future probability corrections. You are "
        "given only aggregated performance and calibration metrics for the "
        "block, not individual match stories - reason from the metrics, not "
        "from anecdotes.\n"
    )

    metrics = {
        "n_bets": report.get("n_bets"),
        "roi": _fmt(report.get("roi")),
        "clv": _fmt(report.get("clv")),
        "brier": _fmt(report.get("brier")),
        "brier_by_outcome": {
            k: _fmt(v) for k, v in (report.get("brier_by_outcome") or {}).items()
        },
        "overconfidence": _fmt(report.get("overconfidence")),
    }

    parts.append("\n# Block metrics\n")
    parts.append(json.dumps(metrics, ensure_ascii=False, indent=2))
    parts.append("\n")

    parts.append(
        "\n# Task\n"
        "Rewrite the playbook, keeping the same section structure "
        "(## Priors, ## Rules, ## Hypotheses, ## Banned, ## Notes). For "
        "every change you make (added, edited, removed, or promoted/demoted "
        "item), state the metric-based reason (e.g. \"brier_by_outcome.D is "
        "high -> tighten draw prior\", \"overconfidence positive -> shrink "
        "correction magnitude\"). Be disciplined and concise: prefer removing "
        "or downgrading a rule over inventing a new one unless the metrics "
        "clearly support it. Do not reference individual matches - only the "
        "aggregated metrics above.\n"
    )

    parts.append(
        "\n# Output format\n"
        "Return the full rewritten playbook as markdown text (same section "
        "headers), followed by a short changelog listing each change and its "
        "metric-based reason.\n"
    )

    return "".join(parts)


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}" if isinstance(value, float) else str(value)
