from __future__ import annotations


def anonymize_teams(team_sequence: list[str]) -> tuple[list[str], dict[str, str]]:
    """Map team names to Team_N ids, first-seen order = stable & deterministic.
    Returns (anonymized_sequence, {original: anon})."""
    mapping: dict[str, str] = {}
    out: list[str] = []
    for t in team_sequence:
        if t not in mapping:
            mapping[t] = f"Team_{len(mapping)}"
        out.append(mapping[t])
    return out, mapping
