from __future__ import annotations
from pathlib import Path

# 1 likvidní + 1 zanedbaná liga (finalizovat dle aktivních + Pinnacle pokrytí, spec §10)
LEAGUES = [
    ("soccer_brazil_campeonato", "liquid"),
    ("soccer_sweden_superettan", "neglected"),
]
REGIONS = "eu,uk"
MIN_EDGE = 0.03
ODDS_MIN = 1.6
ODDS_MAX = 8.0
# anti-leak: snímkování/sázení se v run_once brání proti now >= kickoff (viz run.py),
# takže pin_close = poslední snapshot PŘED výkopem. Žádná časová konstanta tu netřeba.

STATE_DIR = Path("live_state")  # NENÍ v .gitignore, commituje se
BETS_FILE = STATE_DIR / "bets.jsonl"
LINES_FILE = STATE_DIR / "lines.json"
