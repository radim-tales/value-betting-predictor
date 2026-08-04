from __future__ import annotations
from pathlib import Path

# 1 likvidní + 1 zanedbaná liga (finalizovat dle aktivních + Pinnacle pokrytí, spec §10)
LEAGUES = [
    ("soccer_brazil_campeonato", "liquid"),
    ("soccer_sweden_superettan", "neglected"),
]
REGIONS = "eu,uk"
# Rozpočet kreditů (free tier The Odds API = 500/měsíc, ověřeno naživo 22.7.2026):
#   odds stojí 1 kredit ZA REGION -> "eu,uk" = 2 kredity/liga/běh
#   scores(daysFrom) = 2 kredity/liga/běh, list_sports = 0
# 2 ligy x 3 odds běhy = 12/den + settle 1x denně 2 ligy x 2 = 4 -> 16/den ~ 480/měsíc.
# Settle se pouští 1x/den podle uloženého data (Store.last_settle_date), NE podle
# now.hour - GitHub cron se opožďuje a hodinový gate skoro nikdy nesedl (bug do 08/2026).
MIN_EDGE = 0.03
ODDS_MIN = 1.6
ODDS_MAX = 8.0
# anti-leak: snímkování/sázení se v run_once brání proti now >= kickoff (viz run.py),
# takže pin_close = poslední snapshot PŘED výkopem. Žádná časová konstanta tu netřeba.

STATE_DIR = Path("live_state")  # NENÍ v .gitignore, commituje se
BETS_FILE = STATE_DIR / "bets.jsonl"
LINES_FILE = STATE_DIR / "lines.json"
