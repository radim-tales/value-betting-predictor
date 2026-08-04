from __future__ import annotations
import json
from pathlib import Path

class Store:
    def __init__(self, bets_file, lines_file):
        self.bets_file = Path(bets_file)
        self.lines_file = Path(lines_file)
        self.state_file = self.lines_file.parent / "settle_state.json"
        self.bets_file.parent.mkdir(parents=True, exist_ok=True)

    def last_settle_date(self) -> str | None:
        if not self.state_file.exists():
            return None
        return json.loads(self.state_file.read_text(encoding="utf-8")).get("last_settle_date")

    def mark_settled(self, date_str: str):
        self.state_file.write_text(json.dumps({"last_settle_date": date_str}), encoding="utf-8")

    def load_bets(self) -> list[dict]:
        if not self.bets_file.exists():
            return []
        return [json.loads(l) for l in self.bets_file.read_text(encoding="utf-8").splitlines() if l.strip()]

    def _bet_keys(self) -> set:
        return {(b["match_id"], b["outcome"], b["book"]) for b in self.load_bets()}

    def add_bet(self, bet: dict) -> bool:
        """Append if (match_id,outcome,book) not seen. Returns True if added."""
        if (bet["match_id"], bet["outcome"], bet["book"]) in self._bet_keys():
            return False
        with self.bets_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(bet, ensure_ascii=False) + "\n")
        return True

    def load_lines(self) -> dict:
        if not self.lines_file.exists():
            return {}
        return json.loads(self.lines_file.read_text(encoding="utf-8"))

    def _save_lines(self, lines: dict):
        self.lines_file.write_text(json.dumps(lines, ensure_ascii=False, indent=1), encoding="utf-8")

    def update_line(self, match_id: str, meta: dict, pin_fair: dict):
        lines = self.load_lines()
        ln = lines.get(match_id, {**meta, "pin_open": None, "pin_close": None,
                                   "result": None, "settled": False})
        if ln["pin_open"] is None:
            ln["pin_open"] = pin_fair
        ln["pin_close"] = pin_fair                       # last one before kickoff wins
        lines[match_id] = ln
        self._save_lines(lines)

    def set_result(self, match_id: str, result: str):
        lines = self.load_lines()
        if match_id in lines:
            lines[match_id]["result"] = result
            lines[match_id]["settled"] = True
            self._save_lines(lines)
