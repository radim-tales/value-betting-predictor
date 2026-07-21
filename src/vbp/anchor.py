from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression

class EloAnchor:
    def __init__(self, k=20.0, home_adv=70.0, start_rating=1500.0):
        self.k = k
        self.home_adv = home_adv
        self.start = start_rating
        self._r: dict[str, float] = {}
        self._clf: LogisticRegression | None = None
        self._classes = ["H", "D", "A"]

    def rating(self, team: str) -> float:
        return self._r.get(team, self.start)

    def delta(self, home: str, away: str) -> float:
        return self.rating(home) - self.rating(away) + self.home_adv

    def update(self, match: dict) -> None:
        h, a, res = match["HomeTeam"], match["AwayTeam"], match["FTR"]
        exp_h = 1.0 / (1.0 + 10 ** (-(self.delta(h, a)) / 400.0))
        s_h = 1.0 if res == "H" else (0.5 if res == "D" else 0.0)
        change = self.k * (s_h - exp_h)
        self._r[h] = self.rating(h) + change
        self._r[a] = self.rating(a) - change

    def run_and_collect(self, matches: list[dict]) -> list[float]:
        """Walk-forward: record PRE-match delta, then update ratings. Returns deltas."""
        diffs = []
        for m in matches:
            diffs.append(self.delta(m["HomeTeam"], m["AwayTeam"]))
            self.update(m)
        return diffs

    def fit_mapping(self, deltas: list[float], labels: list[str]) -> None:
        X = np.asarray(deltas, dtype=float).reshape(-1, 1)
        y = np.asarray(labels)
        self._clf = LogisticRegression(max_iter=1000)
        self._clf.fit(X, y)

    def predict_proba(self, delta: float) -> dict[str, float]:
        if self._clf is None:
            raise RuntimeError("mapping not fit - call fit_mapping on train first")
        probs = self._clf.predict_proba([[delta]])[0]
        cls = list(self._clf.classes_)
        # robust to a class missing from the training split: absent outcome -> 0.0, then renormalize.
        out = {c: (float(probs[cls.index(c)]) if c in cls else 0.0) for c in self._classes}
        tot = sum(out.values())
        return {k: v / tot for k, v in out.items()}
