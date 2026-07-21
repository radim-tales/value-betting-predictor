from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Seasons:
    train: list[str]
    validation: list[str]
    locked_test: list[str]

@dataclass
class AnchorCfg:
    type: str = "elo_softmax_map"
    k: float = 20.0
    home_adv: float = 70.0
    start_rating: float = 1500.0

@dataclass
class ValueCfg:
    min_edge: float = 0.03
    odds_min: float = 1.6
    odds_max: float = 4.5
    skip_first_rounds: int = 4
    stake: float = 1.0
    max_bets_per_match: int = 1

@dataclass
class LlmCfg:
    correct_model: str = "claude-haiku-4-5"
    reflect_model: str = "claude-sonnet-5"
    temp_correct: float = 0.0
    reflect_effort: str = "medium"

@dataclass
class Config:
    league: str
    seasons: Seasons
    odds_source: str
    devig: str
    anchor: AnchorCfg
    value: ValueCfg
    llm: LlmCfg

def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    seasons = Seasons(**raw["seasons"])
    all_splits = seasons.train + seasons.validation + seasons.locked_test
    if len(all_splits) != len(set(all_splits)):
        raise ValueError("season splits overlap - train/validation/locked_test must be disjoint")
    cfg = Config(
        league=raw["league"],
        seasons=seasons,
        odds_source=raw["odds_source"],
        devig=raw["devig"],
        anchor=AnchorCfg(**raw.get("anchor", {})),
        value=ValueCfg(**raw.get("value", {})),
        llm=LlmCfg(**raw.get("llm", {})),
    )
    if not (0 < cfg.value.min_edge < 0.5):
        raise ValueError("min_edge out of range")
    if cfg.value.odds_min >= cfg.value.odds_max:
        raise ValueError("odds_min must be < odds_max")
    return cfg
