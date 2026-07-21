import pytest
from vbp.config import load_config, Config

def test_loads_defaults_from_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "league: E1\n"
        "seasons:\n  train: ['2122','2223']\n  validation: ['2324']\n  locked_test: ['2425']\n"
        "odds_source: pinnacle\ndevig: shin\n"
        "anchor: {type: elo_softmax_map, k: 20, home_adv: 70, start_rating: 1500}\n"
        "value: {min_edge: 0.03, odds_min: 1.6, odds_max: 4.5, skip_first_rounds: 4, stake: 1.0, max_bets_per_match: 1}\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert isinstance(cfg, Config)
    assert cfg.league == "E1"
    assert cfg.seasons.train == ["2122", "2223"]
    assert cfg.value.min_edge == 0.03
    assert cfg.value.odds_max == 4.5

def test_rejects_overlapping_splits(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "league: E1\n"
        "seasons:\n  train: ['2122','2324']\n  validation: ['2324']\n  locked_test: ['2425']\n"
        "odds_source: pinnacle\ndevig: shin\n"
        "anchor: {type: elo_softmax_map, k: 20, home_adv: 70, start_rating: 1500}\n"
        "value: {min_edge: 0.03, odds_min: 1.6, odds_max: 4.5, skip_first_rounds: 4, stake: 1.0, max_bets_per_match: 1}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overlap"):
        load_config(p)
