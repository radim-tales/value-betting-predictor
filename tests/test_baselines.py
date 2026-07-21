from vbp.baselines import always_favorite_pick, noise_probs


def test_always_favorite_picks_lowest_odds():
    odds = {"H": 1.80, "D": 3.40, "A": 4.50}
    assert always_favorite_pick(odds) == "H"


def test_noise_probs_sum_to_one_and_are_reproducible():
    fair = {"H": 0.5, "D": 0.3, "A": 0.2}
    a = noise_probs(fair, sigma=0.03, seed=1)
    b = noise_probs(fair, sigma=0.03, seed=1)
    assert a == b
    assert abs(sum(a.values()) - 1.0) < 1e-9
