from vbp.corrections import apply_correction

ANCHOR = {"H": 0.50, "D": 0.30, "A": 0.20}


def test_applies_zero_sum_delta():
    p, skipped = apply_correction(ANCHOR, {"dH": 0.03, "dD": -0.01, "dA": -0.02})
    assert not skipped
    assert abs(sum(p.values()) - 1.0) < 1e-9
    assert p["H"] > ANCHOR["H"]


def test_skips_when_delta_not_zero_sum():
    # deltas sum to +0.10 -> outside tolerance -> skip, return anchor
    p, skipped = apply_correction(ANCHOR, {"dH": 0.10, "dD": 0.0, "dA": 0.0}, zero_sum_tol=0.02)
    assert skipped
    assert p == ANCHOR


def test_clips_and_renormalizes():
    # push A negative -> clipped to 0.01 then renormalized
    p, skipped = apply_correction({"H": 0.60, "D": 0.35, "A": 0.05},
                                  {"dH": 0.0, "dD": 0.10, "dA": -0.10})
    assert not skipped
    assert p["A"] >= 0.01
    assert abs(sum(p.values()) - 1.0) < 1e-9
