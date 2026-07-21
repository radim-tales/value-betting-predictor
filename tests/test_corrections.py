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


def test_default_zero_sum_tol_is_point_zero_eight():
    # spec §10 locks the default skip tolerance at 0.08: a delta summing to
    # +0.05 must NOT be skipped, but +0.10 must be skipped, under the default.
    p, skipped = apply_correction(ANCHOR, {"dH": 0.05, "dD": 0.0, "dA": 0.0})
    assert not skipped
    assert abs(sum(p.values()) - 1.0) < 1e-9

    p2, skipped2 = apply_correction(ANCHOR, {"dH": 0.10, "dD": 0.0, "dA": 0.0})
    assert skipped2
    assert p2 == ANCHOR


def test_output_always_sums_to_one_even_with_custom_bounds():
    # With custom bounds (clip_lo=0.05, clip_hi=0.80) that don't satisfy
    # 2*clip_lo + clip_hi == 1, a single projection pass can pin all three
    # coordinates at their clip bounds simultaneously, leaving the naive
    # result summing to 0.9 instead of 1.0. The safety-net renormalization
    # must restore the sum-to-one invariant.
    p, skipped = apply_correction(
        {"H": 0.85, "D": 0.10, "A": 0.05},
        {"dH": 0.14, "dD": -0.095, "dA": -0.045},
        clip_lo=0.05, clip_hi=0.80,
    )
    assert not skipped
    assert abs(sum(p.values()) - 1.0) < 1e-9
