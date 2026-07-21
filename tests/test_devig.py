import numpy as np
import pytest
from vbp.devig import devig, shin, proportional


def test_output_sums_to_one():
    p = devig([1.90, 3.50, 4.20], method="shin")
    assert abs(sum(p) - 1.0) < 1e-9


def test_fair_book_returns_input():
    # odds with zero margin: probabilities 0.5/0.3/0.2 -> odds 2.0/3.333/5.0
    odds = [2.0, 1 / 0.3, 5.0]
    p = devig(odds, method="shin")
    assert np.allclose(p, [0.5, 0.3, 0.2], atol=1e-3)


def test_preserves_ordering():
    p = devig([1.90, 3.50, 4.20], method="shin")
    assert p[0] > p[1] > p[2]


def test_shin_removes_more_margin_from_longshot_than_proportional():
    # NOTE: [1.90, 3.50, 6.50] from the task spec implies sum(1/odds) = 0.9659,
    # i.e. an underround (no bookmaker margin), which trips shin()'s s<=1
    # fallback and returns proportional exactly -> equal, not strictly less.
    # Using odds with real overround (sum(1/odds) = 1.031, ~3% margin) so the
    # favorite-longshot bias is actually exercised.
    odds = [1.83, 3.30, 5.50]
    ps = shin(odds)
    pp = proportional(odds)
    # favorite-longshot bias: Shin assigns lower fair prob to the longshot than naive proportional
    assert ps[2] < pp[2]


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        devig([2.0, 3.0, 4.0], method="nope")
