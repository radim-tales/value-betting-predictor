from __future__ import annotations
import numpy as np
from scipy.optimize import brentq


def proportional(odds: list[float]) -> np.ndarray:
    r = 1.0 / np.asarray(odds, dtype=float)
    return r / r.sum()


def power(odds: list[float]) -> np.ndarray:
    """Fair probs p_i = (1/odds_i)**k, k chosen so sum(p)=1."""
    r = 1.0 / np.asarray(odds, dtype=float)
    f = lambda k: (r ** k).sum() - 1.0
    k = brentq(f, 0.5, 5.0)
    p = r ** k
    return p / p.sum()


def shin(odds: list[float]) -> np.ndarray:
    """Shin (1992) inversion: recover fair probabilities assuming a proportion z of
    insider money. Solve z so fair probs sum to 1."""
    r = 1.0 / np.asarray(odds, dtype=float)
    s = r.sum()

    def probs(z):
        return (np.sqrt(z * z + 4.0 * (1.0 - z) * r * r / s) - z) / (2.0 * (1.0 - z))

    if s <= 1.0 + 1e-12:  # no margin -> normalized implied
        return r / s
    z = brentq(lambda z: probs(z).sum() - 1.0, 1e-9, 0.5)
    p = probs(z)
    return p / p.sum()


_METHODS = {"shin": shin, "power": power, "proportional": proportional}


def devig(odds: list[float], method: str = "shin") -> np.ndarray:
    if method not in _METHODS:
        raise ValueError(f"unknown devig method: {method}")
    return _METHODS[method](odds)
