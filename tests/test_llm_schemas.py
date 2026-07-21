import pytest

from vbp.llm.schemas import Correction, CorrectionBatch


def test_correction_batch_parses():
    b = CorrectionBatch(corrections=[
        Correction(match_id="0", dH=0.03, dD=-0.01, dA=-0.02, rationale="home form"),
    ])
    assert b.corrections[0].match_id == "0"
    assert b.corrections[0].dH == 0.03


def test_delta_hard_cap_enforced():
    with pytest.raises(Exception):
        Correction(match_id="1", dH=0.5, dD=-0.25, dA=-0.25)  # |dH| > 0.15 hard cap
