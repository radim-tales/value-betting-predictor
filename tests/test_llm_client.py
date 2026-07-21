from vbp.llm.client import FakeLLM
from vbp.llm.schemas import CorrectionBatch, Correction


def test_fake_llm_returns_scripted_corrections():
    fake = FakeLLM(corrections=[CorrectionBatch(corrections=[
        Correction(match_id="0", dH=0.02, dD=-0.01, dA=-0.01)])],
        reflections=["## Priors\n- updated\n"])
    batch = fake.correct(prompt="ignored")
    assert isinstance(batch, CorrectionBatch)
    assert batch.corrections[0].match_id == "0"
    assert fake.reflect(prompt="ignored") == "## Priors\n- updated\n"


def test_fake_llm_logs_calls():
    fake = FakeLLM(corrections=[CorrectionBatch(corrections=[])], reflections=["x"])
    fake.correct(prompt="P1")
    fake.reflect(prompt="P2")
    assert [c["kind"] for c in fake.calls] == ["correct", "reflect"]


def test_replay_llm_reproduces_from_log_by_prompt_hash():
    import hashlib
    from vbp.llm.client import ReplayLLM
    log = [
        {"kind": "correct", "prompt_sha": hashlib.sha256(b"CP").hexdigest(),
         "corrections": {"corrections": [{"match_id": "0", "dH": 0.02, "dD": -0.01, "dA": -0.01}]}},
        {"kind": "reflect", "prompt_sha": hashlib.sha256(b"RP").hexdigest(), "text": "## Priors\n- r\n"},
    ]
    r = ReplayLLM(log)
    b = r.correct("CP")
    assert b.corrections[0].dH == 0.02
    assert r.reflect("RP") == "## Priors\n- r\n"
