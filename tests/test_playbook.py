from pathlib import Path

from vbp.playbook import Playbook

SEED = (Path(__file__).parent / "fixtures" / "seed_playbook.md").read_text(encoding="utf-8")


def test_roundtrip_parse_serialize():
    pb = Playbook.parse(SEED)
    assert "Priors" in pb.sections
    out = pb.serialize()
    assert Playbook.parse(out).sections["Priors"] == pb.sections["Priors"]


def test_enforce_max_chars_trims_notes_first():
    pb = Playbook.parse(SEED)
    pb.sections["Notes"] = ["x" * 5000]
    pb.sections["Rules"] = ["important rule"]
    pb.enforce_limits(max_chars=200, max_rules=12)
    out = pb.serialize()
    assert len(out) <= 400            # notes trimmed; rules kept
    assert "important rule" in out


def test_enforce_max_rules_keeps_first_n():
    pb = Playbook.parse(SEED)
    pb.sections["Rules"] = [f"rule {i}" for i in range(20)]
    pb.enforce_limits(max_chars=10000, max_rules=12)
    assert len(pb.sections["Rules"]) == 12
