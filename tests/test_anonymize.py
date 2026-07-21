from vbp.anonymize import anonymize_teams


def test_consistent_mapping_within_season():
    m = ["Alpha", "Beta", "Alpha", "Gamma"]
    ids, mapping = anonymize_teams(m)
    assert ids[0] == ids[2]            # Alpha -> same id both times
    assert ids[0] != ids[1]
    assert set(mapping.values()) == set(ids)
    assert all(v.startswith("Team_") for v in mapping.values())


def test_mapping_is_deterministic_given_order():
    a, _ = anonymize_teams(["X", "Y", "Z"])
    b, _ = anonymize_teams(["X", "Y", "Z"])
    assert a == b
