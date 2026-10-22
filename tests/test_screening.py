from types import SimpleNamespace

from kycgraph.model import ListEntry, PersonRecord
from kycgraph.resolution import Entity
from kycgraph.screening import HIT, list_name_similarity, score, screen


def _world(*persons):
    records = {p.key: p for p in persons}
    entities = {p.key: Entity(p.key, "person", [p.key], p.name, {}, {p.source}, 1.0, None) for p in persons}
    # padding population so token commonness is meaningful
    for i in range(30):
        key = f"REG:pad{i}"
        records[key] = PersonRecord(key, "REG", f"Arven Filler{i}")
        entities[key] = Entity(key, "person", [key], f"Arven Filler{i}", {}, {"REG"}, 1.0, None)
    return SimpleNamespace(entities=entities), records


SANCTIONED = ListEntry("SL-1", "sanctions", "ZORAN KOSTENENKO", ["Зоран Костененко"], "1983-07-14", "KS", "Asset freeze")


def test_exact_name_and_birth_date_is_a_hit():
    res, recs = _world(PersonRecord("A:1", "A", "Zoran Kostenenko", "1983-07-14", "KS"))
    hits = screen(res, [SANCTIONED], recs)
    assert [(h.person, h.decision) for h in hits] == [("A:1", "hit")]


def test_cyrillic_and_variant_spelling_still_match():
    res, recs = _world(PersonRecord("C:1", "C", "Зоран Костененко", "1983-07-14"),
                       PersonRecord("B:1", "B", "KOSTENENKO, Zoran", "1983"))
    hits = {h.person: h for h in screen(res, [SANCTIONED], recs)}
    assert hits["C:1"].decision == "hit"
    assert hits["B:1"].confidence >= HIT


def test_namesake_with_different_birth_date_is_suppressed():
    res, recs = _world(PersonRecord("REG:9", "REG", "Zoran Kostenenko", "1961-02-02", "VA"))
    assert screen(res, [SANCTIONED], recs) == []


def test_shared_given_name_with_different_surname_is_not_a_match():
    assert list_name_similarity(["boris", "radenski"], ["boris", "nedarets"]) < 0.85
    res, recs = _world(PersonRecord("A:2", "A", "Zoran Kostarets"))
    assert all(h.decision != "hit" for h in screen(res, [SANCTIONED], recs))


def test_score_is_monotone_in_evidence():
    assert score(0.95, 1, 1, 0.2) > score(0.95, 0, 1, 0.2) > score(0.95, -1, 1, 0.2)
    assert score(0.95, 0, 0, 0.9) < score(0.95, 0, 0, 0.1)  # common names need more support


def test_screening_against_ground_truth(analysis):
    from kycgraph.evaluate import screening_metrics

    m = screening_metrics(analysis)
    assert m["precision"] >= 0.9 and m["recall"] >= 0.9
