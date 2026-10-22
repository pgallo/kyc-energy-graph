import numpy as np
import pytest

from kycgraph import resolution
from kycgraph.model import CompanyRecord, Dataset, PersonRecord, Truth
from kycgraph.resolution import (
    COMPANY_FEATURES,
    PERSON_FEATURES,
    Prepared,
    blocking_keys,
    candidate_pairs,
    cannot_link,
    cluster,
    features,
    labelled_pairs,
    pairwise_metrics,
    prepare,
)


def _ds(companies=(), persons=()):
    return Dataset(0, "2026-06-30", list(companies), list(persons), [], [], [], [], Truth())


def _person(key, source, name, dob=None, nid=None):
    return Prepared(key, "person", source, name.split(), set(name.split()), set(), {"nid": nid} if nid else {}, dob=dob)


def test_transliteration_and_typo_variants_share_a_block():
    prep = prepare(_ds(persons=[
        PersonRecord("REG:1", "REG", "Galen Zhelanin", "1986-05-16"),
        PersonRecord("C:1", "C", "Гален Желанин", "1986-05-16"),
        PersonRecord("B:1", "B", "ZHELANNIN, Galen", "1986"),
    ]))
    keys = [blocking_keys(p) for p in prep.values()]
    assert keys[0] & keys[1] and keys[0] & keys[2]
    pairs, _ = candidate_pairs(prep)
    assert pairs == {("B:1", "C:1"), ("B:1", "REG:1"), ("C:1", "REG:1")}


def test_same_source_pairs_are_never_candidates():
    prep = prepare(_ds(companies=[
        CompanyRecord("A:1", "A", "Ulmesk Solar Ltd"), CompanyRecord("A:2", "A", "Ulmesk Solar AD"),
    ]))
    assert candidate_pairs(prep)[0] == set()


def test_oversized_blocks_are_skipped(monkeypatch):
    monkeypatch.setattr(resolution, "MAX_BLOCK", 2)
    prep = prepare(_ds(companies=[CompanyRecord(f"{s}:1", s, "Ulmesk Solar") for s in ("A", "B", "C")]))
    pairs, skipped = candidate_pairs(prep)
    assert pairs == set() and skipped >= 1


def test_feature_vector_is_explainable_and_ordered():
    prep = prepare(_ds(companies=[
        CompanyRecord("REG:VR-000123", "REG", "Kvaravel Grid Contractors Ltd", "VA", "12 Lipova Street, Velmira",
                      reg_no="VR-000123", incorporated="2001-02-03"),
        CompanyRecord("A:1", "A", "KVARAVEL GRID CONTRACTORS", "VA", "12 Lipova St., Velmira",
                      reg_no="000123", incorporated="2001-02-03"),
    ]))
    x = features(prep["REG:VR-000123"], prep["A:1"])
    named = dict(zip(COMPANY_FEATURES, x, strict=True))
    assert named["name_sim"] == 1.0 and named["id_agree"] == 1.0 and named["inc_agree"] == 1.0
    assert len(PERSON_FEATURES) == len(features(_person("a", "A", "x y"), _person("b", "B", "x y")))


def test_cannot_link_rules():
    assert cannot_link(_person("a", "A", "x"), _person("b", "A", "x"))
    assert cannot_link(_person("a", "A", "x", nid="1"), _person("b", "B", "x", nid="2"))
    assert cannot_link(_person("a", "A", "x", dob="1970-01-01"), _person("b", "B", "x", dob="1980-01-01"))
    assert not cannot_link(_person("a", "A", "x", dob="1970"), _person("b", "B", "x", dob="1970-01-01"))


def test_clustering_cuts_weakest_edge_between_conflicting_records():
    # Two registry persons with the same name, and one feed record linked to both.
    prep = {k: _person(k, k.split(":")[0], "edin nedova", dob) for k, dob in
            [("REG:1", "1950-01-08"), ("REG:2", "1962-04-30"), ("A:1", None), ("B:1", "1950-01-08")]}
    probs = {("A:1", "REG:1"): 0.95, ("A:1", "REG:2"): 0.60, ("B:1", "REG:1"): 0.99, ("A:1", "B:1"): 0.9}
    clusters, cuts = cluster(list(prep), probs, prep)
    assert cuts == 1
    assert {"REG:1", "A:1", "B:1"} in clusters and {"REG:2"} in clusters


def test_clustering_without_conflicts_is_transitive_closure():
    prep = {k: _person(k, k.split(":")[0], "x") for k in ("REG:1", "A:1", "B:1")}
    clusters, cuts = cluster(list(prep), {("A:1", "REG:1"): 0.9, ("A:1", "B:1"): 0.8}, prep)
    assert cuts == 0 and clusters == [{"REG:1", "A:1", "B:1"}]


def test_pairwise_metrics_hand_example():
    truth = {"a": "1", "b": "1", "c": "1", "d": "2"}
    m = pairwise_metrics([{"a", "b"}, {"c", "d"}], truth)
    # predicted pairs: ab (correct), cd (wrong); true pairs: ab, ac, bc
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(1 / 3)


def test_both_match_models_are_calibrated_and_discriminative(engine, dataset):
    for kind in ("company", "person"):
        X, y, _ = labelled_pairs(dataset, kind)
        for model in engine.models[kind].values():
            p = model.proba(X)
            brier = float(np.mean((p - y) ** 2))
            assert brier < 0.02, (kind, model.name, brier)
            assert p[y == 1].mean() > 0.9 and p[y == 0].mean() < 0.05


def test_explanations_rank_features_by_effect(engine, dataset):
    X, y, _ = labelled_pairs(dataset, "person")
    model = engine.models["person"]["logistic"]
    expl = model.explain(X[np.argmax(y)], PERSON_FEATURES)
    assert {n for n, _, _ in expl} == set(PERSON_FEATURES)
    effects = [abs(d) for _, _, d in expl]
    assert effects == sorted(effects, reverse=True)


def test_resolution_metrics_and_model_comparison(analysis):
    res = analysis.res
    for kind in ("company", "person"):
        assert res.metrics[kind]["f1"] > 0.95
        assert res.metrics[kind]["pair_completeness"] > 0.94
        assert res.metrics[kind]["reduction_ratio"] > 0.9
    models = {(r["kind"], r["model"], r["calibration"]) for r in res.comparison}
    assert ("person", "mlp", "platt") in models and ("company", "logistic", "none") in models
    # the noisiest feed must come out as the least reliable
    assert res.reliability["A"] > res.reliability["C"]
