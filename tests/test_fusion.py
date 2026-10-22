import math

import pytest

from kycgraph.risk import PRIOR, fuse, logit, sigmoid, source_weight
from kycgraph.typologies import Evidence


def _ev(kind, lr, conf=1.0, sources=("lists",)):
    return Evidence(kind, "X", "", lr, conf, list(sources))


def test_no_evidence_returns_prior():
    assert fuse([], {}) == pytest.approx(PRIOR)


def test_single_item_adds_discounted_log_lr():
    rel = {"lists": 0.95}
    risk = fuse([_ev("screening_sanctions", 10.0, 0.8)], rel)
    assert logit(risk) == pytest.approx(logit(PRIOR) + 0.8 * 0.95 * math.log(10))


def test_same_kind_counts_once_and_strongest_wins():
    rel = {"lists": 0.95}
    one = fuse([_ev("pep_supplier", 8.0)], rel)
    two = fuse([_ev("pep_supplier", 8.0), _ev("pep_supplier", 4.0)], rel)
    assert one == pytest.approx(two)


def test_independent_kinds_add_and_exculpatory_evidence_subtracts():
    rel = {"lists": 0.95, "model": 0.5}
    a = fuse([_ev("a", 5.0)], rel)
    ab = fuse([_ev("a", 5.0), _ev("b", 5.0)], rel)
    down = fuse([_ev("a", 5.0), _ev("m", 0.2, sources=("model",))], rel)
    assert ab > a > down


def test_source_weight_is_noisy_or():
    rel = {"A": 0.8, "C": 0.6}
    assert source_weight(["A"], rel) == pytest.approx(0.8)
    assert source_weight(["A", "C"], rel) == pytest.approx(1 - 0.2 * 0.4)
    assert source_weight(["unknown"], rel) == pytest.approx(0.5)


def test_sigmoid_logit_roundtrip():
    assert sigmoid(logit(0.37)) == pytest.approx(0.37)


def test_contributions_are_recorded_for_explanation():
    items = [_ev("a", 5.0, 0.5)]
    fuse(items, {"lists": 1.0})
    assert items[0].contribution == pytest.approx(0.5 * math.log(5))
