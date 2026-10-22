import pytest

from kycgraph.model import TYPOLOGIES
from kycgraph.pipeline import true_id
from kycgraph.typologies import DETECTORS


def _flagged(analysis, kind):
    return {true_id(analysis.res, analysis.ds, e.subject) for e in analysis.evidence if e.kind == kind}


def _truth(analysis, kind):
    return {c for case in analysis.ds.truth.typologies[kind] for c in case["companies"]}


@pytest.mark.parametrize("kind", TYPOLOGIES)
def test_detector_matches_ground_truth(analysis, kind):
    flagged, truth = _flagged(analysis, kind), _truth(analysis, kind)
    assert truth, "generator must inject the typology"
    recall = len(flagged & truth) / len(truth)
    precision = len(flagged & truth) / len(flagged) if flagged else 0.0
    assert recall >= 0.66 and precision >= 0.6, (kind, sorted(flagged - truth), sorted(truth - flagged))


@pytest.mark.parametrize("kind", TYPOLOGIES)
def test_evidence_carries_its_trail(analysis, kind):
    items = [e for e in analysis.evidence if e.kind == kind]
    assert items
    for e in items:
        assert e.lr > 1 and 0 < e.confidence <= 1
        assert e.records and e.sources and e.summary


def test_detectors_are_registered_for_every_typology():
    assert set(DETECTORS) == set(TYPOLOGIES)


def test_hidden_owner_chain_goes_through_offshore_layer(analysis):
    g = analysis.ctx.graph
    for e in (x for x in analysis.evidence if x.kind == "hidden_ubo"):
        nodes = [edge.split(" ")[0] for edge in e.edges]
        assert len(e.edges) >= 3
        assert any(g.offshore(n) for n in nodes[1:])


def test_sanctioned_exposure_stays_within_three_hops(analysis):
    for e in (x for x in analysis.evidence if x.kind == "sanctioned_exposure"):
        assert 1 <= len(e.edges) <= 3


def test_decoys_are_not_flagged(analysis):
    """Transparent foreign groups and the busy business centre must not trigger detectors."""
    ds = analysis.ds
    flagged = set().union(*(_flagged(analysis, k) for k in ("hidden_ubo", "shell_cluster")))
    truth = set().union(*(_truth(analysis, k) for k in ("hidden_ubo", "shell_cluster")))
    assert flagged <= truth
    assert ds.tenders  # front-company decoys rely on awards long after incorporation
