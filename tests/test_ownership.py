import networkx as nx
import pytest

from kycgraph.graph import CorporateGraph, integrated_ownership, simple_path_ownership
from kycgraph.resolution import Entity


def _entity(eid, kind="company", jurisdiction="VA"):
    return Entity(eid, kind, [f"REG:{eid}"], eid, {"jurisdiction": jurisdiction}, {"REG"}, 1.0, None)


def _graph(owns, roles=(), persons=(), offshore=()):
    nodes = {n for e in owns for n in e[:2]} | {r[0] for r in roles} | {r[1] for r in roles}
    entities = {n: _entity(n, "person" if n in persons else "company", "MR" if n in offshore else "VA") for n in nodes}
    own, rg = nx.DiGraph(), nx.DiGraph()
    own.add_nodes_from(entities)
    rg.add_nodes_from(entities)
    for u, v, f in owns:
        own.add_edge(u, v, pct=f, sources={"REG"}, spread=0.0, records=[])
    for p, c, r in roles:
        rg.add_edge(p, c, roles={r}, sources={"REG"}, records=[])
    return CorporateGraph(entities, own, rg)


def test_chain_is_product_of_shares():
    eff = integrated_ownership({("P", "H1"): 0.8, ("H1", "H2"): 0.75, ("H2", "T"): 0.5})
    assert eff[("P", "T")] == pytest.approx(0.3)
    assert eff[("H1", "T")] == pytest.approx(0.375)


def test_diamond_sums_paths():
    eff = integrated_ownership({("P", "A"): 1.0, ("P", "B"): 1.0, ("A", "T"): 0.2, ("B", "T"): 0.15})
    assert eff[("P", "T")] == pytest.approx(0.35)


def test_cycle_hand_checked():
    # P owns 60% of A, Q owns 50% of B, A owns 50% of B, B owns 40% of A.
    edges = {("P", "A"): 0.6, ("Q", "B"): 0.5, ("A", "B"): 0.5, ("B", "A"): 0.4}
    eff = integrated_ownership(edges)
    # walks P->A->B->A... : 0.6 / (1 - 0.5 * 0.4) = 0.75
    assert eff[("P", "A")] == pytest.approx(0.75)
    assert eff[("Q", "A")] == pytest.approx(0.25)
    assert eff[("P", "A")] + eff[("Q", "A")] == pytest.approx(1.0)  # cross-holding redistributed to people
    g = nx.DiGraph()
    g.add_edges_from((u, v, {"pct": f}) for (u, v), f in edges.items())
    assert simple_path_ownership(g, "P", "A") == pytest.approx(0.6)
    assert simple_path_ownership(g, "Q", "A") == pytest.approx(0.2)


def test_closed_structure_is_rejected():
    with pytest.raises(ValueError):
        integrated_ownership({("A", "B"): 1.0, ("B", "A"): 1.0})


def test_ubo_threshold_and_control():
    g = _graph([("P", "H", 0.6), ("H", "T", 0.4), ("Q", "T", 0.3), ("R", "T", 0.3), ("S", "H", 0.4)],
               roles=[("D", "T", "director"), ("E", "H", "director")], persons={"P", "Q", "R", "S", "D", "E"})
    kps = {k.person: k for k in g.key_persons("T", threshold=0.25)}
    assert kps["Q"].effective == pytest.approx(0.3) and "ubo" in kps["Q"].roles
    assert "P" not in kps  # 0.6 x 0.4 = 24%, below the 25% threshold and no control
    lower = {k.person: k for k in g.key_persons("T", threshold=0.2)}
    assert lower["P"].effective == pytest.approx(0.24) and "ubo" in lower["P"].roles
    assert "director" in kps["D"].roles
    assert "S" not in kps  # 16% indirect, no control
    # H holds only 40% of T, so H's director has no parent control over T
    assert "E" not in kps


def test_control_through_majority_chain():
    g = _graph([("P", "H", 0.51), ("H", "T", 0.51), ("Q", "H", 0.49), ("R", "T", 0.49)],
               roles=[("E", "H", "director")], persons={"P", "Q", "R", "E"})
    kps = {k.person: k for k in g.key_persons("T")}
    assert g.controlled_by("P") == {"H", "T"}
    assert "controller" in kps["P"].roles and "ubo" in kps["P"].roles
    assert "parent_director" in kps["E"].roles


def test_senior_managing_official_fallback_when_no_owner():
    g = _graph([("S", "T", 1.0)], roles=[("D", "T", "director")], persons={"D"})
    kps = g.key_persons("T")
    assert kps[0].roles == ["director", "senior_managing_official"]


def test_cycle_detection_and_tree_marks_cycle():
    g = _graph([("A", "B", 0.3), ("B", "C", 0.35), ("C", "A", 0.4), ("P", "A", 0.6)], persons={"P"})
    assert sorted(map(sorted, g.cycles())) == [["A", "B", "C"]]
    assert any("(cycle)" in line for line in g.ownership_tree("A"))


def test_overcommitted_shares_are_rescaled():
    g = _graph([("P", "T", 0.7), ("Q", "T", 0.5)], persons={"P", "Q"})
    assert sum(d["pct"] for _, _, d in g.own.in_edges("T", data=True)) == pytest.approx(1.0)
