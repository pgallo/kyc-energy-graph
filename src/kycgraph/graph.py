"""Ownership and control graph over resolved entities.

Effective ownership uses the integrated-ownership formulation: with A[i, j] the share of j
held directly by i, the sum over all ownership walks is T = A (I - A)^-1. In an acyclic
structure this equals the familiar sum of path products; with cross-holdings it
redistributes the circular part to the ultimate owners instead of losing it. A bounded
simple-path variant is kept as a cross-check and for explanations.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from .model import Dataset
from .resolution import Entity, Resolution

ROLE_WEIGHT = {"ubo": 1.0, "controller": 1.0, "director": 0.6, "parent_director": 0.5,
               "signatory": 0.4, "senior_managing_official": 0.6}


def integrated_ownership(edges: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    """All-walks effective ownership T = A (I - A)^-1 for shares given as fractions."""
    nodes = sorted({n for e in edges for n in e})
    idx = {n: i for i, n in enumerate(nodes)}
    a = np.zeros((len(nodes), len(nodes)))
    for (u, v), f in edges.items():
        a[idx[u], idx[v]] += f
    if nodes and max(abs(np.linalg.eigvals(a))) >= 1 - 1e-9:
        raise ValueError("ownership structure is closed (spectral radius >= 1); no ultimate owners")
    t = a @ np.linalg.inv(np.eye(len(nodes)) - a)
    return {(u, v): float(t[idx[u], idx[v]]) for u in nodes for v in nodes
            if u != v and t[idx[u], idx[v]] > 1e-9}


def simple_path_ownership(g: nx.DiGraph, owner: str, company: str, max_depth: int = 10) -> float:
    """Sum of share products over simple paths only (a cycle is traversed at most once)."""
    total = 0.0
    for path in nx.all_simple_paths(g, owner, company, cutoff=max_depth):
        prod = 1.0
        for u, v in itertools.pairwise(path):
            prod *= g[u][v]["pct"]
        total += prod
    return total


@dataclass
class KeyPerson:
    person: str
    name: str
    roles: list[str]
    effective: float
    influence: float
    betweenness: float
    bridge: bool
    companies: int
    nominee_pattern: bool
    paths: list[list[str]] = field(default_factory=list)


class CorporateGraph:
    def __init__(self, entities: dict[str, Entity], own: nx.DiGraph, roles: nx.DiGraph) -> None:
        """``own`` edges carry pct (fraction), sources, spread and records; ``roles`` edges carry roles."""
        self.entities = entities
        self.own = own
        self.roles = roles
        self._normalise_overcommitted()
        self.effective = integrated_ownership({(u, v): d["pct"] for u, v, d in self.own.edges(data=True)})
        undirected = nx.Graph()
        undirected.add_nodes_from(entities)
        undirected.add_edges_from(self.own.edges)
        undirected.add_edges_from(self.roles.edges)
        self.betweenness = nx.betweenness_centrality(undirected, k=min(300, len(undirected)), seed=0)
        self.articulation = set(nx.articulation_points(undirected))
        self._control_cache: dict[str, set[str]] = {}

    @classmethod
    def build(cls, ds: Dataset, res: Resolution) -> CorporateGraph:
        """Lift source relations onto resolved entities and fuse the stated shares."""
        own, roles = nx.DiGraph(), nx.DiGraph()
        own.add_nodes_from(res.entities)
        roles.add_nodes_from(res.entities)
        grouped: dict[tuple[str, str], list] = defaultdict(list)
        for rel in ds.relations:
            u, v = res.of.get(rel.src), res.of.get(rel.dst)
            if u is None or v is None or u == v:
                continue
            if rel.kind == "owns":
                grouped[(u, v)].append(rel)
            else:
                data = roles.get_edge_data(u, v) or {"roles": set(), "sources": set(), "records": []}
                data["roles"].add(rel.kind)
                data["sources"].add(rel.source)
                data["records"].append(f"{rel.src}->{rel.dst}")
                roles.add_edge(u, v, **data)
        for (u, v), rels in grouped.items():
            # One value per source (a source states an edge once), weighted by source reliability.
            by_source = {r.source: r.pct for r in rels}
            weights = {s: res.reliability.get(s, 0.5) for s in by_source}
            pct = sum(by_source[s] * weights[s] for s in by_source) / sum(weights.values())
            if "REG" in by_source:
                pct = by_source["REG"]  # the registry is authoritative where it speaks
            own.add_edge(u, v, pct=pct / 100, sources=set(by_source),
                         spread=max(by_source.values()) - min(by_source.values()),
                         records=[f"{r.src}->{r.dst}" for r in rels])
        return cls(res.entities, own, roles)

    def _normalise_overcommitted(self) -> None:
        """Noisy feeds can make declared stakes sum above 100%; rescale those companies."""
        for c in self.own.nodes:
            total = sum(d["pct"] for _, _, d in self.own.in_edges(c, data=True))
            if total > 1.0:
                for _u, _, d in self.own.in_edges(c, data=True):
                    d["pct"] /= total

    def is_person(self, eid: str) -> bool:
        return self.entities[eid].kind == "person"

    def offshore(self, eid: str) -> bool:
        return self.entities[eid].attrs.get("jurisdiction") == "MR"

    def controlled_by(self, person: str) -> set[str]:
        """Companies where the person commands >50% of votes directly or via controlled companies."""
        if person in self._control_cache:
            return self._control_cache[person]
        controlled: set[str] = set()
        reachable = nx.descendants(self.own, person)
        changed = True
        while changed:
            changed = False
            for c in reachable - controlled:
                votes = sum(d["pct"] for u, _, d in self.own.in_edges(c, data=True)
                            if u == person or u in controlled)
                if votes > 0.5:
                    controlled.add(c)
                    changed = True
        self._control_cache[person] = controlled
        return controlled

    def ownership_paths(self, owner: str, company: str, max_depth: int = 6) -> list[list[str]]:
        return list(nx.all_simple_paths(self.own, owner, company, cutoff=max_depth))

    def cycles(self) -> list[list[str]]:
        companies = [n for n in self.own if not self.is_person(n)]
        return [c for c in nx.simple_cycles(self.own.subgraph(companies)) if len(c) > 1]

    def nominee_pattern(self, person: str, min_boards: int = 5) -> bool:
        boards = sum(1 for _, _, d in self.roles.out_edges(person, data=True) if "director" in d["roles"])
        return boards >= min_boards and self.own.out_degree(person) == 0

    def key_persons(self, company: str, threshold: float = 0.25) -> list[KeyPerson]:
        roles: dict[str, set[str]] = defaultdict(set)
        for anc in nx.ancestors(self.own, company):
            if self.is_person(anc):
                if self.effective.get((anc, company), 0.0) >= threshold:
                    roles[anc].add("ubo")
                if company in self.controlled_by(anc):
                    roles[anc].add("controller")
        for p, _, d in self.roles.in_edges(company, data=True):
            roles[p] |= {"director"} & d["roles"] | {"signatory"} & d["roles"]
        for parent, _, d in self.own.in_edges(company, data=True):
            if not self.is_person(parent) and d["pct"] > 0.5:
                for p, _, rd in self.roles.in_edges(parent, data=True):
                    if "director" in rd["roles"]:
                        roles[p].add("parent_director")
        if not any("ubo" in r or "controller" in r for r in roles.values()):
            for r in roles.values():  # no owner identified: fall back to senior managing officials
                if "director" in r:
                    r.add("senior_managing_official")
        out = []
        for p, r in roles.items():
            eff = self.effective.get((p, company), 0.0)
            influence = max([ROLE_WEIGHT[x] for x in r] + [min(1.0, 2 * eff)])
            out.append(KeyPerson(
                p, self.entities[p].name, sorted(r), eff, influence, self.betweenness.get(p, 0.0),
                p in self.articulation, self.roles.out_degree(p) + self.own.out_degree(p),
                self.nominee_pattern(p), self.ownership_paths(p, company) if eff > 0 else [],
            ))
        return sorted(out, key=lambda k: (-k.influence, -k.effective, k.person))

    def ownership_tree(self, company: str, depth: int = 6) -> list[str]:
        lines = [f"{company}  {self.entities[company].name}"]

        def walk(node: str, prefix: str, seen: frozenset, level: int) -> None:
            parents = sorted(self.own.in_edges(node, data=True), key=lambda e: -e[2]["pct"])
            for i, (u, _, d) in enumerate(parents):
                last = i == len(parents) - 1
                branch = "└── " if last else "├── "
                e = self.entities[u]
                tag = f" [{e.attrs.get('jurisdiction') or e.attrs.get('nationality') or '?'}]"
                eff = self.effective.get((u, company), 0.0)
                src = ",".join(sorted(d["sources"]))
                cyc = "  (cycle)" if u in seen else ""
                lines.append(f"{prefix}{branch}{d['pct']:.1%} {u}  {e.name}{tag}  eff {eff:.1%}  src {src}{cyc}")
                if u not in seen and level < depth:
                    walk(u, prefix + ("    " if last else "│   "), seen | {u}, level + 1)

        walk(company, "", frozenset({company}), 1)
        return lines
