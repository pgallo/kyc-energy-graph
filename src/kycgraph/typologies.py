"""Rule-based typology detectors. Each one emits Evidence carrying its own record trail.

Likelihood ratios are expert-set and documented next to each detector; they express how
much more likely the observed pattern is under "illicit" than under "legitimate".
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

import networkx as nx

from .graph import CorporateGraph
from .model import Dataset
from .normalize import address_tokens, company_tokens, person_tokens
from .resolution import Resolution
from .screening import ScreeningHit
from .similarity import name_similarity

FRONT_WINDOW_DAYS = 180
SANCTION_HOPS = 3
MEDIA_SEVERITY = {"sanctions evasion": 4, "money laundering": 4, "bribery": 3, "fraud": 3,
                  "irregularities": 2, "inquiry": 1}


@dataclass
class Evidence:
    kind: str
    subject: str
    summary: str
    lr: float
    confidence: float
    sources: list[str]
    records: list[str] = field(default_factory=list)
    edges: list[str] = field(default_factory=list)
    contribution: float = 0.0  # filled in by fusion

    @property
    def log_lr(self) -> float:
        return math.log(self.lr)


@dataclass
class Context:
    ds: Dataset
    res: Resolution
    graph: CorporateGraph
    hits: list[ScreeningHit]
    awards: dict[str, list] = field(default_factory=dict)  # supplier entity -> tenders
    names: dict[str, str] = field(default_factory=dict)    # mention key -> name as written

    def __post_init__(self) -> None:
        self.names = {r.key: r.name for r in [*self.ds.companies, *self.ds.persons]}
        if not self.awards:
            awards = defaultdict(list)
            for t in self.ds.tenders:
                awards[self.res.of[t.supplier]].append(t)
            self.awards = dict(awards)

    def edge_text(self, u: str, v: str) -> str:
        d = self.graph.own[u][v]
        return f"{u} -[{d['pct']:.0%} via {','.join(sorted(d['sources']))}]-> {v}"

    def edge_conf(self, u: str, v: str) -> float:
        d = self.graph.own[u][v]
        rel = max(self.res.reliability.get(s, 0.5) for s in d["sources"])
        return rel * max(0.3, 1 - d["spread"] / 30)

    def path_conf(self, path: list[str]) -> float:
        """Weakest edge on the chain. Entity-resolution doubt is deliberately left out here:
        it belongs to the assessment's confidence, not to the strength of the evidence."""
        return min((self.edge_conf(u, v) for u, v in itertools.pairwise(path)), default=1.0)

    def records(self, *entities: str) -> list[str]:
        return [k for e in entities for k in self.res.entities[e].members]


def _companies(ctx: Context) -> list[str]:
    return sorted(e for e, v in ctx.res.entities.items() if v.kind == "company")


def detect_hidden_ubo(ctx: Context, threshold: float = 0.25) -> list[Evidence]:
    """UBO reached through >= 3 layers including an offshore vehicle. LR 10, 20 with nominee directors."""
    g, out = ctx.graph, []
    for c in _companies(ctx):
        nominees = [p for p, _, d in g.roles.in_edges(c, data=True) if "director" in d["roles"] and g.nominee_pattern(p)]
        for p in (a for a in g.own.nodes if g.is_person(a)):
            if g.effective.get((p, c), 0.0) < threshold:
                continue
            for path in g.ownership_paths(p, c):
                if len(path) - 1 >= 3 and any(g.offshore(n) for n in path[1:-1]):
                    lr = 20.0 if nominees else 10.0
                    out.append(Evidence(
                        "hidden_ubo", c,
                        f"Beneficial owner {p} ({g.entities[p].name}) holds {g.effective[(p, c)]:.0%} through "
                        f"{len(path) - 1} layers incl. offshore vehicles" + (f"; nominee directors {nominees}" if nominees else ""),
                        lr, ctx.path_conf(path), sorted({s for u, v in itertools.pairwise(path)
                                                          for s in g.own[u][v]["sources"]}),
                        ctx.records(*path), [ctx.edge_text(u, v) for u, v in itertools.pairwise(path)]))
                    break
    return out


def detect_circular(ctx: Context) -> list[Evidence]:
    """Company-only ownership cycles. LR 8."""
    out = []
    for cyc in ctx.graph.cycles():
        ring = [*cyc, cyc[0]]
        edges = [ctx.edge_text(u, v) for u, v in itertools.pairwise(ring)]
        conf = ctx.path_conf(ring)
        srcs = sorted({s for u, v in itertools.pairwise(ring) for s in ctx.graph.own[u][v]["sources"]})
        for c in cyc:
            out.append(Evidence("circular_ownership", c, f"Member of an ownership cycle of {len(cyc)} companies",
                                8.0, conf, srcs, ctx.records(*cyc), edges))
    return out


def detect_pep_supplier(ctx: Context) -> list[Evidence]:
    """Supplier with tender awards whose key person screens as a PEP. LR 8, 16 if the PEP's body is the buyer."""
    out = []
    peps = defaultdict(list)
    for h in ctx.hits:
        if h.list_name == "pep" and h.decision == "hit":
            peps[h.person].append(h)
    for supplier, tenders in sorted(ctx.awards.items()):
        for kp in ctx.graph.key_persons(supplier):
            for h in peps.get(kp.person, []):
                buyers = {ctx.res.of[t.buyer] for t in tenders}
                linked = h.organisation and any(
                    name_similarity(company_tokens(h.organisation), company_tokens(ctx.res.entities[b].name)) > 0.9
                    for b in buyers)
                out.append(Evidence(
                    "pep_supplier", supplier,
                    f"{kp.name} ({', '.join(kp.roles)}) matches PEP {h.list_id} '{h.note}, {h.organisation}'; "
                    f"supplier won {len(tenders)} award(s)" + ("; PEP body is the awarding buyer" if linked else ""),
                    16.0 if linked else 8.0, h.confidence, ["lists", "tenders"],
                    ctx.records(kp.person) + [t.tender_id for t in tenders], [f"{kp.person} -[{','.join(kp.roles)}]-> {supplier}"]))
    return out


def detect_shell_cluster(ctx: Context, min_size: int = 3) -> list[Evidence]:
    """>= 3 low-substance companies at one address sharing officers. LR 8."""
    as_of = date.fromisoformat(ctx.ds.as_of)
    by_address = defaultdict(list)
    for c in _companies(ctx):
        addr = ctx.res.entities[c].attrs.get("address")
        if addr:
            by_address[" ".join(sorted(address_tokens(addr)))].append(c)
    out = []
    for addr, members in by_address.items():
        if len(members) < min_size:
            continue
        officers = {c: {p for p, _ in ctx.graph.roles.in_edges(c)} for c in members}

        def thin(c: str) -> bool:
            a = ctx.res.entities[c].attrs
            inc = a.get("incorporated")
            young = inc is not None and (as_of - date.fromisoformat(inc)).days < 3 * 365
            return (a.get("employees") or 0) <= 2 and young

        flagged = [c for c in members if thin(c) and any(officers[c] & officers[o] for o in members if o != c)]
        if len(flagged) < min_size:
            continue
        shared = sorted(set.intersection(*(officers[c] for c in flagged)) or set.union(*(officers[c] for c in flagged)))
        for c in flagged:
            out.append(Evidence(
                "shell_cluster", c,
                f"{len(flagged)} low-substance companies registered at '{addr}' share officers {shared[:3]}",
                8.0, ctx.res.reliability.get("REG", 0.9), ["REG"],
                ctx.records(*flagged), [f"{p} -[officer]-> {c}" for p in sorted(officers[c])]))
    return out


def detect_sanctioned_exposure(ctx: Context) -> list[Evidence]:
    """Companies within 3 ownership hops of a sanctions match. LR 20 if >= 50% or controlled, else 10."""
    g, out = ctx.graph, []
    for h in ctx.hits:
        if h.list_name != "sanctions" or h.decision != "hit":
            continue
        reach = nx.single_source_shortest_path(g.own, h.person, cutoff=SANCTION_HOPS)
        for c, path in reach.items():
            if c == h.person:
                continue
            eff = g.effective.get((h.person, c), 0.0)
            control = c in g.controlled_by(h.person)
            out.append(Evidence(
                "sanctioned_exposure", c,
                f"{h.person} ({g.entities[h.person].name}) matches {h.list_id} (conf {h.confidence:.2f}); "
                f"{len(path) - 1} hop(s), effective {eff:.0%}" + (", controlling" if control else ""),
                20.0 if eff >= 0.5 or control else 10.0, h.confidence * ctx.path_conf(path),
                sorted({"lists", *(s for u, v in itertools.pairwise(path) for s in g.own[u][v]["sources"])}),
                ctx.records(*path), [ctx.edge_text(u, v) for u, v in itertools.pairwise(path)]))
    return out


def detect_front_company(ctx: Context) -> list[Evidence]:
    """Award within 180 days of incorporation. LR 4, 10 with <= 2 employees."""
    out = []
    for supplier, tenders in sorted(ctx.awards.items()):
        attrs = ctx.res.entities[supplier].attrs
        if not attrs.get("incorporated"):
            continue
        inc = date.fromisoformat(attrs["incorporated"])
        for t in tenders:
            days = (date.fromisoformat(t.award_date) - inc).days
            if 0 <= days <= FRONT_WINDOW_DAYS:
                staff = attrs.get("employees")
                thin = staff is not None and staff <= 2
                out.append(Evidence(
                    "front_company", supplier,
                    f"Incorporated {inc}, won {t.tender_id} ({t.title}) {days} days later; employees {staff}",
                    10.0 if thin else 4.0, ctx.res.reliability.get("REG", 0.9), ["REG", "tenders"],
                    [*ctx.records(supplier), t.tender_id], [f"{ctx.res.of[t.buyer]} -[award {t.tender_id}]-> {supplier}"]))
                break
    return out


def link_media(ctx: Context) -> list[Evidence]:
    """Adverse-media signal: entity name found in a snippet with severity keywords."""
    names: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for e in ctx.res.entities.values():
        for name in {ctx.names[k] for k in e.members}:
            toks = tuple(company_tokens(name) if e.kind == "company" else person_tokens(name))
            if len(toks) >= 2 or (e.kind == "company" and len(toks[0]) > 6):
                names[e.id].add(toks)
                if e.kind == "person":
                    names[e.id].add(tuple(reversed(toks)))
    out = []
    for item in ctx.ds.media:
        text = " ".join(person_tokens(item.text))
        severity = max((v for k, v in MEDIA_SEVERITY.items() if k in item.text.lower()), default=0)
        if severity == 0:
            continue
        for eid, variants in names.items():
            if any(f" {' '.join(v)} " in f" {text} " for v in variants):
                out.append(Evidence("adverse_media", eid, f"{item.media_id} ({item.date}): \"{item.text}\"",
                                    {1: 1.5, 2: 2.0, 3: 3.0, 4: 5.0}[severity], 0.8, ["media"], [item.media_id]))
    return out


DETECTORS = {
    "hidden_ubo": detect_hidden_ubo,
    "circular_ownership": detect_circular,
    "pep_supplier": detect_pep_supplier,
    "shell_cluster": detect_shell_cluster,
    "sanctioned_exposure": detect_sanctioned_exposure,
    "front_company": detect_front_company,
}


def run_detectors(ctx: Context) -> list[Evidence]:
    evidence = [e for detect in DETECTORS.values() for e in detect(ctx)]
    return evidence + link_media(ctx)
