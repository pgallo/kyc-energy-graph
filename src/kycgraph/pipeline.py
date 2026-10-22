"""End-to-end orchestration: train, resolve, build graph, screen, detect, assess."""

from __future__ import annotations

import functools
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .graph import CorporateGraph
from .model import TYPOLOGIES, Dataset
from .normalize import address_tokens
from .resolution import MatchModel, Resolution, resolve, select_champions, train_models
from .risk import (
    Assessment,
    RiskModel,
    company_features,
    confidence,
    fuse,
    person_evidence,
    propagate,
)
from .screening import screen
from .synth import generate
from .typologies import Context, Evidence, run_detectors

TRAIN_OFFSET, VALIDATION_OFFSET = 1000, 2000


@dataclass
class Analysis:
    ds: Dataset
    ctx: Context
    evidence: list[Evidence]
    person_ev: dict[str, list[Evidence]]
    features: dict[str, np.ndarray]

    @property
    def res(self) -> Resolution:
        return self.ctx.res


def analyse(ds: Dataset, models: dict[str, dict[str, MatchModel]], champions: dict[str, str]) -> Analysis:
    res = resolve(ds, models, champions)
    graph = CorporateGraph.build(ds, res)
    records = {r.key: r for r in [*ds.companies, *ds.persons]}
    ctx = Context(ds, res, graph, screen(res, ds.watchlist, records))
    evidence = run_detectors(ctx)
    pev = person_evidence(ctx, evidence)
    peers: Counter = Counter()
    by_addr = defaultdict(list)
    for eid, e in res.entities.items():
        if e.kind == "company" and e.attrs.get("address"):
            by_addr[" ".join(sorted(address_tokens(e.attrs["address"])))].append(eid)
    for members in by_addr.values():
        for m in members:
            peers[m] = len(members) - 1
    feats = {}
    for eid, e in res.entities.items():
        if e.kind == "company":
            kps = graph.key_persons(eid)
            feats[eid] = company_features(ctx, eid, [x for x in evidence if x.subject == eid], kps, pev, peers)
    return Analysis(ds, ctx, evidence, pev, feats)


def true_id(res: Resolution, ds: Dataset, eid: str) -> str:
    return Counter(ds.truth.mention_entity[k] for k in res.entities[eid].members).most_common(1)[0][0]


def risk_labels(a: Analysis) -> tuple[np.ndarray, np.ndarray]:
    positives = {c for t in TYPOLOGIES for case in a.ds.truth.typologies[t] for c in case["companies"]}
    ids = sorted(a.features)
    X = np.array([a.features[i] for i in ids])
    y = np.array([int(true_id(a.res, a.ds, i) in positives) for i in ids])
    return X, y


@dataclass
class Engine:
    models: dict[str, dict[str, MatchModel]]
    champions: dict[str, str]
    risk_model: RiskModel

    @classmethod
    def build(cls, seed: int) -> Engine:
        """Train match models on one synthetic world, pick champions on a second, and train the
        counterparty model on both. The evaluated world (``seed``) is never used for fitting."""
        train, validation = generate(seed + TRAIN_OFFSET), generate(seed + VALIDATION_OFFSET)
        models = train_models(train)
        champions = select_champions(models, validation)
        parts = [risk_labels(analyse(w, models, champions)) for w in (train, validation)]
        X = np.vstack([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        return cls(models, champions, RiskModel.train(X, y))

    def assess_all(self, a: Analysis) -> dict[str, Assessment]:
        ctx, out = a.ctx, {}
        for eid, e in ctx.res.entities.items():
            if e.kind == "company":
                kps = ctx.graph.key_persons(eid)
                ev = [x for x in a.evidence if x.subject == eid]
                for kp in kps:
                    ev += propagate(ctx, kp, a.person_ev.get(kp.person, []), eid)
                ml_ev, p, expl = self.risk_model.as_evidence(eid, a.features[eid])
                ev.append(ml_ev)
                conf, parts = confidence(ctx, eid, ev, kps)
                out[eid] = Assessment(eid, "company", e.name, fuse(ev, ctx.res.reliability), conf, parts, ev, kps, p, expl)
            elif eid in a.person_ev or ctx.graph.roles.out_degree(eid) or ctx.graph.own.out_degree(eid):
                ev = list(a.person_ev.get(eid, []))
                conf, parts = confidence(ctx, eid, ev, [])
                out[eid] = Assessment(eid, "person", e.name, fuse(ev, ctx.res.reliability), conf, parts, ev)
        return out


@functools.cache
def engine_for(seed: int) -> Engine:
    """Engines are deterministic per seed, so one process trains each at most once."""
    return Engine.build(seed)
