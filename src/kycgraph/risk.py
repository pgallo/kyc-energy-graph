"""Risk fusion, assessment confidence and the learned counterparty model.

Risk and confidence are kept apart on purpose. Risk answers "how likely is this
counterparty to be involved in a typology", fused in log-odds from discounted evidence.
Confidence answers "how much should we trust that number", from data completeness,
provider agreement and entity-resolution certainty. A low-risk score with low confidence
is not a clearance; it is a reason for review.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier

from .graph import KeyPerson
from .model import TYPOLOGIES
from .typologies import Context, Evidence

PRIOR = 0.03
SOURCE_RELIABILITY = {"lists": 0.95, "tenders": 0.95, "media": 0.6, "model": 0.5}
REVIEW_RISK, REVIEW_CONFIDENCE = 0.3, 0.6
ML_FEATURES = [*(f"lr_{t}" for t in TYPOLOGIES), "kp_sanctions_conf", "kp_pep_conf", "media_severity",
               "ubo_depth", "has_ubo", "offshore_ancestors", "nominee_directors", "log_employees",
               "age_years", "tender_awards", "address_peers", "link_conf"]


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def source_weight(sources: list[str], reliability: dict[str, float]) -> float:
    """Noisy-OR of independent sources: corroboration raises trust, never above 1."""
    miss = 1.0
    for s in sources:
        miss *= 1 - reliability.get(s, SOURCE_RELIABILITY.get(s, 0.5))
    return 1 - miss


def fuse(evidence: list[Evidence], reliability: dict[str, float], prior: float = PRIOR) -> float:
    """Discounted log-odds fusion: L = logit(prior) + sum_k max_e(c_e * w_e * ln LR_e).

    Only the strongest item per evidence kind counts, so one typology seen through several
    paths is not counted several times.
    """
    best: dict[str, Evidence] = {}
    for e in evidence:
        e.contribution = e.confidence * source_weight(e.sources, reliability) * e.log_lr
        if e.kind not in best or e.contribution > best[e.kind].contribution:
            best[e.kind] = e
    return sigmoid(logit(prior) + sum(e.contribution for e in best.values()))


@dataclass
class Assessment:
    subject: str
    kind: str
    name: str
    risk: float
    confidence: float
    components: dict[str, float]
    evidence: list[Evidence]
    key_persons: list[KeyPerson] = field(default_factory=list)
    ml_probability: float | None = None
    ml_explanation: list[tuple[str, float, float]] = field(default_factory=list)

    @property
    def review_reasons(self) -> list[str]:
        reasons = []
        if self.risk >= REVIEW_RISK:
            reasons.append(f"risk {self.risk:.2f} >= {REVIEW_RISK}")
        if self.confidence < REVIEW_CONFIDENCE:
            reasons.append(f"confidence {self.confidence:.2f} < {REVIEW_CONFIDENCE}")
        return reasons


def confidence(ctx: Context, subject: str, evidence: list[Evidence],
               key_persons: list[KeyPerson]) -> tuple[float, dict[str, float]]:
    e = ctx.res.entities[subject]
    wanted = (("address", "incorporated", "sector", "employees") if e.kind == "company"
              else ("dob", "nationality", "address"))
    completeness = sum(e.attrs.get(f) not in (None, "") for f in wanted) / len(wanted)
    feeds = len(e.sources - {"REG"})
    sourcing = 1.0 if "REG" in e.sources else min(0.95, 0.5 + 0.15 * feeds)
    spreads = [d["spread"] for _, _, d in ctx.graph.own.in_edges(subject, data=True)]
    edge_agree = 1 - min(1.0, max(spreads, default=0.0) / 20)
    agreement = 0.5 * (e.agreement if e.agreement is not None else 0.7) + 0.5 * edge_agree
    strong = [ctx.res.entities[k.person].link_conf for k in key_persons if k.influence >= 0.6]
    resolution = e.link_conf * (min(strong) if strong else 1.0)
    ev_quality = float(np.mean([x.confidence for x in evidence])) if evidence else 1.0
    parts = {"data_quality": completeness * sourcing, "provider_agreement": agreement,
             "resolution": resolution, "evidence_quality": ev_quality}
    weights = {"data_quality": 0.25, "provider_agreement": 0.25, "resolution": 0.35, "evidence_quality": 0.15}
    conf = math.prod(max(parts[k], 1e-3) ** w for k, w in weights.items())
    return conf, parts


def propagate(ctx: Context, kp: KeyPerson, person_evidence: list[Evidence], company: str) -> list[Evidence]:
    """Key-person evidence becomes company evidence, discounted by the person's influence."""
    out = []
    for e in person_evidence:
        if e.kind in {"screening_sanctions", "screening_pep", "adverse_media"}:
            out.append(Evidence(f"key_person_{e.kind}", company, f"via {kp.person} ({', '.join(kp.roles)}): {e.summary}",
                                e.lr, e.confidence * kp.influence, e.sources, e.records,
                                [f"{kp.person} -[{','.join(kp.roles)}]-> {company}"]))
    return out


def person_evidence(ctx: Context, all_evidence: list[Evidence]) -> dict[str, list[Evidence]]:
    out: dict[str, list[Evidence]] = {}
    for h in ctx.hits:
        if h.list_name == "sanctions":
            ev = Evidence("screening_sanctions", h.person, f"Matches {h.list_id} '{h.entry_name}' ({h.decision}, "
                          f"name {h.name_sim:.2f}, dob {h.dob:+.1f})", 40.0, h.confidence, ["lists"], [h.list_id])
        else:
            ev = Evidence("screening_pep", h.person, f"Matches PEP {h.list_id} '{h.note}, {h.organisation}' "
                          f"({h.decision})", 3.0, h.confidence, ["lists"], [h.list_id])
        out.setdefault(h.person, []).append(ev)
    for e in all_evidence:
        if ctx.res.entities[e.subject].kind == "person":
            out.setdefault(e.subject, []).append(e)
    for p in ctx.graph.own.nodes:
        if ctx.graph.is_person(p) and ctx.graph.nominee_pattern(p):
            boards = ctx.graph.roles.out_degree(p)
            out.setdefault(p, []).append(Evidence("nominee_pattern", p, f"Director of {boards} companies, owns none",
                                                  3.0, ctx.res.entities[p].link_conf, ["REG"], ctx.records(p)))
    return out


# ---- learned counterparty model ---------------------------------------------------------

def company_features(ctx: Context, company: str, evidence: list[Evidence], kps: list[KeyPerson],
                     pev: dict[str, list[Evidence]], address_peers: dict[str, int]) -> np.ndarray:
    g, e = ctx.graph, ctx.res.entities[company]
    lr = {t: max((x.log_lr * x.confidence for x in evidence if x.kind == t), default=0.0) for t in TYPOLOGIES}

    def kp_max(kind: str) -> float:
        return max((x.confidence * k.influence for k in kps for x in pev.get(k.person, []) if x.kind == kind), default=0.0)

    media = max((x.log_lr for x in evidence if x.kind == "adverse_media"), default=0.0)
    ubos = [k for k in kps if "ubo" in k.roles]
    depth = max((len(p) - 1 for k in ubos for p in k.paths), default=0)
    ancestors = [a for a in g.own.predecessors(company)]
    offshore = sum(g.offshore(a) for a in ancestors)
    inc = e.attrs.get("incorporated")
    age = (date.fromisoformat(ctx.ds.as_of) - date.fromisoformat(inc)).days / 365 if inc else 10.0
    return np.array([
        *lr.values(), kp_max("screening_sanctions"), kp_max("screening_pep"), media, depth, float(bool(ubos)),
        offshore, sum(k.nominee_pattern for k in kps), math.log1p(e.attrs.get("employees") or 0), age,
        len(ctx.awards.get(company, [])), address_peers.get(company, 0), e.link_conf,
    ])


@dataclass
class RiskModel:
    model: object
    baseline: np.ndarray
    base_rate: float

    @classmethod
    def train(cls, X: np.ndarray, y: np.ndarray) -> RiskModel:
        base = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.1, random_state=0)
        model = CalibratedClassifierCV(clone(base), method="sigmoid", cv=3).fit(X, y)
        return cls(model, np.median(X, axis=0), float(y.mean()))

    def proba(self, x: np.ndarray) -> float:
        return float(self.model.predict_proba(x[None, :])[0, 1])

    def explain(self, x: np.ndarray, top: int = 4) -> list[tuple[str, float, float]]:
        p = self.proba(x)
        deltas = []
        for i, name in enumerate(ML_FEATURES):
            z = x.copy()
            z[i] = self.baseline[i]
            deltas.append((name, float(x[i]), p - self.proba(z)))
        return sorted(deltas, key=lambda t: -abs(t[2]))[:top]

    def as_evidence(self, company: str, x: np.ndarray) -> tuple[Evidence, float, list]:
        """Turn the model output into a likelihood ratio against its training base rate."""
        p = min(max(self.proba(x), 1e-3), 1 - 1e-3)
        lr = (p / (1 - p)) / (self.base_rate / (1 - self.base_rate))
        lr = min(max(lr, 0.2), 20.0)
        expl = self.explain(x)
        text = ", ".join(f"{n}={v:.2f} ({d:+.2f})" for n, v, d in expl)
        return Evidence("ml_counterparty_model", company, f"Gradient-boosting score {p:.2f}; top drivers: {text}",
                        lr, 0.8, ["model"]), p, expl
