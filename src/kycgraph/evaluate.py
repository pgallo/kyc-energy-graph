"""Metrics against the generator's ground truth."""

from __future__ import annotations

from .model import TYPOLOGIES
from .pipeline import Analysis, true_id
from .risk import REVIEW_RISK, Assessment


def prf(predicted: set, actual: set) -> dict[str, float]:
    tp = len(predicted & actual)
    p = tp / len(predicted) if predicted else (1.0 if not actual else 0.0)
    r = tp / len(actual) if actual else 1.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "tp": tp, "predicted": len(predicted), "actual": len(actual)}


def typology_metrics(a: Analysis) -> dict[str, dict[str, float]]:
    out = {}
    for t in TYPOLOGIES:
        flagged = {true_id(a.res, a.ds, e.subject) for e in a.evidence if e.kind == t}
        actual = {c for case in a.ds.truth.typologies[t] for c in case["companies"]}
        out[t] = prf(flagged, actual)
    return out


def screening_metrics(a: Analysis) -> dict[str, float]:
    predicted = {(true_id(a.res, a.ds, h.person), h.list_id) for h in a.ctx.hits if h.decision == "hit"}
    possible = {(true_id(a.res, a.ds, h.person), h.list_id) for h in a.ctx.hits}
    actual = {(m["entity"], m["list_id"]) for m in a.ds.truth.screening}
    hit = prf(predicted, actual)
    hit["recall_incl_possible"] = len(possible & actual) / len(actual) if actual else 1.0
    hit["possible_only"] = len(possible - predicted)
    return hit


def auc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def risk_metrics(a: Analysis, assessments: dict[str, Assessment]) -> dict[str, float]:
    positives = {c for t in TYPOLOGIES for case in a.ds.truth.typologies[t] for c in case["companies"]}
    companies = [x for x in assessments.values() if x.kind == "company"]
    labels = [int(true_id(a.res, a.ds, x.subject) in positives) for x in companies]
    fused = [x.risk for x in companies]
    ml = [x.ml_probability or 0.0 for x in companies]
    ranked = sorted(zip(fused, labels, strict=True), reverse=True)
    k = sum(labels)
    queue = [x for x in assessments.values() if x.review_reasons]
    return {
        "auc_fused": auc(fused, labels), "auc_ml_only": auc(ml, labels),
        "precision_at_k": sum(y for _, y in ranked[:k]) / k if k else 1.0, "k": k,
        "review_queue": len(queue),
        "queued_high_risk": sum(1 for x in queue if x.risk >= REVIEW_RISK),
        "queued_low_confidence_only": sum(1 for x in queue if x.risk < REVIEW_RISK),
        "positives_queued": sum(1 for x, y in zip(companies, labels, strict=True) if y and x.review_reasons),
    }
