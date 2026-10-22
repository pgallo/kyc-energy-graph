"""Markdown rendering of metrics and per-counterparty assessments."""

from __future__ import annotations

from .model import TYPOLOGIES
from .pipeline import Analysis, Engine
from .resolution import COMPANY_FEATURES, PERSON_FEATURES, Resolution
from .risk import PRIOR, Assessment, logit, source_weight


def table(headers: list[str], rows: list[list[object]]) -> str:
    def fmt(v: object) -> str:
        if isinstance(v, float):
            return f"{v:.4f}" if 0 < abs(v) < 0.01 else f"{v:.3f}"
        return str(v).replace("|", "/")

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


def resolution_report(res: Resolution, champions: dict[str, str]) -> str:
    rows = [[k, champions[k], m["precision"], m["recall"], m["f1"], m["entities"], m["true_entities"],
             m["pair_completeness"], m["reduction_ratio"], m["min_cuts"]] for k, m in res.metrics.items()]
    out = ["## Entity resolution (clustered, champion model)",
           table(["kind", "model", "precision", "recall", "F1", "entities", "true", "pair completeness",
                  "reduction ratio", "min-cuts"], rows), "",
           "## Match models: logistic regression vs neural network (out-of-sample pairs, threshold 0.5)",
           table(["kind", "model", "calibration", "precision", "recall", "F1", "Brier", "ECE", "cluster F1"],
                 [[r["kind"], r["model"], r["calibration"], r["precision"], r["recall"], r["f1"], r["brier"],
                   r["ece"], r["cluster_f1"]] for r in res.comparison]), "",
           "## Estimated source reliability (agreement with the registry)",
           table(["source", "reliability"], [[s, v] for s, v in sorted(res.reliability.items())])]
    return "\n".join(out)


def explain_pair(res: Resolution, a: str, b: str) -> str:
    """Show why two records were (or were not) linked: features and their attributions."""
    from .resolution import features

    pa, pb = res.prepared[a], res.prepared[b]
    model = res.model[pa.kind]
    x = features(pa, pb)
    names = COMPANY_FEATURES if pa.kind == "company" else PERSON_FEATURES
    p = model.proba(x[None, :])[0]
    rows = [[n, v, d] for n, v, d in model.explain(x, names)]
    return f"Match probability {p:.3f} ({model.name})\n\n" + table(["feature", "value", "effect on log-odds"], rows)


def assessment_report(a: Assessment, analysis: Analysis, engine: Engine, review_status: str = "") -> str:
    ctx = analysis.ctx
    e = ctx.res.entities[a.subject]
    band = "high" if a.risk >= 0.5 else "medium" if a.risk >= 0.2 else "low"
    out = [
        f"# Counterparty assessment: {a.subject}",
        "",
        f"**{a.name}** (synthetic data, seed {analysis.ds.seed}, as of {analysis.ds.as_of})",
        "",
        table(["risk", "band", "confidence", "review"],
              [[a.risk, band, a.confidence, "; ".join(a.review_reasons) or "not required"]]),
    ]
    if review_status:
        out += ["", f"Review log: {review_status}"]
    out += ["", "## Entity", "",
            f"Resolved from {len(e.members)} record(s) in sources {', '.join(sorted(e.sources))}: "
            + ", ".join(f"`{k}`" for k in e.members), "",
            table(["attribute", "value"], [[k, v] for k, v in e.attrs.items() if v is not None]), "",
            "## Ownership tree", "",
            "Shares are fused across sources; `eff` is effective (integrated) ownership in this company.", "",
            "```", *ctx.graph.ownership_tree(a.subject), "```", "",
            "## Key persons", ""]
    hits = {h.person: h for h in ctx.hits}
    out.append(table(
        ["person", "name", "roles", "effective", "influence", "betweenness", "bridge", "nominee pattern", "screening"],
        [[k.person, k.name, ", ".join(k.roles), f"{k.effective:.1%}", k.influence, k.betweenness, k.bridge,
          k.nominee_pattern, f"{hits[k.person].list_id} {hits[k.person].decision} ({hits[k.person].confidence:.2f})"
          if k.person in hits else "none"] for k in a.key_persons]))
    out += ["", "## Evidence and fusion", "",
            f"Fusion in log-odds from prior {PRIOR:.0%} (logit {logit(PRIOR):+.2f}). Each item contributes "
            "confidence x source weight x ln(LR); only the strongest item per kind is counted.", ""]
    best: dict[str, float] = {}
    for x in a.evidence:
        best[x.kind] = max(best.get(x.kind, float("-inf")), x.contribution)
    rows = []
    for x in sorted(a.evidence, key=lambda x: -abs(x.contribution)):
        counted = "yes" if best[x.kind] == x.contribution else "no (weaker duplicate)"
        rows.append([x.kind, x.lr, x.confidence, source_weight(x.sources, ctx.res.reliability), x.contribution,
                     counted, x.summary])
    out.append(table(["kind", "LR", "confidence", "source weight", "contribution", "counted", "summary"], rows))
    total = logit(PRIOR) + sum(best.values())
    out += ["", f"Posterior logit {total:+.2f} gives risk {a.risk:.3f}.", "", "### Evidence chains", ""]
    for i, x in enumerate([x for x in a.evidence if x.kind != "ml_counterparty_model"], start=1):
        out.append(f"{i}. **{x.kind}** (sources: {', '.join(x.sources)})")
        if x.edges:
            out += [f"   - edge: `{edge}`" for edge in x.edges]
        if x.records:
            out.append("   - records: " + ", ".join(f"`{r}`" for r in x.records[:12])
                       + (" ..." if len(x.records) > 12 else ""))
    out += ["", "## Learned counterparty model", "",
            f"Gradient boosting (calibrated) probability {a.ml_probability:.3f}. Largest drivers by occlusion "
            "(probability change when the feature is reset to the training median):", "",
            table(["feature", "value", "effect"], [[n, v, d] for n, v, d in a.ml_explanation]), "",
            "## Confidence", "",
            "Confidence is separate from risk: it measures how far the assessment can be trusted.", "",
            table(["component", "value"], [[k, v] for k, v in a.components.items()] + [["overall", a.confidence]])]
    return "\n".join(out) + "\n"


def run_all_report(analysis: Analysis, engine: Engine, typ: dict, scr: dict, risk: dict,
                   assessments: dict[str, Assessment]) -> str:
    out = [resolution_report(analysis.res, engine.champions), "",
           "## Screening (hits at confidence >= 0.80)",
           table(["precision", "recall", "F1", "hits", "recall incl. possible", "possible only"],
                 [[scr["precision"], scr["recall"], scr["f1"], scr["predicted"], scr["recall_incl_possible"],
                   scr["possible_only"]]]), "",
           "## Typology detection vs ground truth (company level)",
           table(["typology", "precision", "recall", "F1", "flagged", "actual"],
                 [[t, typ[t]["precision"], typ[t]["recall"], typ[t]["f1"], typ[t]["predicted"], typ[t]["actual"]]
                  for t in TYPOLOGIES]), "",
           "## Counterparty risk",
           table(["AUC fused", "AUC ML only", f"precision@{risk['k']}", "review queue", "high risk",
                  "low confidence only", "positives queued"],
                 [[risk["auc_fused"], risk["auc_ml_only"], risk["precision_at_k"], risk["review_queue"],
                   risk["queued_high_risk"], risk["queued_low_confidence_only"], risk["positives_queued"]]]), "",
           "## Top counterparties by fused risk"]
    top = sorted((x for x in assessments.values() if x.kind == "company"), key=lambda x: -x.risk)[:10]
    out.append(table(["company", "name", "risk", "confidence", "ML p", "evidence kinds"],
                     [[x.subject, x.name, x.risk, x.confidence, x.ml_probability or 0.0,
                       ", ".join(sorted({e.kind for e in x.evidence if e.contribution > 0}))] for x in top]))
    return "\n".join(out)
