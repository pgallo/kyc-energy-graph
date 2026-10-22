from kycgraph.cli import FLOORS, main
from kycgraph.evaluate import risk_metrics, typology_metrics


def test_risk_and_confidence_are_separate_and_bounded(assessments):
    for a in assessments.values():
        assert 0 < a.risk < 1 and 0 < a.confidence <= 1
    risks = [a.risk for a in assessments.values()]
    confs = [a.confidence for a in assessments.values()]
    assert len({round(r, 3) for r in risks}) > 10 and len({round(c, 3) for c in confs}) > 10


def test_fused_risk_ranks_typology_companies_first(analysis, assessments):
    m = risk_metrics(analysis, assessments)
    assert m["auc_fused"] >= FLOORS["auc_fused"]
    assert m["precision_at_k"] >= 0.8
    assert m["positives_queued"] >= 0.8 * m["k"]


def test_ml_model_is_part_of_the_explained_fusion(assessments):
    companies = [a for a in assessments.values() if a.kind == "company"]
    assert all(any(e.kind == "ml_counterparty_model" for e in a.evidence) for a in companies)
    assert all(a.ml_explanation for a in companies)


def test_every_typology_meets_floors(analysis):
    for kind, m in typology_metrics(analysis).items():
        assert m["recall"] >= FLOORS["typology_recall"], kind
        assert m["precision"] >= FLOORS["typology_precision"], kind


def test_cli_generate_assess_review(tmp_path, analysis, capsys):
    data = str(tmp_path)
    assert main(["--data", data, "generate"]) == 0
    sanction_case = analysis.ds.truth.typologies["sanctioned_exposure"][0]["companies"][-1]
    company = next(e for e, ent in analysis.res.entities.items()
                   if ent.kind == "company" and analysis.ds.truth.mention_entity[ent.members[0]] == sanction_case)
    out = tmp_path / "report.md"
    assert main(["--data", data, "assess", company, "--out", str(out)]) == 0
    text = out.read_text()
    for heading in ("## Ownership tree", "## Key persons", "## Evidence and fusion", "### Evidence chains", "## Confidence"):
        assert heading in text
    assert "sanctioned_exposure" in text
    assert main(["--data", data, "review", "decide", company, "--decision", "escalate", "--analyst", "t1"]) == 0
    assert main(["--data", data, "review", "verify"]) == 0
    capsys.readouterr()
    assert main(["--data", data, "assess", "NOPE"]) == 2


def test_cli_run_all_meets_floors(tmp_path, capsys):
    assert main(["--data", str(tmp_path), "run-all"]) == 0
    out = capsys.readouterr().out
    assert "Floors: all met" in out
    assert "| person | mlp | platt |" in out and "| company | logistic | none |" in out
