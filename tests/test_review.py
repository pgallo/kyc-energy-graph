import json
from datetime import UTC, datetime

import pytest

from kycgraph.review import DecisionLog, build_queue
from kycgraph.risk import Assessment

NOW = datetime(2026, 7, 1, tzinfo=UTC)


def test_append_and_verify(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl")
    log.append("CASE-1", "escalate", "analyst1", "chain looks layered", 0.8, 0.9, now=NOW)
    log.append("CASE-1", "clear", "analyst2", "documents received", 0.8, 0.9, now=NOW)
    assert log.verify() == (True, "ok")
    assert log.latest()["CASE-1"]["decision"] == "clear"
    assert [e["seq"] for e in log.entries()] == [1, 2]


def test_tampering_is_detected(tmp_path):
    path = tmp_path / "d.jsonl"
    log = DecisionLog(path)
    log.append("CASE-1", "escalate", "a", now=NOW)
    log.append("CASE-2", "clear", "a", now=NOW)
    lines = path.read_text().splitlines()
    first = json.loads(lines[0])
    first["decision"] = "clear"
    path.write_text("\n".join([json.dumps(first, sort_keys=True), lines[1]]) + "\n")
    ok, msg = log.verify()
    assert not ok and "entry 1" in msg


def test_deleting_an_entry_breaks_the_chain(tmp_path):
    path = tmp_path / "d.jsonl"
    log = DecisionLog(path)
    for i in range(3):
        log.append(f"CASE-{i}", "clear", "a", now=NOW)
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n")
    assert not log.verify()[0]


def test_invalid_input_is_rejected(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl")
    with pytest.raises(ValueError):
        log.append("CASE-1", "approve", "a")
    with pytest.raises(ValueError):
        log.append("CASE-1", "clear", "")


def test_queue_routes_high_risk_and_low_confidence_and_drops_closed_cases(tmp_path):
    def a(sid, risk, conf):
        return Assessment(sid, "company", sid, risk, conf, {}, [])

    assessments = {"H": a("H", 0.9, 0.9), "L": a("L", 0.05, 0.3), "OK": a("OK", 0.05, 0.95)}
    log = DecisionLog(tmp_path / "d.jsonl")
    assert [c.subject for c in build_queue(assessments, log)] == ["H", "L"]
    log.append("CASE-H", "clear", "a", now=NOW)
    log.append("CASE-L", "request_info", "a", now=NOW)
    assert [c.subject for c in build_queue(assessments, log)] == ["L"]
