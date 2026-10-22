"""Human review: a queue of cases and an append-only, hash-chained decision log."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .risk import Assessment

DECISIONS = ("clear", "escalate", "request_info")
GENESIS = "0" * 64


@dataclass
class Case:
    case_id: str
    subject: str
    kind: str
    name: str
    risk: float
    confidence: float
    reasons: list[str]


def build_queue(assessments: dict[str, Assessment], log: DecisionLog | None = None) -> list[Case]:
    closed = {cid for cid, e in (log.latest() if log else {}).items() if e["decision"] != "request_info"}
    cases = [Case(f"CASE-{a.subject}", a.subject, a.kind, a.name, a.risk, a.confidence, a.review_reasons)
             for a in assessments.values() if a.review_reasons]
    return sorted((c for c in cases if c.case_id not in closed), key=lambda c: (-c.risk, c.confidence, c.case_id))


def _digest(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


class DecisionLog:
    """JSON-lines file opened only in append mode; each entry commits to its predecessor."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def append(self, case_id: str, decision: str, analyst: str, note: str = "",
               risk: float | None = None, confidence: float | None = None, now: datetime | None = None) -> dict:
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}")
        if not analyst:
            raise ValueError("analyst is required")
        existing = self.entries()
        entry = {
            "seq": len(existing) + 1,
            "ts": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
            "case_id": case_id, "decision": decision, "analyst": analyst, "note": note,
            "risk": None if risk is None else round(risk, 4),
            "confidence": None if confidence is None else round(confidence, 4),
            "prev": existing[-1]["hash"] if existing else GENESIS,
        }
        entry["hash"] = _digest(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def verify(self) -> tuple[bool, str]:
        prev = GENESIS
        for i, e in enumerate(self.entries(), start=1):
            if e.get("seq") != i:
                return False, f"entry {i}: sequence gap"
            if e.get("prev") != prev:
                return False, f"entry {i}: chain broken"
            if _digest(e) != e.get("hash"):
                return False, f"entry {i}: content altered"
            prev = e["hash"]
        return True, "ok"

    def latest(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for e in self.entries():
            out[e["case_id"]] = e
        return out
