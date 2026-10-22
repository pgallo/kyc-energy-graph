"""Fuzzy screening of resolved persons against the synthetic sanctions and PEP lists.

The scorecard is deliberately hand-weighted and monotone: name similarity drives the score,
birth date agreement confirms or vetoes it, nationality nudges it, and very common names
are penalised. False positives are controlled by the date veto and by a two-level decision
(hit / possible) so that weak matches go to an analyst rather than straight into risk.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .model import ListEntry
from .normalize import dob_agreement, person_tokens, phonetic
from .resolution import Resolution
from .similarity import name_similarity, weakest_token_similarity

HIT, POSSIBLE = 0.80, 0.50


@dataclass
class ScreeningHit:
    person: str
    list_id: str
    list_name: str
    entry_name: str
    matched_name: str
    name_sim: float
    dob: float
    nationality: float
    commonness: float
    confidence: float
    decision: str
    note: str
    organisation: str | None


def _token_commonness(res: Resolution) -> dict[str, float]:
    counts: Counter = Counter()
    for e in res.entities.values():
        if e.kind == "person":
            counts.update(set(person_tokens(e.name)))
    top = max(counts.values(), default=1)
    return {t: math.log1p(c) / math.log1p(top) for t, c in counts.items()}


def list_name_similarity(a: list[str], b: list[str]) -> float:
    """Whole-name similarity averaged with the weakest token alignment.

    A shared given name must not carry a different surname over the threshold, so the
    worst-aligned token pulls the score down.
    """
    return 0.5 * name_similarity(a, b) + 0.5 * weakest_token_similarity(a, b)


def score(name_sim: float, dob: float, nationality: float, commonness: float) -> float:
    z = 14 * (name_sim - 0.9) + 2.5 * dob + 0.8 * nationality - 1.5 * commonness + 0.5
    return 1 / (1 + math.exp(-z))


def screen(res: Resolution, entries: list[ListEntry], records: dict[str, object]) -> list[ScreeningHit]:
    common = _token_commonness(res)
    hits = []
    for e in res.entities.values():
        if e.kind != "person":
            continue
        names = {records[k].name for k in e.members}
        tokens = [person_tokens(n) for n in names]
        phon = {phonetic(t) for toks in tokens for t in toks}
        dobs = [records[k].dob for k in e.members if records[k].dob]
        nats = {records[k].nationality for k in e.members if records[k].nationality}
        for entry in entries:
            entry_tokens = [person_tokens(n) for n in [entry.name, *entry.aliases]]
            if not phon & {phonetic(t) for toks in entry_tokens for t in toks}:
                continue  # blocking: no shared phonetic token
            sim, best = max((list_name_similarity(a, b), " ".join(a)) for a in tokens for b in entry_tokens)
            if sim < 0.8:
                continue
            agreements = [dob_agreement(d, entry.dob) for d in dobs] or [0.0]
            dob = max(agreements) if max(agreements) > 0 else min(agreements)
            nat = 0.0 if not nats or not entry.nationality else (1.0 if entry.nationality in nats else -1.0)
            toks = [t for toks in tokens for t in toks]
            commonness = sum(common.get(t, 0.0) for t in toks) / max(len(toks), 1)
            conf = score(sim, dob, nat, commonness)
            if conf < POSSIBLE:
                continue
            hits.append(ScreeningHit(e.id, entry.list_id, entry.list_name, entry.name, best, sim, dob, nat,
                                     commonness, conf, "hit" if conf >= HIT else "possible", entry.note,
                                     entry.organisation))
    return sorted(hits, key=lambda h: -h.confidence)
