"""Entity resolution: normalise, block, score pairs with calibrated models, cluster safely.

Two match models are trained on a *separately seeded* world (so evaluation is out of
sample): a logistic-regression baseline and a small multi-layer perceptron. Both are
calibrated with Platt scaling. Clustering takes the transitive closure of confident links
but enforces cannot-link constraints (same source, conflicting identifiers, conflicting
birth dates) by cutting the weakest edges with a minimum cut.
"""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .model import Dataset
from .normalize import (
    address_tokens,
    blocking_tokens,
    company_tokens,
    dob_agreement,
    id_digits,
    name_key,
    person_tokens,
    phonetic,
)
from .similarity import jaccard, name_similarity, weakest_token_similarity

COMPANY_FEATURES = ["name_sim", "weakest_token", "token_jaccard", "phonetic_overlap", "id_agree", "address_sim",
                    "jurisdiction_agree", "inc_agree", "sector_agree", "shared_links"]
PERSON_FEATURES = ["name_sim", "weakest_token", "token_jaccard", "phonetic_overlap", "dob", "nationality_agree",
                   "address_sim", "id_agree", "shared_links"]
FEATURES = {"company": COMPANY_FEATURES, "person": PERSON_FEATURES}
MAX_BLOCK = 80


@dataclass
class Prepared:
    key: str
    kind: str
    source: str
    tokens: list[str]
    phon: set[str]
    address: set[str]
    ids: dict[str, str]
    dob: str | None = None
    nationality: str | None = None
    jurisdiction: str | None = None
    inc: str | None = None
    sector: str | None = None
    links: set[str] = field(default_factory=set)


def prepare(ds: Dataset) -> dict[str, Prepared]:
    out: dict[str, Prepared] = {}
    for c in ds.companies:
        toks = company_tokens(c.name)
        ids = {k: v for k, v in (("reg", id_digits(c.reg_no)), ("tax", c.tax_id), ("foreign", c.foreign_reg)) if v}
        out[c.key] = Prepared(c.key, "company", c.source, toks, {phonetic(t) for t in toks},
                              address_tokens(c.address), ids, jurisdiction=c.jurisdiction,
                              inc=c.incorporated, sector=c.sector)
    for p in ds.persons:
        toks = person_tokens(p.name)
        ids = {"nid": p.national_id} if p.national_id else {}
        out[p.key] = Prepared(p.key, "person", p.source, toks, {phonetic(t) for t in toks},
                              address_tokens(p.address), ids, dob=p.dob, nationality=p.nationality)
    for rel in ds.relations:  # neighbour signatures: who sits on / owns what, per source
        a, b = out.get(rel.src), out.get(rel.dst)
        if a and b:
            a.links.add(_signature(b))
            b.links.add(_signature(a))
    return out


def _signature(p: Prepared) -> str:
    return " ".join(sorted(phonetic(t) for t in blocking_tokens(p.tokens)))


def blocking_keys(p: Prepared) -> set[str]:
    keys = {f"{k}:{v}" for k, v in p.ids.items()}
    if p.kind == "company":
        for t in blocking_tokens(p.tokens):
            keys |= {f"n:{phonetic(t)}", f"p:{t[:4]}"}
    else:
        year = (p.dob or "")[:4]
        keys.add("pp:" + " ".join(sorted(p.phon)))
        for t in p.tokens:
            keys |= {f"py:{phonetic(t)}:{year}", f"pf:{t[:3]}:{year}"}
    return keys


def candidate_pairs(prep: dict[str, Prepared]) -> tuple[set[tuple[str, str]], int]:
    blocks: dict[str, list[str]] = defaultdict(list)
    for p in prep.values():
        for k in blocking_keys(p):
            blocks[p.kind[0] + k].append(p.key)
    pairs, skipped = set(), 0
    for members in blocks.values():
        if len(members) > MAX_BLOCK:
            skipped += 1
            continue
        for a, b in itertools.combinations(sorted(members), 2):
            if prep[a].source != prep[b].source:
                pairs.add((a, b))
    return pairs, skipped


def _agree(a: str | None, b: str | None) -> float:
    if not a or not b:
        return 0.0
    return 1.0 if a == b else -1.0


def _id_agree(a: dict[str, str], b: dict[str, str]) -> float:
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    return 1.0 if any(a[k] == b[k] for k in shared) else -1.0


def features(a: Prepared, b: Prepared) -> np.ndarray:
    common = [
        name_similarity(a.tokens, b.tokens),
        weakest_token_similarity(a.tokens, b.tokens),
        jaccard(set(a.tokens), set(b.tokens)),
        jaccard(a.phon, b.phon),
    ]
    if a.kind == "company":
        inc = 0.0
        if a.inc and b.inc:
            inc = 1.0 if a.inc == b.inc else (0.3 if abs(int(a.inc[:4]) - int(b.inc[:4])) <= 1 else -1.0)
        rest = [_id_agree(a.ids, b.ids), jaccard(a.address, b.address), _agree(a.jurisdiction, b.jurisdiction),
                inc, _agree(a.sector, b.sector), jaccard(a.links, b.links)]
    else:
        rest = [dob_agreement(a.dob, b.dob), _agree(a.nationality, b.nationality),
                jaccard(a.address, b.address), _id_agree(a.ids, b.ids), jaccard(a.links, b.links)]
    return np.array(common + rest)


def cannot_link(a: Prepared, b: Prepared) -> bool:
    if a.source == b.source:
        return True  # each source is internally de-duplicated
    if _id_agree(a.ids, b.ids) < 0:
        return True
    return a.kind == "person" and dob_agreement(a.dob, b.dob) < 0


# ---- models -------------------------------------------------------------------------

BASE_MODELS = {
    "logistic": LogisticRegression(C=1.0, max_iter=2000),
    "mlp": MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=1500, random_state=0),
}


@dataclass
class MatchModel:
    name: str
    raw: object
    calibrated: object
    baseline: np.ndarray  # mean training vector, used for occlusion attributions

    def proba(self, X: np.ndarray, calibrated: bool = True) -> np.ndarray:
        model = self.calibrated if calibrated else self.raw
        return model.predict_proba(X)[:, 1]

    def explain(self, x: np.ndarray, names: list[str]) -> list[tuple[str, float, float]]:
        """Occlusion attribution: change in log-odds when a feature is reset to its value in a
        typical non-match. Log-odds do not saturate the way probabilities do near 0 and 1."""

        def log_odds(v: np.ndarray) -> float:
            p = float(np.clip(self.proba(v[None, :])[0], 1e-6, 1 - 1e-6))
            return float(np.log(p / (1 - p)))

        base = log_odds(x)
        out = []
        for i, name in enumerate(names):
            z = x.copy()
            z[i] = self.baseline[i]
            out.append((name, float(x[i]), base - log_odds(z)))
        return sorted(out, key=lambda t: -abs(t[2]))


def labelled_pairs(ds: Dataset, kind: str) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    prep = prepare(ds)
    pairs, _ = candidate_pairs(prep)
    pairs = sorted(p for p in pairs if prep[p[0]].kind == kind)
    truth = ds.truth.mention_entity
    X = np.array([features(prep[a], prep[b]) for a, b in pairs])
    y = np.array([int(truth[a] == truth[b]) for a, b in pairs])
    return X, y, pairs


def select_champions(models: dict[str, dict[str, MatchModel]], validation: Dataset) -> dict[str, str]:
    """Pick, per entity kind, the calibrated model with the lowest Brier score on a validation world."""
    champions = {}
    for kind, candidates in models.items():
        X, y, _ = labelled_pairs(validation, kind)
        champions[kind] = min(candidates, key=lambda n: brier_score_loss(y, candidates[n].proba(X)))
    return champions


def train_models(train: Dataset) -> dict[str, dict[str, MatchModel]]:
    models: dict[str, dict[str, MatchModel]] = {}
    for kind in ("company", "person"):
        X, y, _ = labelled_pairs(train, kind)
        models[kind] = {}
        for name, base in BASE_MODELS.items():
            raw = make_pipeline(StandardScaler(), clone(base)).fit(X, y)
            cal = CalibratedClassifierCV(make_pipeline(StandardScaler(), clone(base)), method="sigmoid", cv=3)
            cal.fit(X, y)
            models[kind][name] = MatchModel(name, raw, cal, X[y == 0].mean(axis=0))
    return models


# ---- clustering ---------------------------------------------------------------------

def cluster(keys: list[str], probs: dict[tuple[str, str], float], prep: dict[str, Prepared],
            threshold: float = 0.5) -> tuple[list[set[str]], int]:
    """Transitive closure of links >= threshold, then min-cut until no cannot-link pair shares a cluster."""
    g = nx.Graph()
    g.add_nodes_from(keys)
    g.add_edges_from((a, b, {"weight": p}) for (a, b), p in probs.items() if p >= threshold)
    cuts = 0
    stack = [set(c) for c in nx.connected_components(g)]
    done: list[set[str]] = []
    while stack:
        comp = stack.pop()
        bad = next(((a, b) for a, b in itertools.combinations(sorted(comp), 2) if cannot_link(prep[a], prep[b])),
                   None) if len(comp) > 1 else None
        if bad is None:
            done.append(comp)
            continue
        sub = g.subgraph(comp).copy()
        _, (left, _right) = nx.minimum_cut(sub, bad[0], bad[1], capacity="weight")
        g.remove_edges_from([(u, v) for u, v in sub.edges if (u in left) != (v in left)])
        cuts += 1
        stack.extend(set(c) for c in nx.connected_components(g.subgraph(comp)))
    return done, cuts


def pairwise_metrics(clusters: list[set[str]], truth: dict[str, str]) -> dict[str, float]:
    def n2(n: int) -> int:
        return n * (n - 1) // 2

    predicted = sum(n2(len(c)) for c in clusters)
    correct = sum(n2(v) for c in clusters for v in Counter(truth[k] for k in c).values())
    actual = sum(n2(v) for v in Counter(truth[k] for c in clusters for k in c).values())
    p = correct / predicted if predicted else 1.0
    r = correct / actual if actual else 1.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


# ---- resolved entities ----------------------------------------------------------------

@dataclass
class Entity:
    id: str
    kind: str
    members: list[str]
    name: str
    attrs: dict[str, object]
    sources: set[str]
    link_conf: float
    agreement: float | None


@dataclass
class Resolution:
    entities: dict[str, Entity]
    of: dict[str, str]  # mention key -> entity id
    probs: dict[tuple[str, str], float]
    metrics: dict[str, dict[str, float]]
    comparison: list[dict[str, object]]
    reliability: dict[str, float]
    prepared: dict[str, Prepared]
    model: dict[str, MatchModel]


def _modal(values: list) -> object | None:
    values = [v for v in values if v not in (None, "")]
    return Counter(values).most_common(1)[0][0] if values else None


def _entity(members: set[str], ds_index: dict[str, object], prep: dict[str, Prepared],
            probs: dict[tuple[str, str], float], neighbours: dict[str, list[tuple[str, float]]]) -> Entity:
    ordered = sorted(members, key=lambda k: (k.split(":")[0] != "REG", k))
    recs = [ds_index[k] for k in ordered]
    kind = prep[ordered[0]].kind
    reg = recs[0] if recs[0].source == "REG" else None
    eid = reg.key.split(":", 1)[1] if reg else "X-" + ordered[0].replace(":", "-")
    latin = [r.name for r in recs if r.name.isascii()] or [r.name for r in recs]
    name = reg.name if reg else _modal(latin)
    if kind == "company":
        fields = ("jurisdiction", "address", "incorporated", "sector", "employees", "reg_no", "foreign_reg")
        agree_fields = ("address", "incorporated", "sector", "jurisdiction")
    else:
        fields = ("dob", "nationality", "address", "national_id")
        agree_fields = ("dob", "nationality", "address")
    attrs = {f: getattr(reg, f) if reg and getattr(reg, f) is not None else _modal([getattr(r, f) for r in recs])
             for f in fields}
    rates = []
    for f in agree_fields:
        vals = [getattr(r, f) for r in recs if getattr(r, f)]
        if len(vals) >= 2:
            rates.append(Counter(vals).most_common(1)[0][1] / len(vals))
    keys = [name_key(prep[k].tokens) for k in ordered]
    if len(keys) >= 2:
        rates.append(Counter(keys).most_common(1)[0][1] / len(keys))
    attach = []
    for k in ordered:
        inside = [p for o, p in neighbours[k] if o in members]
        outside = [p for o, p in neighbours[k] if o not in members]
        conf = max(inside) if inside else 1.0
        attach.append(min(conf, 1 - max(outside, default=0.0)))
    return Entity(eid, kind, ordered, name, attrs, {r.source for r in recs}, float(min(attach)),
                  float(np.mean(rates)) if rates else None)


def resolve(ds: Dataset, models: dict[str, dict[str, MatchModel]], champions: dict[str, str],
            threshold: float = 0.5) -> Resolution:
    prep = prepare(ds)
    pairs, skipped = candidate_pairs(prep)
    truth = ds.truth.mention_entity
    index = {r.key: r for r in [*ds.companies, *ds.persons]}
    metrics: dict[str, dict[str, float]] = {}
    comparison: list[dict[str, object]] = []
    all_probs: dict[tuple[str, str], float] = {}
    entities: dict[str, Entity] = {}
    of: dict[str, str] = {}
    for kind in ("company", "person"):
        kind_pairs = sorted(p for p in pairs if prep[p[0]].kind == kind)
        keys = [k for k, p in prep.items() if p.kind == kind]
        X = np.array([features(prep[a], prep[b]) for a, b in kind_pairs])
        y = np.array([int(truth[a] == truth[b]) for a, b in kind_pairs])
        for name, model in models[kind].items():
            for calibrated in (False, True):
                p = model.proba(X, calibrated)
                probs = dict(zip(kind_pairs, p.tolist(), strict=True))
                clusters, _ = cluster(keys, probs, prep, threshold)
                pred = p >= threshold
                tp = int((pred & (y == 1)).sum())
                prec, rec = tp / max(int(pred.sum()), 1), tp / max(int(y.sum()), 1)
                comparison.append({
                    "kind": kind, "model": name, "calibration": "platt" if calibrated else "none",
                    "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                    "brier": float(brier_score_loss(y, p)), "ece": expected_calibration_error(p, y),
                    "cluster_f1": pairwise_metrics(clusters, truth)["f1"],
                })
        p = models[kind][champions[kind]].proba(X)
        probs = dict(zip(kind_pairs, p.tolist(), strict=True))
        all_probs.update(probs)
        clusters, cuts = cluster(keys, probs, prep, threshold)
        true_pairs = {pr for pr in itertools.combinations(sorted(keys), 2)
                      if truth[pr[0]] == truth[pr[1]] and prep[pr[0]].source != prep[pr[1]].source}
        found = len(true_pairs & set(kind_pairs))
        metrics[kind] = {
            **pairwise_metrics(clusters, truth),
            "pair_completeness": found / len(true_pairs) if true_pairs else 1.0,
            "reduction_ratio": 1 - len(kind_pairs) / (len(keys) * (len(keys) - 1) / 2),
            "candidate_pairs": len(kind_pairs), "oversized_blocks": skipped,
            "min_cuts": cuts, "entities": len(clusters), "true_entities": len({truth[k] for k in keys}),
        }
        neighbours: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for (a, b), pr in probs.items():
            neighbours[a].append((b, pr))
            neighbours[b].append((a, pr))
        for c in clusters:
            e = _entity(c, index, prep, probs, neighbours)
            entities[e.id] = e
            for k in c:
                of[k] = e.id
    return Resolution(entities, of, all_probs, metrics, comparison, _reliability(entities, index),
                      prep, {k: v[champions[k]] for k, v in models.items()})


def _reliability(entities: dict[str, Entity], index: dict[str, object]) -> dict[str, float]:
    """Per-feed agreement with the registry on co-resolved records (Laplace smoothed)."""
    agree: Counter = Counter()
    total: Counter = Counter()
    for e in entities.values():
        reg = index[e.members[0]]
        if reg.source != "REG":
            continue
        fields = ("address", "incorporated", "name") if e.kind == "company" else ("dob", "address", "name")
        for k in e.members[1:]:
            rec = index[k]
            for f in fields:
                a, b = getattr(reg, f), getattr(rec, f)
                if a and b:
                    total[rec.source] += 1
                    agree[rec.source] += int(a == b)
    rel = {s: (agree[s] + 1) / (total[s] + 2) for s in total}
    rel["REG"] = 0.98
    return rel
