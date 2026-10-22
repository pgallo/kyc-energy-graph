"""String similarity measures used by resolution and screening."""

from __future__ import annotations


def jaro(a: str, b: str) -> float:
    if a == b:
        return 1.0 if a else 0.0
    if not a or not b:
        return 0.0
    window = max(max(len(a), len(b)) // 2 - 1, 0)
    a_match = [False] * len(a)
    b_match = [False] * len(b)
    matches = 0
    for i, ch in enumerate(a):
        lo, hi = max(0, i - window), min(len(b), i + window + 1)
        for j in range(lo, hi):
            if not b_match[j] and b[j] == ch:
                a_match[i] = b_match[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    a_seq = [c for c, m in zip(a, a_match, strict=True) if m]
    b_seq = [c for c, m in zip(b, b_match, strict=True) if m]
    transpositions = sum(x != y for x, y in zip(a_seq, b_seq, strict=True)) / 2
    return (matches / len(a) + matches / len(b) + (matches - transpositions) / matches) / 3


def jaro_winkler(a: str, b: str, prefix_weight: float = 0.1) -> float:
    sim = jaro(a, b)
    prefix = 0
    for x, y in zip(a[:4], b[:4], strict=False):
        if x != y:
            break
        prefix += 1
    return sim + prefix * prefix_weight * (1 - sim)


def monge_elkan(a: list[str], b: list[str]) -> float:
    """Symmetric Monge-Elkan: each token matched to its best counterpart."""
    if not a or not b:
        return 0.0

    def one_way(x: list[str], y: list[str]) -> float:
        return sum(max(jaro_winkler(t, u) for u in y) for t in x) / len(x)

    return (one_way(a, b) + one_way(b, a)) / 2


def name_similarity(a: list[str], b: list[str]) -> float:
    """Best of sorted-token Jaro-Winkler and Monge-Elkan (order and typo tolerant)."""
    sorted_jw = jaro_winkler(" ".join(sorted(a)), " ".join(sorted(b)))
    return max(sorted_jw, monge_elkan(a, b))


def weakest_token_similarity(a: list[str], b: list[str]) -> float:
    """Best match of the worst-aligned token of the shorter name.

    A shared given name or sector word must not carry a different surname or root.
    """
    if not a or not b:
        return 0.0
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return min(max(jaro_winkler(t, u) for u in long_) for t in short)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
