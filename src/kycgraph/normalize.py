"""Name, address and identifier normalisation.

The pipeline is: Unicode fold -> Cyrillic transliteration -> punctuation cleanup ->
legal-form stripping (companies) -> spelling-variant folding -> phonetic keys.
Each step is a pure function so it can be tested and explained in isolation.
"""

from __future__ import annotations

import re
import unicodedata

CYRILLIC = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya",
}

# Trailing tokens that denote a legal form rather than the name itself.
LEGAL_FORMS = {
    "ltd", "limited", "plc", "llc", "inc", "corp", "corporation", "sa", "ag", "gmbh",
    "bv", "nv", "ad", "ood", "as", "spa", "srl", "co", "company",
}

# Words too common in this sector to be useful as blocking keys.
GENERIC = {
    "energy", "power", "grid", "gas", "services", "service", "holding", "holdings",
    "capital", "investments", "group", "the", "and", "solar", "wind", "parks",
    "renewables", "engineering", "construction", "trading", "networks", "distribution",
    "generation", "thermal", "transmission", "contractors", "installations", "consulting",
    "logistics", "state", "of",
}

# Ordered spelling-variant folds. They map the transliteration conventions seen in the
# feeds (zh/j, kh/h, ts/tz/c, y/i, ch/tch, -sky/-ski) onto one canonical spelling.
_FOLDS = [
    (re.compile(r"shch"), "sh"),
    (re.compile(r"tch"), "ch"),
    (re.compile(r"tz"), "ts"),
    (re.compile(r"kh"), "h"),
    (re.compile(r"zh"), "j"),
    (re.compile(r"sky\b"), "ski"),
    (re.compile(r"c(?!h)"), "ts"),
    (re.compile(r"y"), "i"),
]

_SOUNDEX = {
    **dict.fromkeys("bfpv", "1"), **dict.fromkeys("cgjkqsxz", "2"), **dict.fromkeys("dt", "3"),
    "l": "4", **dict.fromkeys("mn", "5"), "r": "6",
}


def transliterate(text: str) -> str:
    out = []
    for ch in text:
        low = ch.lower()
        if low in CYRILLIC:
            rep = CYRILLIC[low]
            out.append(rep.capitalize() if ch != low and rep else rep)
        else:
            out.append(ch)
    return "".join(out)


def ascii_fold(text: str) -> str:
    text = transliterate(text)
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def clean(text: str) -> str:
    """Lowercase ASCII with dots removed (S.A. -> sa) and other punctuation as spaces."""
    text = ascii_fold(text).lower().replace(".", "").replace("&", " and ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_legal_form(tokens: list[str]) -> list[str]:
    toks = list(tokens)
    while len(toks) > 1 and toks[-1] in LEGAL_FORMS:
        toks.pop()
    return toks


def fold_variants(token: str) -> str:
    for pattern, rep in _FOLDS:
        token = pattern.sub(rep, token)
    return token


def phonetic(token: str, length: int = 4) -> str:
    """Soundex-style key computed on the variant-folded token."""
    token = fold_variants(token.lower())
    if not token:
        return ""
    code, last = [token[0]], _SOUNDEX.get(token[0], "")
    for ch in token[1:]:
        digit = _SOUNDEX.get(ch, "")
        if digit and digit != last:
            code.append(digit)
        last = digit
    return ("".join(code) + "000")[:length]


def person_tokens(name: str) -> list[str]:
    """Folded tokens of a person name; handles the 'SURNAME, Given' convention."""
    if "," in name:
        surname, _, given = name.partition(",")
        name = f"{given} {surname}"
    return [fold_variants(t) for t in clean(name).split()]


def company_tokens(name: str) -> list[str]:
    return [fold_variants(t) for t in strip_legal_form(clean(name).split())]


def name_key(tokens: list[str]) -> str:
    return " ".join(sorted(tokens))


def blocking_tokens(tokens: list[str]) -> list[str]:
    significant = [t for t in tokens if t not in GENERIC and len(t) > 2]
    return significant or tokens[:1]


def address_tokens(address: str | None) -> set[str]:
    if not address:
        return set()
    stop = {"street", "st", "office", "suite", "floor"}
    return {fold_variants(t) for t in clean(address).split() if t not in stop}


def id_digits(identifier: str | None) -> str | None:
    if not identifier:
        return None
    digits = re.sub(r"\D", "", identifier)
    return digits.lstrip("0") or None


def parse_dob(value: str | None) -> tuple[int, int | None, int | None] | None:
    """Return (year, a, b) where a/b are the two remaining date fields as given."""
    if not value:
        return None
    parts = value.split("-")
    try:
        if len(parts) == 1:
            return int(parts[0]), None, None
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None


def dob_agreement(a: str | None, b: str | None) -> float:
    """1 exact, 0.5 partial (year only, or day/month swapped), -1 conflict, 0 missing."""
    pa, pb = parse_dob(a), parse_dob(b)
    if pa is None or pb is None:
        return 0.0
    if pa[0] != pb[0]:
        return -1.0
    if pa[1] is None or pb[1] is None:
        return 0.5
    if pa[1:] == pb[1:]:
        return 1.0
    if pa[1:] == (pb[2], pb[1]):
        return 0.5
    return -1.0
