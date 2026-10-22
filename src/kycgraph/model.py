"""Source records, list entries, procurement events and ground truth.

Every record is a *mention* of a real-world entity as one source saw it. Mention keys
are "<SOURCE>:<local id>", so the source of any piece of evidence is always traceable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

DOMESTIC = "VA"   # Republic of Valdoria (fictional)
OFFSHORE = "MR"   # Marenta Islands (fictional secrecy jurisdiction)

SECTORS = {
    "generation": "EN35.11",
    "renewables": "EN35.11R",
    "grid": "EN35.12",
    "distribution": "EN35.13",
    "trading": "EN35.14",
    "gas": "EN35.22",
    "epc": "EN42.22",
    "holding": "EN64.20",
    "services": "EN71.12",
}
UTILITY_SECTORS = {"grid", "distribution", "gas"}
SUPPLIER_SECTORS = {"epc", "renewables", "services"}

TYPOLOGIES = (
    "hidden_ubo",
    "circular_ownership",
    "pep_supplier",
    "shell_cluster",
    "sanctioned_exposure",
    "front_company",
)


@dataclass
class CompanyRecord:
    key: str
    source: str
    name: str
    jurisdiction: str | None = None
    address: str | None = None
    reg_no: str | None = None
    tax_id: str | None = None
    foreign_reg: str | None = None
    sector: str | None = None
    incorporated: str | None = None
    employees: int | None = None


@dataclass
class PersonRecord:
    key: str
    source: str
    name: str
    dob: str | None = None
    nationality: str | None = None
    address: str | None = None
    national_id: str | None = None


@dataclass
class Relation:
    source: str
    kind: str  # owns | director | signatory
    src: str   # mention key of owner or officer
    dst: str   # mention key of company
    pct: float | None = None


@dataclass
class ListEntry:
    list_id: str
    list_name: str  # sanctions | pep
    name: str
    aliases: list[str] = field(default_factory=list)
    dob: str | None = None
    nationality: str | None = None
    note: str = ""
    organisation: str | None = None


@dataclass
class MediaItem:
    media_id: str
    date: str
    text: str


@dataclass
class Tender:
    tender_id: str
    buyer: str     # registry mention key
    supplier: str  # registry mention key
    award_date: str
    value: float
    title: str


@dataclass
class Truth:
    mention_entity: dict[str, str] = field(default_factory=dict)
    typologies: dict[str, list[dict]] = field(default_factory=dict)
    screening: list[dict] = field(default_factory=list)


@dataclass
class Dataset:
    seed: int
    as_of: str
    companies: list[CompanyRecord]
    persons: list[PersonRecord]
    relations: list[Relation]
    watchlist: list[ListEntry]
    media: list[MediaItem]
    tenders: list[Tender]
    truth: Truth

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "dataset.json").write_text(json.dumps(asdict(self), indent=1, ensure_ascii=False))

    @classmethod
    def load(cls, directory: Path) -> Dataset:
        raw = json.loads((directory / "dataset.json").read_text())
        return cls(
            seed=raw["seed"],
            as_of=raw["as_of"],
            companies=_build(CompanyRecord, raw["companies"]),
            persons=_build(PersonRecord, raw["persons"]),
            relations=_build(Relation, raw["relations"]),
            watchlist=_build(ListEntry, raw["watchlist"]),
            media=_build(MediaItem, raw["media"]),
            tenders=_build(Tender, raw["tenders"]),
            truth=Truth(**raw["truth"]),
        )


def _build(cls, rows: list[dict]) -> list:
    names = {f.name for f in fields(cls)}
    return [cls(**{k: v for k, v in row.items() if k in names}) for row in rows]
