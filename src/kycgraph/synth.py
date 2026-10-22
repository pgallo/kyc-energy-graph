"""Seeded generator of a fictional energy-sector corporate world and its noisy sources.

The generator first builds a *true* world (companies, persons, holdings, roles, tenders),
injects the risk typologies with ground truth, and then renders that world through four
sources: a clean domestic registry and three commercial feeds with different coverage
and noise profiles. Nothing here refers to real people, companies, places or lists.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta

from .model import (
    DOMESTIC,
    OFFSHORE,
    SECTORS,
    TYPOLOGIES,
    CompanyRecord,
    Dataset,
    ListEntry,
    MediaItem,
    PersonRecord,
    Relation,
    Tender,
    Truth,
)

AS_OF = date(2026, 6, 30)
FOREIGN = ("KS", "OL")  # Kestrany and Olvenia, fictional neighbours

GIVEN = [
    "Arven", "Boris", "Daria", "Edin", "Fedra", "Galen", "Irena", "Kasimir", "Lumira",
    "Milan", "Nadia", "Oleg", "Petra", "Radan", "Selma", "Teodor", "Vesna", "Zoran",
    "Ilka", "Marek", "Anika", "Dusan", "Lada", "Stefan", "Vera", "Roman", "Elena", "Timur",
]
SUR_PRE = [
    "Zhel", "Khar", "Tsel", "Bor", "Ost", "Vel", "Dar", "Mir", "Kras", "Lub", "Ned",
    "Rad", "Sav", "Tom", "Vran", "Grad", "Chern", "Shev", "Kost", "Pol", "Zhur", "Khol",
]
SUR_MID = ["", "en", "o", "ar", "ov", "in", "ek", "an"]
SUR_SUF = ["ov", "ova", "ski", "ich", "enko", "ets", "ak", "in", "yev", "chuk"]
ROOT_A = ["Kvar", "Dras", "Ulm", "Vrest", "Zelt", "Orsk", "Brav", "Telm", "Skar", "Grend",
          "Hald", "Murv", "Pesk", "Ryst", "Lond", "Fjel"]
ROOT_B = ["avel", "ingor", "eth", "umar", "ovrik", "esk", "altor", "imund", "arex", "ostel"]
SECTOR_WORDS = {
    "generation": ["Power Generation", "Thermal Power"],
    "renewables": ["Solar", "Wind Parks", "Renewables"],
    "grid": ["Grid", "Transmission"],
    "distribution": ["Power Distribution", "Electric Networks"],
    "trading": ["Energy Trading", "Power Trading"],
    "gas": ["Gas Distribution", "Gas Networks"],
    "epc": ["Engineering & Construction", "Grid Contractors", "Installations"],
    "holding": ["Holding", "Capital", "Investments"],
    "services": ["Services", "Consulting", "Logistics"],
}
LEGAL = {DOMESTIC: ["Ltd", "AD", "OOD", "PLC"], OFFSHORE: ["Ltd", "Inc", "Corp"],
         "KS": ["GmbH", "AG"], "OL": ["SA", "Ltd"]}
STREETS = ["Lipova", "Kovarna", "Station", "Harbour", "Mill", "Rosen", "Tower", "Granite",
           "Linden", "Quarry", "Weir", "Chapel"]
CITIES = ["Velmira", "Dornhavn", "Sarkovy", "Lunetz", "Ostrel", "Brannick"]
TENDER_TITLES = ["Substation maintenance", "Pipeline inspection", "Meter replacement",
                 "Transformer supply", "Line rerouting", "SCADA upgrade", "Valve station works"]
LAT2CYR = [
    ("shch", "щ"), ("zh", "ж"), ("kh", "х"), ("ts", "ц"), ("ch", "ч"), ("sh", "ш"), ("yu", "ю"), ("ya", "я"),
    *zip("abvgdezijklmnoprstufhcwy", "абвгдезийклмнопрстуфхквй", strict=True),
]
TRANSLIT_RULES = [("zh", "j"), ("kh", "h"), ("ts", "tz"), ("ts", "c"), ("y", "i"),
                  ("ch", "tch"), ("ski", "sky")]


@dataclass(frozen=True)
class FeedProfile:
    name: str
    coverage: float
    typo: float
    translit: float
    cyrillic: float
    swap: float
    stale_address: float
    dob_missing: float
    dob_year_only: float
    dob_swap: float
    pct_noise: float
    reg_no: float
    tax_id: float
    foreign_reg: float
    national_id: float
    drop_relation: float


FEEDS = (
    FeedProfile("A", 0.75, 0.06, 0.15, 0.0, 0.10, 0.10, 0.15, 0.0, 0.0, 0.0, 0.6, 0.0, 0.8, 0.4, 0.05),
    FeedProfile("B", 0.65, 0.12, 0.35, 0.0, 0.30, 0.25, 0.20, 0.2, 0.0, 3.0, 0.0, 0.50, 0.6, 0.0, 0.10),
    FeedProfile("C", 0.60, 0.18, 0.30, 0.3, 0.20, 0.30, 0.30, 0.0, 0.15, 8.0, 0.0, 0.0, 0.0, 0.0, 0.15),
)


def to_cyrillic(text: str) -> str:
    words = []
    for word in text.split():
        low, out, i = word.lower(), [], 0
        while i < len(low):
            for lat, cyr in LAT2CYR:
                if low.startswith(lat, i):
                    out.append(cyr)
                    i += len(lat)
                    break
            else:
                out.append(low[i])
                i += 1
        words.append("".join(out).capitalize())
    return " ".join(words)


class _World:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.companies: dict[str, dict] = {}
        self.persons: dict[str, dict] = {}
        self.owns: list[tuple[str, str, float]] = []
        self.roles: list[tuple[str, str, str]] = []
        self.tenders: list[dict] = []
        self.truth = {t: [] for t in TYPOLOGIES}
        self.forced: set[tuple[str, str]] = set()
        self.names: set[str] = set()
        self.lists: list[ListEntry] = []
        self.screen_truth: list[dict] = []
        self.media: list[MediaItem] = []

    # ---- entity factories -------------------------------------------------------
    def _date_between(self, start: date, end: date) -> date:
        return start + timedelta(days=self.rng.randint(0, max((end - start).days, 0)))

    def _address(self, jurisdiction: str) -> str:
        r = self.rng
        if jurisdiction == OFFSHORE:
            return f"Suite {r.randint(1, 40)}, {r.choice(['Coral', 'Reef', 'Pelican'])} House, Port Anselm"
        if jurisdiction in FOREIGN:
            return f"{r.randint(1, 90)} {r.choice(STREETS)}gasse, Kestburg"
        return f"{r.randint(1, 120)} {r.choice(STREETS)} Street, {r.choice(CITIES)}"

    def _surname(self) -> str:
        r = self.rng
        return (r.choice(SUR_PRE) + r.choice(SUR_MID) + r.choice(SUR_SUF)).capitalize()

    def person(self, nationality: str = DOMESTIC, given: str | None = None,
               surname: str | None = None) -> str:
        r = self.rng
        if given is None:
            while True:
                given, surname = r.choice(GIVEN), self._surname()
                if f"{given} {surname}" not in self.names:
                    break
        self.names.add(f"{given} {surname}")
        pid = f"P{len(self.persons) + 1:04d}"
        self.persons[pid] = {
            "id": pid, "given": given, "surname": surname,
            "dob": self._date_between(date(1950, 1, 1), date(1995, 12, 31)),
            "nationality": nationality, "address": self._address(nationality),
            "nid": f"NID-{r.randint(1_000_000, 9_999_999)}" if nationality == DOMESTIC else None,
        }
        return pid

    def company(self, sector: str, jurisdiction: str = DOMESTIC, inc: date | None = None,
                employees: int | None = None, address: str | None = None) -> str:
        r = self.rng
        while True:
            root = r.choice(ROOT_A) + r.choice(["", "en", "ar", "ul"]) + r.choice(ROOT_B)
            if root not in self.names:
                break
        self.names.add(root)
        cid = f"C{len(self.companies) + 1:04d}"
        n = len(self.companies) + 1
        self.companies[cid] = {
            "id": cid, "root": root, "name": f"{root} {r.choice(SECTOR_WORDS[sector])}",
            "legal": r.choice(LEGAL[jurisdiction]), "sector": sector, "jurisdiction": jurisdiction,
            "address": address or self._address(jurisdiction),
            "prev_address": self._address(jurisdiction) if r.random() < 0.4 else None,
            "inc": inc or self._date_between(date(1992, 1, 1), date(2022, 12, 31)),
            "employees": employees if employees is not None else r.randint(5, 400),
            "reg_no": f"VR-{n * 37 % 1_000_000:06d}" if jurisdiction == DOMESTIC else None,
            "tax_id": f"VA{r.randint(100_000_000, 999_999_999)}" if jurisdiction == DOMESTIC else None,
            "foreign_reg": None if jurisdiction == DOMESTIC else f"{jurisdiction}-{r.randint(10_000, 99_999)}",
        }
        return cid

    def own(self, owner: str, company: str, pct: float, forced: bool = False) -> None:
        self.owns.append((owner, company, round(pct, 1)))
        if forced:
            self.forced.add((owner, company))

    def split_owners(self, company: str, owners: list[str]) -> None:
        weights = [self.rng.uniform(1, 4) for _ in owners]
        total = sum(weights)
        for owner, w in zip(owners, weights, strict=True):
            self.own(owner, company, 100 * w / total)

    def role(self, person: str, company: str, role: str = "director") -> None:
        self.roles.append((person, company, role))

    def board(self, company: str, pool: list[str], k: int | None = None) -> None:
        k = k or self.rng.randint(1, 3)
        for p in self.rng.sample(pool, k):
            self.role(p, company)
        if self.rng.random() < 0.5:
            self.role(self.rng.choice(pool), company, "signatory")

    def tender(self, buyer: str, supplier: str, award: date) -> None:
        r = self.rng
        self.tenders.append({
            "buyer": buyer, "supplier": supplier, "award": award,
            "value": round(r.uniform(0.2, 25.0), 2),
            "title": f"{r.choice(TENDER_TITLES)}, lot {r.randint(1, 9)}",
        })

    def by_sector(self, *sectors: str) -> list[str]:
        return [c for c, v in self.companies.items() if v["sector"] in sectors]


def _build_world(seed: int) -> _World:
    w = _World(seed)
    r = w.rng
    domestic = [w.person() for _ in range(340)]
    foreigners = [w.person(r.choice(FOREIGN)) for _ in range(20)]

    state = w.company("holding", employees=40)
    w.companies[state]["name"] = "Valdoria State Energy Holding"
    w.board(state, domestic, 3)
    utilities = []
    for sector, n in (("distribution", 4), ("grid", 3), ("gas", 4)):
        for _ in range(n):
            u = w.company(sector, employees=r.randint(300, 3000))
            w.own(state, u, 100.0)
            w.board(u, domestic, r.randint(2, 3))
            utilities.append(u)
    holdings = []
    for _ in range(40):
        h = w.company("holding", employees=r.randint(1, 8))
        w.split_owners(h, r.sample(domestic, r.randint(1, 2)))
        w.board(h, domestic)
        holdings.append(h)
    for sector, n in (("epc", 45), ("renewables", 30), ("services", 30), ("trading", 18), ("generation", 8)):
        for _ in range(n):
            c = w.company(sector)
            owners = [r.choice(holdings) if r.random() < 0.4 else r.choice(domestic)
                      for _ in range(r.randint(1, 3))]
            w.split_owners(c, list(dict.fromkeys(owners)))
            w.board(c, domestic)
    # Legitimate foreign groups: layered but transparent (decoys for hidden_ubo).
    for _ in range(3):
        owner = r.choice(foreigners)
        top = w.company("holding", "KS")
        mid = w.company("holding", "KS")
        op = w.company(r.choice(["renewables", "generation"]))
        w.own(owner, top, 100.0, forced=True)
        w.own(top, mid, 100.0, forced=True)
        w.own(mid, op, 100.0, forced=True)
        w.board(op, domestic, 2)
    suppliers = w.by_sector("epc", "renewables", "services")
    for _ in range(150):
        award = w._date_between(date(2023, 1, 1), AS_OF)
        eligible = [s for s in suppliers if w.companies[s]["inc"] < award - timedelta(days=730)]
        w.tender(r.choice(utilities), r.choice(eligible), award)

    used = set()
    nominees = [w.person() for _ in range(2)]
    _inject(w, domestic, foreigners, utilities, nominees, used)
    _make_lists(w, domestic, used)
    _make_media(w)
    return w


def _inject(w: _World, domestic: list[str], foreigners: list[str], utilities: list[str],
            nominees: list[str], used: set[str]) -> None:
    r = w.rng
    recent = (AS_OF - timedelta(days=700), AS_OF - timedelta(days=60))

    def fresh() -> str:
        p = r.choice([d for d in domestic if d not in used])
        used.add(p)
        return p

    for i in range(3):  # hidden beneficial owner behind offshore layers and nominees
        hidden = w.person(r.choice(FOREIGN))
        x1, x2 = w.company("holding", OFFSHORE, employees=0), w.company("holding", OFFSHORE, employees=0)
        d = w.company("holding", employees=1)
        t = w.company(r.choice(["epc", "renewables"]))
        w.own(hidden, x1, 80.0, forced=True)
        w.own(w.person(OFFSHORE), x1, 20.0)
        w.own(x1, x2, 75.0, forced=True)
        w.own(x2, d, 70.0, forced=True)
        w.own(fresh(), d, 30.0)
        w.own(d, t, 90.0)
        w.own(fresh(), t, 10.0)
        for c in (d, t):
            w.role(nominees[i % 2], c)
        w.role(nominees[(i + 1) % 2], t, "signatory")
        w.truth["hidden_ubo"].append({"companies": [t, d], "persons": [hidden]})
    for _ in range(2):  # circular ownership
        a, b, c = (w.company(r.choice(["trading", "generation", "holding"])) for _ in range(3))
        w.own(a, b, 30.0)
        w.own(b, c, 35.0)
        w.own(c, a, 40.0)
        for x, pct in ((a, 60.0), (b, 70.0), (c, 65.0)):
            w.own(fresh(), x, pct)
            w.board(x, domestic)
        w.truth["circular_ownership"].append({"companies": [a, b, c], "persons": []})
    peps = []
    for i in range(3):  # PEP behind a supplier that wins tenders from a utility
        pep, buyer = w.person(), r.choice(utilities)
        s = w.company("epc", inc=w._date_between(date(2016, 1, 1), date(2021, 1, 1)))
        w.own(pep, s, 51.0)
        w.own(fresh(), s, 49.0)
        w.role(pep if i == 0 else fresh(), s)
        for _ in range(r.randint(2, 3)):
            w.tender(buyer, s, w._date_between(date(2024, 1, 1), AS_OF))
        org = "Ministry of Energy" if i == 2 else w.companies[buyer]["name"]
        peps.append((pep, "Deputy Minister for Energy" if i == 2 else "Board member", org))
        w.truth["pep_supplier"].append({"companies": [s], "persons": [pep]})
    w.pep_typology = peps
    for k in range(2):  # shell companies sharing an address and officers
        address = f"{r.randint(1, 60)} {r.choice(STREETS)} Street, {r.choice(CITIES)}, office {k + 3}"
        shared = fresh()
        members = []
        for _ in range(r.randint(4, 5)):
            s = w.company(r.choice(["services", "trading", "holding"]),
                          inc=w._date_between(*recent), employees=r.choice([0, 0, 1]), address=address)
            w.own(fresh(), s, 100.0)
            w.role(nominees[k], s)
            w.role(shared, s, "signatory")
            members.append(s)
        w.truth["shell_cluster"].append({"companies": members, "persons": [nominees[k], shared]})
    hub = f"{r.randint(1, 60)} Tower Street, Velmira, business centre"
    for _ in range(6):  # decoy: busy but legitimate business centre
        c = w.company(r.choice(["services", "trading"]), employees=r.randint(10, 80), address=hub)
        w.own(fresh(), c, 100.0)
        w.board(c, domestic)
    w.sanctioned = []
    for _ in range(2):  # sanctioned person reached through three ownership hops
        s = w.person(r.choice(FOREIGN))
        x = w.company("holding", OFFSHORE, employees=0)
        y = w.company("holding", employees=2)
        z = w.company("epc")
        w.own(s, x, 70.0, forced=True)
        w.own(w.person(OFFSHORE), x, 30.0)
        w.own(x, y, 60.0, forced=True)
        w.own(fresh(), y, 40.0)
        w.own(y, z, 55.0)
        w.own(fresh(), z, 45.0)
        w.board(y, domestic, 1)
        w.board(z, domestic, 2)
        w.tender(r.choice(utilities), z, w._date_between(date(2024, 1, 1), AS_OF))
        w.sanctioned.append(s)
        w.truth["sanctioned_exposure"].append({"companies": [x, y, z], "persons": [s]})
    for _ in range(3):  # front company created shortly before winning a tender
        award = w._date_between(date(2024, 6, 1), AS_OF - timedelta(days=10))
        f = w.company(r.choice(["epc", "services"]), inc=award - timedelta(days=r.randint(20, 90)),
                      employees=r.randint(0, 2))
        w.own(fresh(), f, 100.0)
        w.role(fresh(), f)
        w.tender(r.choice(utilities), f, award)
        w.truth["front_company"].append({"companies": [f], "persons": []})
    for _ in range(5):  # decoys: new companies with no awards
        c = w.company("services", inc=w._date_between(*recent), employees=r.randint(0, 3))
        w.own(fresh(), c, 100.0)
        w.role(fresh(), c)
    for _ in range(2):  # decoys: awards long after incorporation
        award = w._date_between(date(2025, 6, 1), AS_OF)
        c = w.company("epc", inc=award - timedelta(days=r.randint(400, 600)), employees=r.randint(5, 30))
        w.own(fresh(), c, 100.0)
        w.role(fresh(), c)
        w.tender(r.choice(utilities), c, award)
    for _ in range(4):  # nominees also sit on unrelated boards (makes them high-degree)
        w.role(r.choice(nominees), r.choice(w.by_sector("services", "trading")))


def _make_lists(w: _World, domestic: list[str], used: set[str]) -> None:
    r = w.rng
    n = 0

    def entry(list_name: str, person: dict | None, name: str, aliases: list[str], dob: str | None,
              nat: str, note: str, org: str | None = None) -> None:
        nonlocal n
        n += 1
        prefix = "SL" if list_name == "sanctions" else "PL"
        w.lists.append(ListEntry(f"{prefix}-{n:04d}", list_name, name, aliases, dob, nat, note, org))
        if person is not None:
            w.screen_truth.append({"entity": person["id"], "list_id": f"{prefix}-{n:04d}"})

    for i, pid in enumerate(w.sanctioned):
        p = w.persons[pid]
        full = f"{p['given']} {p['surname']}"
        dob = p["dob"].isoformat() if i == 0 else str(p["dob"].year)
        entry("sanctions", p, full.upper(), [to_cyrillic(full), _translit(r, full)], dob,
              p["nationality"], "Asset freeze, fictional programme 7")
    for pid, position, org in w.pep_typology:
        p = w.persons[pid]
        entry("pep", p, f"{p['given']} {p['surname']}", [], p["dob"].isoformat(), DOMESTIC, position, org)
    winners = {t["supplier"] for t in w.tenders}
    parents = {o for o, c, _ in w.owns if c in winners}
    exposed = winners | parents
    linked = {o for o, c, _ in w.owns if c in exposed} | {p for p, c, _ in w.roles if c in exposed}
    genuine = [d for d in domestic if d not in used and d not in linked][:8]
    for pid in genuine:  # PEPs not involved in any typology
        used.add(pid)
        p = w.persons[pid]
        entry("pep", p, f"{p['given']} {p['surname']}", [], p["dob"].isoformat(), DOMESTIC,
              "Member of regional council", "Velmira Regional Council")
    for pid in genuine[:2]:  # namesakes of PEPs in the registry: a screening trap
        p = w.persons[pid]
        twin = w.person(DOMESTIC, p["given"], p["surname"])
        w.role(twin, r.choice(w.by_sector("trading", "generation")))
    for _ in range(14):  # list entries with no counterpart in the world
        given, sur = r.choice(GIVEN), w._surname()
        dob = f"19{r.randint(50, 95)}-0{r.randint(1, 9)}-1{r.randint(0, 9)}"
        entry(r.choice(["sanctions", "pep"]), None, f"{given} {sur}", [], dob, r.choice(FOREIGN), "Fictional listing")
    for pid in r.sample([d for d in domestic if d not in used], 4):  # decoys: same name, other person
        p = w.persons[pid]
        entry(r.choice(["sanctions", "pep"]), None, f"{p['given']} {p['surname']}", [],
              f"{p['dob'].year + r.choice([-9, -6, 7, 11])}-03-14", r.choice(FOREIGN), "Fictional listing")


def _translit(r: random.Random, text: str) -> str:
    low = text.lower()
    options = [(a, b) for a, b in TRANSLIT_RULES if a in low]
    if not options:
        return text
    a, b = r.choice(options)
    return " ".join(t.capitalize() for t in low.replace(a, b, 1).split())


def _typo(r: random.Random, word: str) -> str:
    if len(word) < 4:
        return word
    i = r.randint(1, len(word) - 2)
    op = r.choice(["del", "sub", "swap", "ins"])
    letter = r.choice("aeiourstnl")
    if op == "del":
        return word[:i] + word[i + 1:]
    if op == "sub":
        return word[:i] + letter + word[i + 1:]
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word[:i] + letter + word[i:]


def _make_media(w: _World) -> None:
    r = w.rng
    adverse = [
        "{name} named in inquiry into tender irregularities at a regional utility",
        "Prosecutors examine bribery claims involving {name}",
        "Auditors flag fraud risk in payments routed by {name} to offshore vehicles",
        "Investigators allege {name} took part in a sanctions evasion scheme",
    ]
    neutral = ["{name} opens a new solar park near Velmira", "{name} reports stable annual results",
               "{name} sponsors a local engineering scholarship", "{name} signs a framework maintenance agreement"]
    subjects = [
        w.companies[w.truth["hidden_ubo"][0]["companies"][0]]["name"],
        w.companies[w.truth["pep_supplier"][1]["companies"][0]]["name"],
        w.companies[w.truth["front_company"][0]["companies"][0]]["name"],
    ]
    s = w.persons[w.sanctioned[0]]
    subjects.append(_translit(r, f"{s['given']} {s['surname']}"))
    subjects.append(f"{r.choice(GIVEN)} {w._surname()}")  # adverse item about nobody we know
    for i, name in enumerate(subjects):
        template = adverse[i % len(adverse)]
        w.media.append(MediaItem(f"M{len(w.media) + 1:03d}", f"2026-0{r.randint(1, 6)}-1{r.randint(0, 9)}",
                                 template.format(name=name)))
    for _ in range(25):
        c = w.companies[r.choice(list(w.companies))]
        w.media.append(MediaItem(f"M{len(w.media) + 1:03d}", f"2026-0{r.randint(1, 6)}-0{r.randint(1, 9)}",
                                 r.choice(neutral).format(name=c["name"])))


# ---- rendering into sources -------------------------------------------------------

def _render(w: _World, seed: int) -> Dataset:
    r = random.Random(seed * 7919 + 1)
    companies: list[CompanyRecord] = []
    persons: list[PersonRecord] = []
    relations: list[Relation] = []
    truth = Truth()
    reg_key: dict[str, str] = {}

    for cid, c in w.companies.items():
        if c["jurisdiction"] != DOMESTIC:
            continue
        key = f"REG:{c['reg_no']}"
        reg_key[cid] = key
        companies.append(CompanyRecord(key, "REG", f"{c['name']} {c['legal']}", DOMESTIC, c["address"],
                                       c["reg_no"], c["tax_id"], None, SECTORS[c["sector"]],
                                       c["inc"].isoformat(), c["employees"]))
        truth.mention_entity[key] = cid
    for pid, p in w.persons.items():
        if p["nid"] is None:
            continue
        key = f"REG:{p['nid']}"
        reg_key[pid] = key
        persons.append(PersonRecord(key, "REG", f"{p['given']} {p['surname']}", p["dob"].isoformat(),
                                    DOMESTIC, p["address"], p["nid"]))
        truth.mention_entity[key] = pid
    for owner, company, pct in w.owns:
        if owner in reg_key and company in reg_key:
            relations.append(Relation("REG", "owns", reg_key[owner], reg_key[company], pct))
    for person, company, role in w.roles:
        if person in reg_key and company in reg_key:
            relations.append(Relation("REG", role, reg_key[person], reg_key[company]))

    forced_feed = {edge: r.choice(FEEDS).name for edge in sorted(w.forced)}
    for feed in FEEDS:
        covered = {e for e in list(w.companies) + list(w.persons) if r.random() < feed.coverage}
        for (u, v), name in forced_feed.items():
            if name == feed.name:
                covered |= {u, v}
        local: dict[str, str] = {}
        for i, eid in enumerate(sorted(covered)):
            key = f"{feed.name}:{feed.name}{i + 1:05d}"
            local[eid] = key
            truth.mention_entity[key] = eid
            if eid in w.companies:
                companies.append(_feed_company(r, feed, key, w.companies[eid]))
            else:
                persons.append(_feed_person(r, feed, key, w.persons[eid]))
        for owner, company, pct in w.owns:
            if owner in local and company in local:
                forced = forced_feed.get((owner, company)) == feed.name
                if forced or r.random() > feed.drop_relation:
                    noisy = min(100.0, max(0.5, pct + r.uniform(-feed.pct_noise, feed.pct_noise)))
                    relations.append(Relation(feed.name, "owns", local[owner], local[company], round(noisy, 1)))
        for person, company, role in w.roles:
            if person in local and company in local and r.random() > feed.drop_relation:
                relations.append(Relation(feed.name, role, local[person], local[company]))

    tenders = [Tender(f"T{i + 1:04d}", reg_key[t["buyer"]], reg_key[t["supplier"]], t["award"].isoformat(),
                      t["value"], t["title"]) for i, t in enumerate(sorted(w.tenders, key=lambda t: t["award"]))]
    truth.typologies = w.truth
    truth.screening = w.screen_truth
    return Dataset(seed, AS_OF.isoformat(), companies, persons, relations, w.lists, w.media, tenders, truth)


def _feed_company(r: random.Random, f: FeedProfile, key: str, c: dict) -> CompanyRecord:
    root = _typo(r, c["root"]) if r.random() < f.typo else c["root"]
    name = c["name"].replace(c["root"], root)
    if r.random() < f.typo:  # short-form trading name without the sector words
        name = root
    legal = c["legal"]
    roll = r.random()
    if roll < 0.2:
        legal = ""
    elif roll < 0.3:
        legal = {"Ltd": "Limited", "AD": "A.D.", "PLC": "P.L.C."}.get(legal, legal)
    name = f"{name} {legal}".strip()
    if r.random() < 0.15:
        name = name.upper()
    address = c["prev_address"] if c["prev_address"] and r.random() < f.stale_address else c["address"]
    reg = None
    if c["reg_no"] and r.random() < f.reg_no:
        digits = c["reg_no"][3:]
        reg = r.choice([c["reg_no"], "VR" + digits, digits, digits.lstrip("0")])
    inc = c["inc"]
    if f.pct_noise > 5 and r.random() < 0.2:
        inc = inc.replace(year=inc.year + r.choice([-1, 1]), day=min(inc.day, 28))
    employees = c["employees"] if f.name == "A" else (None if r.random() < 0.5 else
                                                     int(c["employees"] * r.uniform(0.8, 1.2)))
    return CompanyRecord(
        key, f.name, name, c["jurisdiction"], address, reg,
        c["tax_id"] if r.random() < f.tax_id else None,
        c["foreign_reg"] if c["foreign_reg"] and r.random() < f.foreign_reg else None,
        SECTORS[c["sector"]] if r.random() < 0.7 else None, inc.isoformat(), employees,
    )


def _feed_person(r: random.Random, f: FeedProfile, key: str, p: dict) -> PersonRecord:
    given, surname = p["given"], p["surname"]
    if r.random() < f.translit:
        surname = _translit(r, surname)
    if r.random() < f.typo:
        surname = _typo(r, surname)
    if r.random() < f.typo / 2:
        given = _typo(r, given)
    if r.random() < f.cyrillic:
        name = to_cyrillic(f"{given} {surname}")
    elif r.random() < f.swap:
        name = f"{surname.upper()}, {given}"
    else:
        name = f"{given} {surname}"
    d = p["dob"]
    dob: str | None = d.isoformat()
    if r.random() < f.dob_missing:
        dob = None
    elif r.random() < f.dob_year_only:
        dob = str(d.year)
    elif r.random() < f.dob_swap and d.day <= 12 and d.day != d.month:
        dob = f"{d.year}-{d.day:02d}-{d.month:02d}"
    return PersonRecord(
        key, f.name, name, dob, p["nationality"] if r.random() < 0.8 else None,
        p["address"] if r.random() > f.stale_address else None,
        p["nid"] if p["nid"] and r.random() < f.national_id else None,
    )


def generate(seed: int = 7) -> Dataset:
    return _render(_build_world(seed), seed)
