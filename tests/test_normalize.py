from kycgraph.normalize import (
    clean,
    company_tokens,
    dob_agreement,
    fold_variants,
    id_digits,
    person_tokens,
    phonetic,
    transliterate,
)


def test_cyrillic_transliteration():
    assert transliterate("Желович") == "Zhelovich"
    assert transliterate("Цельенко") == "Tselenko"


def test_legal_form_stripping_handles_dotted_forms():
    base = company_tokens("Kvaravel Grid Contractors")
    assert len(base) == 3
    assert company_tokens("Kvaravel Grid Contractors Ltd.") == base
    assert company_tokens("KVARAVEL GRID CONTRACTORS A.D.") == base
    assert company_tokens("Ulmesk P.L.C.") == ["ulmesk"]
    # a name made only of a legal-form word keeps it rather than becoming empty
    assert company_tokens("AS") == ["as"]


def test_ampersand_and_punctuation():
    assert clean("Engineering & Construction, Ltd") == "engineering and construction ltd"


def test_spelling_variants_fold_to_one_form():
    variants = ["zhelovich", "jelovich", "zhelovitch"]
    assert len({fold_variants(v) for v in variants}) == 1
    assert fold_variants("tzelenko") == fold_variants("tselenko") == fold_variants("celenko")
    assert fold_variants("ostansky") == fold_variants("ostanski")
    assert fold_variants("kharov") == fold_variants("harov")


def test_phonetic_key_tolerates_vowel_typos_and_variants():
    assert phonetic("Radanchuk") == phonetic("Radenchuk")
    assert phonetic("Zhelovich") == phonetic("Jelovitch")
    assert phonetic("Radanchuk") != phonetic("Polova")


def test_person_tokens_reorder_surname_first_convention():
    assert sorted(person_tokens("NEDANAK, Selma")) == sorted(person_tokens("Selma Nedanak"))
    assert person_tokens("Зоран Костененко") == ["zoran", "kostenenko"]


def test_identifier_digits_normalise_formats():
    assert id_digits("VR-000123") == id_digits("VR000123") == id_digits("123") == "123"
    assert id_digits(None) is None


def test_dob_agreement_levels():
    assert dob_agreement("1980-03-04", "1980-03-04") == 1.0
    assert dob_agreement("1980-03-04", "1980-04-03") == 0.5  # day/month swapped
    assert dob_agreement("1980", "1980-04-03") == 0.5        # year only
    assert dob_agreement("1981-03-04", "1980-03-04") == -1.0
    assert dob_agreement(None, "1980-03-04") == 0.0
