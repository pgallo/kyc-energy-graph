import pytest

from kycgraph.similarity import jaccard, jaro_winkler, monge_elkan, name_similarity, weakest_token_similarity


def test_jaro_winkler_reference_values():
    assert jaro_winkler("martha", "marhta") == pytest.approx(0.961, abs=1e-3)
    assert jaro_winkler("dwayne", "duane") == pytest.approx(0.84, abs=1e-2)
    assert jaro_winkler("", "") == 0.0
    assert jaro_winkler("abc", "abc") == 1.0


def test_name_similarity_is_order_insensitive():
    assert name_similarity(["selma", "nedanak"], ["nedanak", "selma"]) == 1.0
    assert monge_elkan(["a"], []) == 0.0


def test_weakest_token_penalises_different_surname():
    same_given = weakest_token_similarity(["fedra", "tselanin"], ["fedra", "holanin"])
    typo = weakest_token_similarity(["fedra", "tselanin"], ["fedra", "tselannin"])
    assert typo > 0.95 > same_given


def test_jaccard():
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert jaccard(set(), {"a"}) == 0.0
