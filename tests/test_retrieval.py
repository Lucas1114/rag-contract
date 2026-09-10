import numpy as np
import pytest

from rag_contract.retrieval import ranked_sections, search


def test_hits_come_back_best_first(tiny_index):
    hits = search(tiny_index, np.array([1.0, 0.0], dtype=np.float32), k=4)
    assert [h.rank for h in hits] == [1, 2, 3, 4]
    assert [h.section_id for h in hits] == [
        "rfc9110#1",
        "rfc9110#2",
        "rfc9111#3",
        "rfc9111#4",
    ]
    assert [round(h.score, 3) for h in hits] == [1.0, 0.8, 0.6, 0.0]


def test_every_hit_resolves_to_its_source_section(tiny_index):
    for hit in search(tiny_index, np.array([1.0, 0.0], dtype=np.float32), k=4):
        assert hit.chunk.id.startswith(hit.section_id + "/")


def test_k_larger_than_the_index_returns_everything(tiny_index):
    assert len(search(tiny_index, np.array([1.0, 0.0], dtype=np.float32), k=99)) == 4


def test_k_must_be_positive(tiny_index):
    with pytest.raises(ValueError):
        search(tiny_index, np.array([1.0, 0.0], dtype=np.float32), k=0)


def test_dimension_mismatch_is_rejected(tiny_index):
    with pytest.raises(ValueError, match="dimensions"):
        search(tiny_index, np.array([1.0, 0.0, 0.0], dtype=np.float32))


def test_ranked_sections_keeps_the_best_rank_of_each(tiny_index):
    hits = search(tiny_index, np.array([1.0, 0.0], dtype=np.float32), k=4)
    # Two chunks share no section here, so order is preserved as-is.
    assert ranked_sections(hits) == [
        "rfc9110#1",
        "rfc9110#2",
        "rfc9111#3",
        "rfc9111#4",
    ]
    assert ranked_sections(hits + hits) == ranked_sections(hits)
