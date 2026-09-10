import numpy as np
import pytest

from rag_contract.chunking import ChunkParams
from rag_contract.corpus import load_documents
from rag_contract.embedding import normalise
from rag_contract.index import IndexError_, compute_version, load_index


def test_version_is_stable_for_the_same_inputs():
    documents = load_documents()
    params = ChunkParams()
    assert compute_version(documents, "m", params) == compute_version(
        documents, "m", params
    )
    assert len(compute_version(documents, "m", params)) == 12


def test_version_changes_with_the_embedding_model():
    documents = load_documents()
    params = ChunkParams()
    assert compute_version(documents, "a", params) != compute_version(
        documents, "b", params
    )


def test_version_changes_with_the_chunk_parameters():
    documents = load_documents()
    assert compute_version(documents, "m", ChunkParams()) != compute_version(
        documents, "m", ChunkParams(chunk_words=100, overlap_words=10)
    )


def test_a_missing_index_says_how_to_build_one(tmp_path):
    with pytest.raises(IndexError_, match="build-index"):
        load_index(tmp_path)


def test_normalise_puts_vectors_on_the_unit_sphere():
    unit = normalise(np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32))
    assert np.allclose(np.linalg.norm(unit, axis=1), 1.0)


def test_normalise_leaves_a_zero_vector_alone():
    unit = normalise(np.array([[0.0, 0.0]], dtype=np.float32))
    assert np.allclose(unit, 0.0)


def test_question_vectors_are_addressed_by_question_id(tiny_index):
    assert np.allclose(tiny_index.question_vector("qa"), [1.0, 0.0])
    with pytest.raises(IndexError_, match="rebuild"):
        tiny_index.question_vector("nope")
