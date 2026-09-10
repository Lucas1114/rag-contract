"""A hand-built index, so retrieval and scoring are tested without the API.

Vectors here are placed on the unit sphere by hand rather than embedded, which
makes every expected rank in the retrieval and evaluation tests arithmetic
rather than a property of some model.
"""

import numpy as np

from rag_contract.chunking import Chunk
from rag_contract.index import Index, IndexMeta


def make_chunk(section_id: str, ordinal: int = 0) -> Chunk:
    rfc, number = section_id.split("#")
    return Chunk(
        id=f"{section_id}/{ordinal}",
        section_id=section_id,
        rfc=rfc,
        section_number=number,
        section_title=f"Section {number}",
        ordinal=ordinal,
        text=f"body of {section_id} part {ordinal}",
        citation=f"{rfc.upper()} Section {number}",
    )


def make_index(
    section_ids,
    vectors,
    question_ids=(),
    question_vectors=None,
    questions_fingerprint="q" * 64,
) -> Index:
    vectors = np.asarray(vectors, dtype=np.float32)
    if question_vectors is None:
        question_vectors = np.zeros((len(question_ids), vectors.shape[1]), np.float32)
    question_vectors = np.asarray(question_vectors, dtype=np.float32)
    chunks = [make_chunk(s, i) for i, s in enumerate(section_ids)]
    meta = IndexMeta(
        version="0" * 12,
        embedding_model="test",
        embedding_dimensions=vectors.shape[1],
        chunk_words=220,
        overlap_words=40,
        corpus_fingerprint="f" * 64,
        documents=[],
        section_count=len(set(section_ids)),
        chunk_count=len(chunks),
        questions_fingerprint=questions_fingerprint,
        question_ids=list(question_ids),
        built_at="2026-09-10T00:00:00Z",
    )
    return Index(
        meta=meta,
        chunks=chunks,
        vectors=vectors,
        question_vectors=question_vectors,
        question_ids=list(question_ids),
    )
