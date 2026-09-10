"""Brute-force cosine retrieval over the committed vectors.

868 chunks of 1024 float32 is 3.5 MB. A single numpy matrix-vector product
ranks all of them in well under a millisecond, which leaves the latency budget
of guarantee 5 entirely to the answer step. A vector database would add a
component to version, operate and cut over for no measurable benefit — and
knowing which problems do not need one is part of the point.

Vectors are L2-normalised at build time, so cosine similarity is a dot product.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunking import Chunk
from .index import Index


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk, and where it placed."""

    rank: int  # 1-based
    score: float  # cosine similarity, in [-1, 1]
    chunk: Chunk

    @property
    def section_id(self) -> str:
        """The annotation contract: every hit resolves to its source section."""
        return self.chunk.section_id


def search(index: Index, query_vector: np.ndarray, k: int = 10) -> list[Hit]:
    """The top k chunks by cosine similarity, best first."""
    if k <= 0:
        raise ValueError("k must be positive")
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    if query.shape[0] != index.vectors.shape[1]:
        raise ValueError(
            f"query has {query.shape[0]} dimensions, index has {index.vectors.shape[1]}"
        )

    scores = index.vectors @ query
    k = min(k, scores.shape[0])
    # argpartition finds the top k without sorting all 868, then sort just those.
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top], kind="stable")]
    return [
        Hit(rank=rank, score=float(scores[position]), chunk=index.chunks[position])
        for rank, position in enumerate(top, start=1)
    ]


def ranked_sections(hits: list[Hit]) -> list[str]:
    """Section ids in hit order, deduplicated, keeping the best rank of each."""
    seen: list[str] = []
    for hit in hits:
        if hit.section_id not in seen:
            seen.append(hit.section_id)
    return seen
