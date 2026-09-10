"""The committed index: chunks, vectors, and the version that identifies them.

The index version is content-addressed:

    sha256(corpus fingerprint + embedding model + chunk parameters)[:12]

Every input that could change a retrieval result is in that hash, so two
indexes with the same version hold the same vectors over the same text. Guarantee
4 returns this version in response metadata; here it is what ties a committed
vector file to the corpus revision it was built from.

Question vectors are committed alongside chunk vectors. The question set is
fixed, so they are as much a build artefact as the corpus vectors are, and
committing them is what lets the retrieval eval run in CI with no network call
and no key. They carry the sha256 of `eval/questions.yaml`, so an edited
question set cannot be scored against stale vectors.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .chunking import Chunk, ChunkParams
from .corpus import Document, corpus_fingerprint

INDEX_DIR = Path(__file__).resolve().parents[2] / "index"
META_PATH = INDEX_DIR / "meta.json"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"
VECTORS_PATH = INDEX_DIR / "vectors.npy"
QUESTION_VECTORS_PATH = INDEX_DIR / "question_vectors.npy"

VERSION_LENGTH = 12


class IndexError_(RuntimeError):
    """The committed index is missing or inconsistent with itself."""


def compute_version(documents: list[Document], model: str, params: ChunkParams) -> str:
    """The content address of an index over this corpus, model and parameters."""
    material = "\n".join(
        [corpus_fingerprint(documents), f"model={model}", params.fingerprint()]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:VERSION_LENGTH]


@dataclass(frozen=True)
class IndexMeta:
    version: str
    embedding_model: str
    embedding_dimensions: int
    chunk_words: int
    overlap_words: int
    corpus_fingerprint: str
    documents: list[dict]
    section_count: int
    chunk_count: int
    questions_fingerprint: str
    question_ids: list[str]
    built_at: str


@dataclass(frozen=True)
class Index:
    """Chunks and their vectors, loaded from the committed artefacts."""

    meta: IndexMeta
    chunks: list[Chunk]
    vectors: np.ndarray  # (n_chunks, d), L2-normalised float32
    question_vectors: np.ndarray  # (n_questions, d), L2-normalised float32
    question_ids: list[str]

    @property
    def version(self) -> str:
        return self.meta.version

    def question_vector(self, question_id: str) -> np.ndarray:
        try:
            position = self.question_ids.index(question_id)
        except ValueError as exc:
            raise IndexError_(
                f"no committed vector for question {question_id}; rebuild the index"
            ) from exc
        return self.question_vectors[position]


def write_index(
    *,
    documents: list[Document],
    chunks: list[Chunk],
    vectors: np.ndarray,
    question_ids: list[str],
    question_vectors: np.ndarray,
    questions_fingerprint: str,
    section_count: int,
    model: str,
    dimensions: int,
    params: ChunkParams,
    directory: Path = INDEX_DIR,
) -> IndexMeta:
    """Write the four index artefacts. Called only by `build-index`."""
    if vectors.shape[0] != len(chunks):
        raise IndexError_(f"{vectors.shape[0]} vectors for {len(chunks)} chunks")
    if question_vectors.shape[0] != len(question_ids):
        raise IndexError_(
            f"{question_vectors.shape[0]} vectors for {len(question_ids)} questions"
        )

    directory.mkdir(parents=True, exist_ok=True)
    meta = IndexMeta(
        version=compute_version(documents, model, params),
        embedding_model=model,
        embedding_dimensions=dimensions,
        chunk_words=params.chunk_words,
        overlap_words=params.overlap_words,
        corpus_fingerprint=corpus_fingerprint(documents),
        documents=[{"rfc": d.rfc, "sha256": d.sha256} for d in documents],
        section_count=section_count,
        chunk_count=len(chunks),
        questions_fingerprint=questions_fingerprint,
        question_ids=list(question_ids),
        built_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    (directory / "meta.json").write_text(json.dumps(asdict(meta), indent=2) + "\n")
    with (directory / "chunks.jsonl").open("w") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), sort_keys=True) + "\n")
    np.save(directory / "vectors.npy", vectors.astype(np.float32))
    np.save(directory / "question_vectors.npy", question_vectors.astype(np.float32))
    return meta


def load_index(directory: Path = INDEX_DIR) -> Index:
    """Load the committed index, checking it is consistent with itself.

    Pure file reads: no network, no key.
    """
    meta_path = directory / "meta.json"
    if not meta_path.is_file():
        raise IndexError_(
            f"no index at {directory}. Run `rag-contract build-index` "
            "(this is the one command that calls the embedding API)."
        )
    meta = IndexMeta(**json.loads(meta_path.read_text()))

    chunks = [
        Chunk(**json.loads(line))
        for line in (directory / "chunks.jsonl").read_text().splitlines()
        if line.strip()
    ]
    vectors = np.load(directory / "vectors.npy")
    question_vectors = np.load(directory / "question_vectors.npy")

    if len(chunks) != meta.chunk_count:
        raise IndexError_(
            f"meta.json declares {meta.chunk_count} chunks, chunks.jsonl holds {len(chunks)}"
        )
    if vectors.shape != (len(chunks), meta.embedding_dimensions):
        raise IndexError_(
            f"vectors.npy is {vectors.shape}, expected "
            f"{(len(chunks), meta.embedding_dimensions)}"
        )
    if question_vectors.shape != (
        len(meta.question_ids),
        meta.embedding_dimensions,
    ):
        raise IndexError_(
            f"question_vectors.npy is {question_vectors.shape}, expected "
            f"{(len(meta.question_ids), meta.embedding_dimensions)}"
        )

    return Index(
        meta=meta,
        chunks=chunks,
        vectors=vectors,
        question_vectors=question_vectors,
        question_ids=list(meta.question_ids),
    )
