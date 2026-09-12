"""When the committed index stops describing the corpus.

The index version is `sha256(corpus + embedding model + chunk parameters)`, so
the question "is this index still the right one?" has an exact answer: rebuild
the address from what is on disk now and compare. If it differs, the committed
vectors were built from inputs this commit no longer has, and every number the
eval reports is a number about a corpus that is gone.

The trigger is a red build, not a rebuild
-----------------------------------------

Nothing here re-embeds anything. It cannot: embedding calls a paid API with a
key, and the entire evaluation story rests on CI having neither. So a corpus
change does not trigger an automatic rebuild — it triggers a failure that names
the command a human runs, and the rebuilt vectors arrive in the same commit as
the corpus change or the build stays red. That is the strongest form available
to a project whose artefacts are committed by hand, and it is the same shape as
every other guarantee here: the machine cannot do the work, so the machine
refuses to let the work be skipped.

Divergence is reported per input rather than as one boolean, because "the
index is stale" is not actionable and "corpus/rfc9110.txt changed" is.

Two hashes, not one
-------------------

The question set is checked here too, and it is deliberately not part of the
index version. Chunk vectors do not depend on `questions.yaml` — the committed
*question* vectors do. So an edited question set invalidates one artefact and
not the other, they are reported as separate divergences, and only one of them
moves the version. Conflating them would mean either a version that changes
when nothing about the corpus did, or a stale question vector that no hash
catches.
"""

from __future__ import annotations

from dataclasses import dataclass

from .chunking import ChunkParams
from .corpus import corpus_fingerprint, load_documents
from .embedding import MODEL
from .evalset import questions_fingerprint
from .index import Index, compute_version


@dataclass(frozen=True)
class Divergence:
    """One input to the index that has changed since it was built."""

    input: str
    recorded: str
    current: str
    rebuilds_version: bool  # whether this input is part of the content address

    def to_dict(self) -> dict:
        return {
            "input": self.input,
            "recorded": self.recorded,
            "current": self.current,
            "rebuilds_version": self.rebuilds_version,
        }

    def line(self) -> str:
        return f"{self.input}: index has {self.recorded}, disk has {self.current}"


REBUILD_COMMAND = "rag-contract build-index"


@dataclass(frozen=True)
class IndexStatus:
    """Whether the committed index still describes what is on disk."""

    version: str
    expected_version: str
    divergences: tuple[Divergence, ...]

    @property
    def fresh(self) -> bool:
        return not self.divergences

    @property
    def message(self) -> str:
        if self.fresh:
            return f"index {self.version} describes the corpus on disk"
        changed = ", ".join(d.input for d in self.divergences)
        return (
            f"index {self.version} was built from inputs that have since "
            f"changed ({changed}). Run `{REBUILD_COMMAND}`; the rebuilt index "
            f"will be {self.expected_version}."
        )

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "expected_version": self.expected_version,
            "fresh": self.fresh,
            "divergences": [d.to_dict() for d in self.divergences],
        }


def index_status(index: Index, *, params: ChunkParams | None = None) -> IndexStatus:
    """Compare a loaded index against the corpus and question set on disk.

    Pure file reads and hashing: no network and no key. It is *not* cheap, and
    this docstring used to say it was. Measured over the HTTP surface, the
    `/health` route that calls it costs 7.6 ms — as much as the slowest
    question the service answers — because it re-reads and re-hashes the whole
    corpus on every call.

    Left as it is rather than cached, deliberately. A cached answer to "does
    the index still describe the corpus?" is an answer that can be stale in the
    one direction that matters, and guarantee 4 is worth more than the
    milliseconds. What the measurement changed is not this function but what
    covers it: `/health` is the most expensive route on the surface and the
    only one with no deadline, which is why the rate limit in `ratelimit.py`
    applies to it rather than exempting it the way health checks usually are.
    """
    params = params or ChunkParams()
    documents = load_documents()
    meta = index.meta

    divergences = []
    current_corpus = corpus_fingerprint(documents)
    if current_corpus != meta.corpus_fingerprint:
        divergences.append(
            Divergence(
                input="corpus",
                recorded=meta.corpus_fingerprint[:12],
                current=current_corpus[:12],
                rebuilds_version=True,
            )
        )
    if MODEL != meta.embedding_model:
        divergences.append(
            Divergence(
                input="embedding model",
                recorded=meta.embedding_model,
                current=MODEL,
                rebuilds_version=True,
            )
        )
    recorded_params = ChunkParams(
        chunk_words=meta.chunk_words, overlap_words=meta.overlap_words
    )
    if params != recorded_params:
        divergences.append(
            Divergence(
                input="chunk parameters",
                recorded=recorded_params.fingerprint(),
                current=params.fingerprint(),
                rebuilds_version=True,
            )
        )
    current_questions = questions_fingerprint()
    if current_questions != meta.questions_fingerprint:
        # Not part of the content address: the question set does not change
        # what was embedded from the corpus, only which question vectors the
        # eval is entitled to score against.
        divergences.append(
            Divergence(
                input="question set",
                recorded=meta.questions_fingerprint[:12],
                current=current_questions[:12],
                rebuilds_version=False,
            )
        )

    return IndexStatus(
        version=meta.version,
        expected_version=compute_version(documents, MODEL, params),
        divergences=tuple(divergences),
    )
