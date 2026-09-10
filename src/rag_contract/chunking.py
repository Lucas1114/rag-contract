"""Splitting sections into the units that get embedded and retrieved.

Chunks never span sections. That is not a quality decision but a contract one:
`eval/questions.yaml` annotates expected passages as section ids, and a chunk
straddling two sections could not say which one it came from. Section-scoped
chunking makes the provenance on every chunk exact.

Chunk size and overlap are fixed here and recorded in the index version. They
are tuning knobs the eval harness measures, not properties anything depends on;
changing them changes the index version and leaves every annotation valid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .sections import Section

# Fixed chunk parameters. Deliberately not tuned: the harness exists to measure
# the effect of changing them, and any change is visible in the index version.
CHUNK_WORDS = 220
OVERLAP_WORDS = 40


@dataclass(frozen=True)
class ChunkParams:
    chunk_words: int = CHUNK_WORDS
    overlap_words: int = OVERLAP_WORDS

    def __post_init__(self) -> None:
        if self.overlap_words >= self.chunk_words:
            raise ValueError("overlap_words must be smaller than chunk_words")

    def fingerprint(self) -> str:
        """The form these parameters take in the index version."""
        return f"chunk_words={self.chunk_words},overlap_words={self.overlap_words}"


DEFAULT_PARAMS = ChunkParams()


@dataclass(frozen=True)
class Chunk:
    """One embedded unit, and the section it is answerable for."""

    id: str  # "rfc9110#9.2.1/0"
    section_id: str  # "rfc9110#9.2.1"
    rfc: str
    section_number: str
    section_title: str
    ordinal: int  # position within the section, 0-based
    text: str  # the section text this chunk covers
    citation: str  # human-readable provenance

    @property
    def embedding_text(self) -> str:
        """What is actually sent to the embedding model.

        The citation is prepended so that the section a chunk belongs to is
        part of what is embedded. A chunk taken from the middle of a section
        otherwise carries no trace of its own subject.
        """
        return f"{self.citation}\n\n{self.text}"


def _words(text: str) -> list[str]:
    return text.split()


def _rewrap(words: list[str]) -> str:
    """Reassemble chunk text as single-spaced prose.

    RFC text is hard-wrapped at 72 columns; the line breaks carry no meaning
    and only add tokens.
    """
    return re.sub(r"\s+", " ", " ".join(words)).strip()


def chunk_section(
    section: Section, params: ChunkParams = DEFAULT_PARAMS
) -> list[Chunk]:
    """Split one section into overlapping chunks, in order.

    Sections with no prose of their own — parent headings whose content lives
    entirely in subsections — produce no chunks.
    """
    words = _words(section.text)
    if not words:
        return []

    step = params.chunk_words - params.overlap_words
    chunks = []
    start = 0
    ordinal = 0
    while start < len(words):
        window = words[start : start + params.chunk_words]
        chunks.append(
            Chunk(
                id=f"{section.id}/{ordinal}",
                section_id=section.id,
                rfc=section.rfc,
                section_number=section.number,
                section_title=section.title,
                ordinal=ordinal,
                text=_rewrap(window),
                citation=section.citation,
            )
        )
        if start + params.chunk_words >= len(words):
            break
        start += step
        ordinal += 1
    return chunks


def chunk_sections(
    sections: list[Section], params: ChunkParams = DEFAULT_PARAMS
) -> list[Chunk]:
    """Every chunk of every section, in document order."""
    return [c for s in sections for c in chunk_section(s, params)]
