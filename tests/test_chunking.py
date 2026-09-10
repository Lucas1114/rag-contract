"""Chunking is measured by the harness; what it must guarantee is provenance."""

import pytest

from rag_contract.chunking import ChunkParams, chunk_section, chunk_sections
from rag_contract.corpus import load_documents
from rag_contract.sections import Section, parse_corpus


def section(text: str, ident: str = "rfc9110#9.2.1") -> Section:
    return Section(
        id=ident,
        rfc="rfc9110",
        number="9.2.1",
        title="Safe Methods",
        text=text,
        ordinal=0,
    )


@pytest.fixture(scope="module")
def corpus_chunks():
    return chunk_sections(parse_corpus(load_documents()))


def test_every_chunk_resolves_to_its_section(corpus_chunks):
    section_ids = {s.id for s in parse_corpus(load_documents())}
    assert all(c.section_id in section_ids for c in corpus_chunks)
    assert all(c.id.startswith(c.section_id + "/") for c in corpus_chunks)


def test_chunk_ids_are_unique(corpus_chunks):
    assert len({c.id for c in corpus_chunks}) == len(corpus_chunks)


def test_no_chunk_exceeds_the_configured_size(corpus_chunks):
    params = ChunkParams()
    assert max(len(c.text.split()) for c in corpus_chunks) <= params.chunk_words


def test_sections_with_no_prose_produce_no_chunks():
    assert chunk_section(section("")) == []
    assert chunk_section(section("   \n\n  ")) == []


def test_a_short_section_becomes_exactly_one_chunk():
    chunks = chunk_section(section("Short body text."))
    assert len(chunks) == 1
    assert chunks[0].id == "rfc9110#9.2.1/0"
    assert chunks[0].ordinal == 0


def test_chunks_overlap_by_the_configured_amount():
    params = ChunkParams(chunk_words=10, overlap_words=4)
    words = [f"w{i}" for i in range(26)]
    chunks = chunk_section(section(" ".join(words)), params)

    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first == words[:10]
    assert second[: params.overlap_words] == first[-params.overlap_words :]


def test_chunks_cover_the_whole_section():
    params = ChunkParams(chunk_words=10, overlap_words=4)
    words = [f"w{i}" for i in range(26)]
    chunks = chunk_section(section(" ".join(words)), params)
    covered = set()
    for chunk in chunks:
        covered.update(chunk.text.split())
    assert covered == set(words)


def test_overlap_must_be_smaller_than_the_chunk():
    with pytest.raises(ValueError):
        ChunkParams(chunk_words=10, overlap_words=10)


def test_hard_wrapping_is_removed():
    chunks = chunk_section(section("A line\nbroken by\nRFC wrapping."))
    assert chunks[0].text == "A line broken by RFC wrapping."


def test_embedding_text_carries_the_citation():
    chunk = chunk_section(section("Body."))[0]
    assert chunk.embedding_text.startswith("RFC 9110 Section 9.2.1: Safe Methods")
    assert "Body." in chunk.embedding_text


def test_chunk_parameters_are_part_of_the_index_version():
    assert ChunkParams(chunk_words=10, overlap_words=4).fingerprint() != (
        ChunkParams().fingerprint()
    )
