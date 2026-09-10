"""The corpus is a committed build input; these tests hold it to its manifest."""

from rag_contract.corpus import corpus_fingerprint, depaginate, load_documents


def test_every_document_matches_its_manifest_hash():
    # load_documents raises CorpusError on any mismatch.
    documents = load_documents()
    assert [d.rfc for d in documents] == [
        "rfc3986",
        "rfc6265",
        "rfc8259",
        "rfc9110",
        "rfc9111",
        "rfc9112",
    ]


def test_depagination_removes_every_page_artefact():
    for document in load_documents():
        assert "\f" not in document.text
        assert "[Page " not in document.text
        assert "﻿" not in document.text


def test_depagination_is_a_no_op_on_unpaginated_text():
    text = "1.  Introduction\n\n   Body text.\n"
    assert depaginate(text) == text


def test_depagination_drops_header_and_footer_but_keeps_body():
    paginated = (
        "   First page body.\n\n\n"
        "Berners-Lee, et al.         Standards Track                     [Page 1]\n"
        "\f\n"
        "RFC 3986                   URI Generic Syntax               January 2005\n"
        "\n\n   Second page body.\n"
    )
    result = depaginate(paginated)
    assert "[Page 1]" not in result
    assert "URI Generic Syntax" not in result
    assert "First page body." in result
    assert "Second page body." in result


def test_fingerprint_is_stable_and_content_addressed():
    documents = load_documents()
    assert corpus_fingerprint(documents) == corpus_fingerprint(documents)
    assert len(corpus_fingerprint(documents)) == 64
