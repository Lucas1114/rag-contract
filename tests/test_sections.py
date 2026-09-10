"""The section parser underwrites the annotation contract.

If a section id in `eval/questions.yaml` stops resolving, the question set no
longer means what it says, so that is asserted here rather than left to the
eval run.
"""

import pytest
import yaml

from rag_contract.corpus import load_documents
from rag_contract.evalset import QUESTIONS_PATH, load_questions
from rag_contract.sections import parse_corpus, section_index


@pytest.fixture(scope="module")
def sections():
    return parse_corpus(load_documents())


@pytest.fixture(scope="module")
def by_id(sections):
    return section_index(sections)


def test_section_ids_are_unique(sections, by_id):
    assert len(by_id) == len(sections)


def test_every_expected_annotation_resolves_to_a_section(by_id):
    missing = [
        section_id
        for question in load_questions()
        for section_id in question.expected
        if section_id not in by_id
    ]
    assert missing == []


def test_every_nearest_annotation_resolves_to_a_section(by_id):
    # `nearest` is metadata rather than a scored field, but a stale id there is
    # still a lie about the corpus.
    raw = yaml.safe_load(QUESTIONS_PATH.read_text())
    missing = [
        section_id
        for entry in raw["questions"]
        for section_id in entry.get("nearest", [])
        if section_id not in by_id
    ]
    assert missing == []


def test_annotated_sections_carry_the_text_they_are_annotated_for(by_id):
    assert "unreserved  = ALPHA / DIGIT" in by_id["rfc3986#2.3"].text
    # RFC text is hard-wrapped, so the sentence is matched across its line break.
    assert (
        "GET, HEAD,\nOPTIONS, and TRACE methods are defined to be safe"
        in by_id["rfc9110#9.2.1"].text
    )
    assert by_id["rfc9111#5.2.2.10"].title == "s-maxage"
    assert by_id["rfc9112#3.2.3"].title == "authority-form"


def test_headings_are_excluded_from_section_text(by_id):
    section = by_id["rfc9110#9.2.1"]
    assert not section.text.startswith("9.2.1.")
    assert section.title == "Safe Methods"


def test_appendices_are_parsed(by_id):
    assert by_id["rfc3986#A"].title == "Collected ABNF for URI"
    assert by_id["rfc9112#C.2.2"].title == "Keep-Alive Connections"


def test_table_of_contents_does_not_leak_into_sections(sections):
    # A TOC entry would show up as a duplicate heading or as dot leaders.
    for section in sections:
        assert ". . ." not in section.title


def test_back_matter_is_excluded(by_id):
    # The last numbered section of RFC 9111 must not swallow Index or Authors'.
    last = by_id["rfc9111#B"]
    assert "Authors' Addresses" not in last.text
    assert "Email: fielding@gbiv.com" not in last.text
