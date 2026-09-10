"""Parsing an RFC into numbered sections.

Section ids are the annotation contract. `eval/questions.yaml` annotates
expected passages as section ids (`rfc9110#9.2.1`), never chunk ids, so that
the question set stays valid across changes to chunk size and overlap. Every
chunk the retriever returns must resolve back to the section it came from, and
that resolution starts here.

Headings in both RFC text formats are unindented and numbered; body text,
lists and ABNF are indented by at least one space. Anchoring the heading
pattern at column 0 is therefore enough on its own to keep the table of
contents out, since its entries are indented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .corpus import Document

# "9.2.1.  Safe Methods", "Appendix A.  Collected ABNF", "B.1.  MIME-Version".
_HEADING = re.compile(
    r"^(?:Appendix\s+)?"
    r"(?P<number>[0-9]+(?:\.[0-9]+)*|[A-Z](?:\.[0-9]+)*)"
    r"\.\s{1,3}(?P<title>\S.*)$"
)

# Unnumbered blocks that close the final section. Everything from here to the
# end of the document is back matter and carries no answerable content.
_BACK_MATTER = frozenset(
    {
        "Index",
        "Acknowledgements",
        "Acknowledgement",
        "Authors' Addresses",
        "Author's Address",
        "Contributors",
    }
)


@dataclass(frozen=True)
class Section:
    """One numbered section of one RFC."""

    id: str  # "rfc9110#9.2.1"
    rfc: str  # "rfc9110"
    number: str  # "9.2.1"
    title: str  # "Safe Methods"
    text: str  # body text, heading excluded, common indentation removed
    ordinal: int  # position within the document, 0-based

    @property
    def citation(self) -> str:
        """Human-readable provenance, carried on every chunk."""
        return f"{self.rfc.upper().replace('RFC', 'RFC ')} Section {self.number}: {self.title}"


def _dedent(lines: list[str]) -> str:
    """Remove the indentation RFC body text is uniformly wrapped in.

    Relative indentation is preserved, so ABNF and lists keep their shape.
    """
    widths = [len(ln) - len(ln.lstrip()) for ln in lines if ln.strip()]
    if not widths:
        return ""
    margin = min(widths)
    out = [ln[margin:] if ln.strip() else "" for ln in lines]
    # Collapse the blank-line runs left behind by page breaks.
    text = "\n".join(out).strip("\n")
    return re.sub(r"\n{3,}", "\n\n", text)


def parse_sections(document: Document) -> list[Section]:
    """Split one document into its numbered sections, in document order."""
    lines = document.text.split("\n")

    starts: list[tuple[int, str, str]] = []  # (line index, number, title)
    end = len(lines)
    for i, line in enumerate(lines):
        if not line or line[0].isspace():
            continue
        match = _HEADING.match(line)
        if match:
            starts.append((i, match.group("number"), match.group("title").strip()))
        elif starts and line.strip() in _BACK_MATTER:
            end = i
            break

    sections = []
    for ordinal, (start, number, title) in enumerate(starts):
        stop = starts[ordinal + 1][0] if ordinal + 1 < len(starts) else end
        sections.append(
            Section(
                id=f"{document.rfc}#{number}",
                rfc=document.rfc,
                number=number,
                title=title,
                text=_dedent(lines[start + 1 : stop]),
                ordinal=ordinal,
            )
        )
    return sections


def parse_corpus(documents: list[Document]) -> list[Section]:
    """Every section of every document, in manifest order."""
    return [s for d in documents for s in parse_sections(d)]


def section_index(sections: list[Section]) -> dict[str, Section]:
    """Section id -> Section, for resolving annotations and chunk provenance."""
    return {s.id: s for s in sections}
