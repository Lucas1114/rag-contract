"""Loading and normalising the committed RFC corpus.

The corpus is fixed and committed verbatim, so every file is checked against
the sha256 recorded in `corpus/manifest.yaml` before it is used. Those hashes
are inputs to the index version; a silently edited corpus file would otherwise
produce an index whose version no longer describes its contents.

Two RFC text formats are present. RFCs 9110, 9111 and 9112 are unpaginated.
RFCs 3986, 6265 and 8259 predate that change and carry form feeds, a running
header on every page and a `[Page N]` footer. `depaginate` removes those
artefacts so that a section's text is continuous prose in both formats.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

CORPUS_DIR = Path(__file__).resolve().parents[2] / "corpus"
MANIFEST_PATH = CORPUS_DIR / "manifest.yaml"

# Page footer, e.g. "Berners-Lee, et al.    Standards Track    [Page 6]".
_PAGE_FOOTER = re.compile(r"^\S.*\[Page \d+\]\s*$")
# Running page header, e.g. "RFC 3986   URI Generic Syntax   January 2005".
_PAGE_HEADER = re.compile(r"^RFC \d+\s{2,}.*\s{2,}\S.*$")


class CorpusError(RuntimeError):
    """The corpus on disk does not match the committed manifest."""


@dataclass(frozen=True)
class Document:
    """One RFC, as committed."""

    rfc: str  # "rfc9110"
    number: int  # 9110
    title: str
    path: Path
    sha256: str
    text: str  # depaginated


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def depaginate(text: str) -> str:
    """Strip BOM, form feeds, running headers and `[Page N]` footers."""
    text = text.lstrip("﻿")
    if "\f" not in text:
        return text

    pages = []
    for page in text.split("\f"):
        lines = page.split("\n")
        # The footer is the last non-blank line of the page it closes.
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and _PAGE_FOOTER.match(lines[-1]):
            lines.pop()
        # The header is the first non-blank line of the page it opens.
        start = 0
        while start < len(lines) and not lines[start].strip():
            start += 1
        if start < len(lines) and _PAGE_HEADER.match(lines[start]):
            start += 1
        lines = lines[start:]
        while lines and not lines[-1].strip():
            lines.pop()
        pages.append("\n".join(lines))

    # A page break falls mid-section, so rejoin with a single blank line: the
    # paragraph boundary the break stood for is preserved, the pagination is not.
    return "\n\n".join(p for p in pages if p.strip())


def load_manifest() -> dict:
    return yaml.safe_load(MANIFEST_PATH.read_text())


def load_documents() -> list[Document]:
    """Load every corpus document, verifying it against the manifest.

    Raises CorpusError if a file is missing or its bytes do not hash to the
    value recorded in the manifest.
    """
    manifest = load_manifest()
    documents = []
    for entry in manifest["documents"]:
        path = CORPUS_DIR / entry["file"]
        if not path.is_file():
            raise CorpusError(f"{entry['file']} is listed in the manifest but missing")
        raw = path.read_bytes()
        digest = _sha256(raw)
        if digest != entry["sha256"]:
            raise CorpusError(
                f"{entry['file']} does not match the manifest: "
                f"expected {entry['sha256']}, found {digest}"
            )
        documents.append(
            Document(
                rfc=f"rfc{entry['rfc']}",
                number=int(entry["rfc"]),
                title=entry["title"],
                path=path,
                sha256=digest,
                text=depaginate(raw.decode("utf-8")),
            )
        )
    return documents


def corpus_fingerprint(documents: list[Document]) -> str:
    """A single hash over the corpus content, in manifest order.

    One of the three inputs to the index version.
    """
    joined = "".join(f"{d.rfc}:{d.sha256}\n" for d in documents)
    return _sha256(joined.encode("utf-8"))
