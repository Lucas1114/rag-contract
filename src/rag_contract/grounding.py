"""The grounding check: does a passage actually say what a claim says it says?

Guarantee 3 turns on this. An answer is a list of claims, each naming the
section it rests on, and this module decides — per claim, deterministically,
with no model and no network — whether the named section supports it.

Why this is not a similarity threshold
--------------------------------------

The obvious design refuses when retrieval scores low. P02 measured that and it
does not work: the top-score distributions of answerable and unanswerable
questions overlap. q20 is answered squarely by RFC 8259 Section 2 at rank 1 and
scores 0.592; u02 asks about HTTP/2 frame layout, which the corpus does not
contain at all, and scores 0.604. Any cutoff that refuses u02 also refuses q20.
Confidence in *retrieval* is not evidence about the *answer*.

So the check moved downstream. It does not ask how close the question was to a
passage; it asks whether the sentence the service is about to emit is present
in the passage it cites. That question has an answer on the text itself.

What the check is, honestly
---------------------------

Three rules, applied in order, all lexical:

1. **The citation must be in the retrieved set.** The drafter saw exactly the
   retrieved passages. A claim citing anything else was written from the
   model's own memory, whatever it says, and is ungrounded by construction.

2. **Every literal in the claim must occur in the cited passage.** Literals are
   the tokens that carry the fact and get invented when a model is guessing:
   numbers (`405`, `4096`), and protocol tokens spelled with internal capitals
   or hyphens (`Set-Cookie`, `HttpOnly`, `ALPHA`). A claim asserting a status
   code the passage never mentions fails here regardless of how well the rest
   of it reads. Section references the claim makes about itself are stripped
   first — they point at the evidence rather than being part of it.

3. **Content-word coverage must clear a floor.** The claim's content words,
   minus stopwords, must largely appear in the passage.

This is lexical overlap, not entailment, and the limitation is deliberate
rather than hidden. A model can defeat rule 3 by quoting the passage and
negating it. What the check buys is that it is *deterministic*: it runs in CI
over committed fixtures with no key, no cost and no second model to trust, and
it catches the failure mode that actually occurs — a fluent sentence asserting
a specific fact the cited passage does not contain. A model-graded check would
be stronger on paper and would move the guarantee onto an unverifiable
dependency, which is the opposite of what this project is for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# The one tunable in the check. A claim rephrases its passage — it does not
# quote it — so full coverage is the wrong bar and would refuse correct
# answers. Two thirds leaves room for connective rewording while still
# requiring most of the claim's substance to be present in the text it cites.
#
# This is a service parameter, not a gate threshold, and deliberately lives in
# code rather than in eval/thresholds.yaml: that file holds outcomes the build
# is held to, and mixing the service's own tuning into it would break the
# property that the gate holds no numbers of its own.
COVERAGE_FLOOR = 0.67

# Small and closed on purpose. A long stopword list starts making judgements
# about which words carry meaning, which is the job this check is trying not
# to do.
# fmt: off
STOPWORDS = frozenset((
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "cannot", "did", "do", "does", "for", "from", "had", "has", "have",
    "how", "in", "into", "is", "it", "its", "may", "must", "not", "of", "on",
    "or", "should", "such", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "those", "to", "was", "were", "what",
    "when", "which", "while", "who", "whose", "will", "with", "would", "you",
    "your",
))
# fmt: on

# Hyphens, slashes and dots are part of a token only between alphanumerics,
# so "HTTP/1.1" and "Set-Cookie" survive intact while the full stop ending a
# sentence does not become part of "TRACE".
_TOKEN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9\-_/.]*[A-Za-z0-9])?")

# A claim often names its own section inside its prose — "In rfc9112#6.3, a
# recipient determines ..." — because the drafter was asked to cite and says so
# twice. Those tokens are a pointer, not an assertion: `rfc9112` and `6.3` are
# facts about where the claim came from, and demanding they appear in the RFC's
# own prose rejects correct claims for being explicit about their source. They
# are removed before either rule looks at the text.
#
# This is fixed here rather than by forbidding it in the prompt, because a
# check that depends on the model's cooperation is not a check.
_SECTION_REFERENCE = re.compile(
    r"\brfc\s?\d+\s?#\s?[A-Za-z0-9.]+"
    r"|\bRFC\s+\d+,?\s+(?:Section|section|§)\s*[A-Za-z0-9.]+"
    r"|\b(?:Section|section|§)\s*\d+(?:\.\d+)*\b",
    re.IGNORECASE,
)


def strip_citations(text: str) -> str:
    """Remove section references, which point at evidence rather than being it."""
    return _SECTION_REFERENCE.sub(" ", text)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text)


def literals(text: str) -> tuple[str, ...]:
    """The tokens in `text` that carry a fact rather than connect one.

    Numbers, and words spelled with an internal capital or a hyphen. These are
    what a guessing model invents, and what a reader checks first.
    """
    found = []
    for token in _tokens(text):
        if token.lower() in STOPWORDS:
            continue
        has_digit = any(c.isdigit() for c in token)
        has_inner_capital = any(c.isupper() for c in token[1:])
        is_all_capitals = token.isupper() and len(token) > 1
        joined = "-" in token or "_" in token
        looks_factual = has_digit or has_inner_capital or is_all_capitals or joined
        if looks_factual and token not in found:
            found.append(token)
    return tuple(found)


def _fold(token: str) -> str:
    """Lowercase, and a single crude plural fold.

    Deliberately not a stemmer. A stemmer is a dependency with opinions; the
    only morphology this check needs is that `characters` matches `character`.
    """
    token = token.lower()
    if len(token) > 3 and token.endswith("es") and not token.endswith("ses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def content_words(text: str) -> list[str]:
    """Folded, deduplicated, stopwords and one- and two-character tokens gone."""
    words: list[str] = []
    for token in _tokens(text):
        if token.lower() in STOPWORDS:
            continue
        folded = _fold(token)
        if len(folded) < 3 and not folded.isdigit():
            continue
        if folded not in words:
            words.append(folded)
    return words


def coverage(claim_text: str, passage_text: str) -> float:
    """Fraction of the claim's content words present in the passage.

    1.0 for a claim with no content words at all — such a claim asserts
    nothing, and rules 1 and 2 have already had their say about it.
    """
    claim = content_words(claim_text)
    if not claim:
        return 1.0
    passage = set(content_words(passage_text))
    return sum(word in passage for word in claim) / len(claim)


@dataclass(frozen=True)
class Claim:
    """One assertion the service is about to make, and what it rests on.

    The answer step returns these rather than prose. A claim that cannot name
    its passage cannot be checked, so the structure is what makes the guarantee
    enforceable at all.
    """

    text: str
    citation: str  # section id, e.g. "rfc9110#9.2.1"


@dataclass(frozen=True)
class Verdict:
    """What the check decided about one claim, and on which rule."""

    claim: Claim
    grounded: bool
    rule: str  # "" when grounded; otherwise the rule that rejected it
    detail: str
    coverage: float
    missing_literals: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        return self.claim.text

    @property
    def citation(self) -> str:
        return self.claim.citation


def check_claim(
    claim: Claim,
    passages: dict[str, str],
    coverage_floor: float = COVERAGE_FLOOR,
) -> Verdict:
    """Apply the three rules to one claim.

    `passages` maps section id to the text of that section *as retrieved* —
    the drafter saw these and nothing else, so this is also the set a citation
    has to fall inside.
    """
    passage = passages.get(claim.citation)
    if passage is None:
        return Verdict(
            claim=claim,
            grounded=False,
            rule="citation not retrieved",
            detail=(
                f"cites {claim.citation}, which is not among the "
                f"{len(passages)} passages retrieved for this question"
            ),
            coverage=0.0,
        )

    # Every token of the passage, not just the ones the passage would itself
    # classify as literal: the question is whether the literal *occurs* here,
    # and "Allow" mid-sentence must satisfy a claim that writes "ALLOW".
    passage_tokens = {token.lower() for token in _tokens(passage)}
    asserted = strip_citations(claim.text)
    missing = tuple(
        literal
        for literal in literals(asserted)
        if literal.lower() not in passage_tokens
    )
    claim_coverage = coverage(asserted, passage)
    if missing:
        return Verdict(
            claim=claim,
            grounded=False,
            rule="literal not in passage",
            detail=(f"{claim.citation} does not contain {', '.join(missing)}"),
            coverage=claim_coverage,
            missing_literals=missing,
        )

    if claim_coverage < coverage_floor:
        return Verdict(
            claim=claim,
            grounded=False,
            rule="coverage below floor",
            detail=(
                f"{claim_coverage:.2f} of the claim's content words appear in "
                f"{claim.citation}, below the {coverage_floor:.2f} floor"
            ),
            coverage=claim_coverage,
        )

    return Verdict(
        claim=claim, grounded=True, rule="", detail="", coverage=claim_coverage
    )


def check_claims(
    claims: list[Claim],
    passages: dict[str, str],
    coverage_floor: float = COVERAGE_FLOOR,
) -> list[Verdict]:
    return [check_claim(c, passages, coverage_floor) for c in claims]
