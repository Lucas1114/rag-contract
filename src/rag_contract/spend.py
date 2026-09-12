"""The cost half of guarantee 5: what this project is allowed to spend.

Where the money actually is
---------------------------

Nowhere on the request path. The served request calls nothing — committed
vectors, committed drafts, no key — which is the property guarantees 1 to 4 all
rest on. So a token ceiling on the request path would be a ceiling on zero.

The spend is in the two commands that are run by hand and commit what they
produce, and it is small and known:

    build-index     293,131 estimated tokens at $0.02/1M   $0.0059
    record-drafts   measured on 2026-09-12                 $0.5700

The second number is an invoice, not an estimate: one full `record-drafts` run,
29 requests, 75,250 input tokens. It is also the measurement that validates the
estimator below — `estimate_tokens` predicted 108,693 input tokens against
75,250 billed, over-estimating by a factor of 1.44. Every figure this module
produces is therefore an over-estimate, which is the direction a cap has to err
in: a cap built on an optimistic estimate authorises a call that breaks it.

Authorising before the call, recording after it
-----------------------------------------------

A cap that notices afterwards is a report. This one is asked *before* each
request whether that request fits inside what is left of the day, and refuses
the call when it does not — the command stops with nothing written rather than
continuing and logging a number someone might read later.

That requires an upper bound on a call's cost before making it, which sounds
impossible for a completion and is not. Embeddings are exact: the cost is a
function of the input text alone. Completions are bounded because the caller
sets the bound — `max_completion_tokens` is in the request, reasoning tokens
count against it, so input tokens plus that ceiling prices the worst the call
can do. The ledger then records what the API says it actually used, so the
running total is real spend rather than accumulated worst cases.

The cap lives in eval/thresholds.yaml
-------------------------------------

`.env.example` used to carry `DAILY_SPEND_CAP_USD=2.00` and nothing read it.
Wiring it up was the obvious fix and is the wrong one: a cap set by whoever
happens to run the command is a cap that is invisible in review and different on
every machine. It belongs with the other committed bars, so that raising it is
an edit to `eval/thresholds.yaml` that shows up in a diff, and so that the gate
and the runtime cannot disagree about what it is. There is deliberately no
environment override.

Prices are inputs, not bars, so they are here rather than in that file. A stale
price makes the cap optimistic in one direction only — the vendor's list price
falling is safe, rising is not — which is the other reason the token estimator
is left over-estimating rather than tightened.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

LEDGER_PATH = Path(__file__).resolve().parents[2] / ".spend" / "ledger.jsonl"

# List prices per 1M tokens, read from developers.openai.com/api/docs/pricing.
# Recorded by hand, with the date, because they are a fact about a vendor rather
# than a decision this project made — and a number quoted with no date is a
# number nobody can check.
PRICES_AS_OF = "2026-09-13"


class SpendError(RuntimeError):
    """A price is unknown, or a ledger entry could not be read."""


class SpendCapExceeded(RuntimeError):
    """A call was refused because it would break the day's cap.

    Raised before the call, not after it. The command stops here with nothing
    written, which is the behaviour change guarantee 5 promises — the
    alternative is a log line after the money is gone.
    """


@dataclass(frozen=True)
class Price:
    """What a model costs per 1M tokens, in USD."""

    input_per_m: float
    output_per_m: float = 0.0

    def usd(self, input_tokens: int, output_tokens: int = 0) -> float:
        return (
            input_tokens * self.input_per_m + output_tokens * self.output_per_m
        ) / 1_000_000


PRICES: dict[str, Price] = {
    # Embeddings have no output side at all, which is what makes build-index
    # priceable to the cent from committed text before it runs.
    "text-embedding-3-small": Price(input_per_m=0.02),
    "gpt-5.5": Price(input_per_m=5.00, output_per_m=30.00),
}


def price_for(model: str) -> Price:
    """The price of a model, resolving a dated snapshot to its family.

    `gpt-5.5-2026-04-23` prices as `gpt-5.5`: the snapshot is pinned so the
    fixtures can be re-recorded reproducibly, not because it is billed
    differently.

    An unknown model raises rather than defaulting. A cap that prices what it
    does not recognise at zero is not a cap, and silently authorising an
    unpriced model is exactly how a spend cap fails in the only way that
    matters.
    """
    if model in PRICES:
        return PRICES[model]
    family = model.rsplit("-", 3)[0]
    if family in PRICES:
        return PRICES[family]
    raise SpendError(
        f"no committed price for {model!r}. Add it to spend.PRICES with the "
        f"date it was read; the cap cannot authorise a call it cannot price."
    )


def estimate_tokens(text: str) -> int:
    """A deliberate over-estimate of a text's token count.

    Three characters per token against English prose that actually tokenises at
    around four and a third — checked against a real invoice, which billed
    75,250 input tokens for a run this function scores at 108,693.

    Over-estimating is the whole point in both places this is used. For
    scheduling it costs a little wall clock on a build that runs once;
    under-estimating costs a 429 part way through. For pricing it makes every
    ceiling conservative; under-estimating authorises a call that breaks the
    cap. A tokeniser would be more accurate, a dependency, and wrong in the
    unsafe direction whenever the vendor changed one.
    """
    return max(1, len(text) // 3)


def embedding_usd(texts: list[str], model: str) -> float:
    """Exact-shaped cost of embedding these texts: no output side to guess."""
    tokens = sum(estimate_tokens(text) for text in texts)
    return price_for(model).usd(tokens)


def completion_bound_usd(prompt: str, max_completion_tokens: int, model: str) -> float:
    """The most one completion can cost, before it is made.

    The input side is known. The output side is bounded because the request
    sets `max_completion_tokens` and reasoning tokens count against it, so this
    is a real ceiling rather than a guess — which is what lets the cap refuse a
    call in advance instead of noticing afterwards.
    """
    price = price_for(model)
    return price.usd(estimate_tokens(prompt), max_completion_tokens)


@dataclass(frozen=True)
class Entry:
    """One API call that was actually made, and what it actually cost."""

    at: str
    command: str
    model: str
    input_tokens: int
    output_tokens: int
    usd: float

    def to_dict(self) -> dict:
        return {
            "at": self.at,
            "command": self.command,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "usd": round(self.usd, 6),
        }


class Ledger:
    """Append-only record of what has been spent, by day.

    A local file and not a committed one. It is machine state — what *this*
    checkout has spent — and committing it would mean a file that changes on
    every hand-run command, in a repository whose other artefacts change only
    when the corpus does. `.gitignore` carries it.
    """

    def __init__(self, path: Path = LEDGER_PATH):
        self.path = path

    def entries(self) -> list[Entry]:
        if not self.path.is_file():
            return []
        entries = []
        for number, line in enumerate(self.path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                entries.append(Entry(**raw))
            except (TypeError, ValueError) as exc:
                raise SpendError(
                    f"{self.path}:{number} is not a ledger entry: {exc}. The cap "
                    "refuses to run against a ledger it cannot total."
                ) from exc
        return entries

    def append(self, entry: Entry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(entry.to_dict()) + "\n")

    def spent_on(self, day: str) -> float:
        return sum(e.usd for e in self.entries() if e.at.startswith(day))


def today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


@dataclass
class Cap:
    """The day's spend cap, asked before every call and told after every call.

    Holds no number of its own: `daily_usd` comes from `eval/thresholds.yaml`
    by way of the command that built it, the same way every other bar in this
    project reaches the code that enforces it.
    """

    daily_usd: float
    ledger: Ledger
    command: str

    def spent_today(self) -> float:
        return self.ledger.spent_on(today())

    def remaining(self) -> float:
        return self.daily_usd - self.spent_today()

    def authorise(self, usd: float, what: str) -> None:
        """Permit one call, or refuse it. Called before the request is sent."""
        remaining = self.remaining()
        if usd > remaining:
            raise SpendCapExceeded(
                f"{what} could cost up to ${usd:.4f} and only ${remaining:.4f} "
                f"of the ${self.daily_usd:.2f} daily cap is left "
                f"(${self.spent_today():.4f} spent today). Stopping before the "
                "call rather than after it. The cap is budget.daily_cap_usd in "
                "eval/thresholds.yaml."
            )

    def record(self, model: str, input_tokens: int, output_tokens: int = 0) -> Entry:
        """Bill one completed call at what it actually used."""
        entry = Entry(
            at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            command=self.command,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usd=price_for(model).usd(input_tokens, output_tokens),
        )
        self.ledger.append(entry)
        return entry
