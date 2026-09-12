"""The latency half of guarantee 5: what a request is allowed to spend.

What a ceiling here is actually for
-----------------------------------

The honest answer took a measurement, because the obvious answer is wrong.

Retrieval is brute-force cosine over 868 chunks of 1536 float32 — one 5 MB
matrix-vector product, 0.026 ms, and flat across all 28 questions. The answer
step replays a committed fixture and is a dictionary lookup. So the served
request has no slow path, and a latency ceiling that claimed to protect
retrieval would be guarding nothing.

Measuring the request by stage says where the time actually goes:

    retrieval            0.026 ms, identical for every question
    passages_from_hits   0.004 ms
    grounding check      0.001 ms to 7.4 ms — a spread of three orders

The grounding check dominates the request by two orders of magnitude, and it is
the one part of the pipeline this project added. Its cost is O(claims x cited
passage size): `check_claim` tokenises the whole cited passage once per claim,
twice over. The corpus bounds the passage side — ten retrieved sections, 13k
characters at worst. Nothing bounds the other side. **The number of claims comes
from the model**, not from the corpus, the question set or the request, and it
is the only input on the request path that the service does not control.

That is what the ceiling is for, and it is not an invented slow path: it is the
one place where an input the service does not own multiplies work the service
does. A drafter returning two hundred claims instead of fifteen produces a
request an order of magnitude slower than any measured here, and today nothing
would stop it.

Why exceeding it abandons the request
-------------------------------------

The tempting behaviour is to stop checking and answer with the claims checked so
far. That is exactly wrong, and the reason is guarantee 3 rather than guarantee
5. A claim that has not been checked is not a claim that passed; an answer
assembled from a truncated grounding check would report `grounded` or `partial`
on the strength of claims nobody verified, which is the precise failure the
whole grounding check exists to prevent. A partially applied check is not a
weaker check, it is an unsound one.

So the deadline is checked *inside* the per-claim loop and raises out of it.
There is no half-checked verdict list for a later stage to be tempted by, and
the response says the request was abandoned rather than answered. Like
`no_context`, this is a service error and not a decision about the corpus, which
is why it is a 503 and why it is not a member of `AnswerState`: `decide()` never
returns it, because reaching it means `decide()` never finished.

Two numbers, not one
--------------------

`request_deadline_ms` is what the service enforces per request. `max_request_ms`
is the bar the gate holds the measured slowest question to, and it sits well
below the deadline on purpose — a build passing while sitting at the edge of the
behaviour ceiling would be a green build reporting that the service is about to
start failing requests. `BudgetLimits` refuses a file where the gate's bar is
not at most half the deadline, on the same principle that `thresholds.yaml` is
refused when it gates refusal from only one side.

The two also catch different things. The gate sees the index every build is
made against and fails *before* a deployment where the ceiling would fire. It
cannot see an index installed at runtime by `registry.install`, which P05 built
and which can be handed anything; the per-request deadline is the backstop for
exactly that, and is the only one of the two that a served request ever reaches.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

Clock = Callable[[], float]


class BudgetError(RuntimeError):
    """The committed budget is missing, malformed, or not self-consistent."""


class DeadlineExceeded(RuntimeError):
    """A request spent its latency budget and was abandoned.

    Carries the spend rather than just a message, because the response has to
    report where the time went — a deadline failure that does not say which
    stage ate the budget tells an operator nothing they can act on.
    """

    def __init__(self, spend: Spend, stage: str):
        self.spend = spend
        self.stage = stage
        super().__init__(
            f"abandoned in {stage} after {spend.elapsed_ms:.1f} ms against a "
            f"{spend.deadline_ms:.1f} ms deadline"
        )

    @property
    def detail(self) -> str:
        """What the response says, in the terms guarantee 3 cares about."""
        return (
            f"{self}. The request was abandoned rather than answered: a "
            "grounding check that stopped part way through has not checked the "
            "remaining claims, and an answer built from it would report a state "
            "no check supports."
        )


@dataclass(frozen=True)
class BudgetLimits:
    """The committed numbers behind guarantee 5, so far.

    Latency first, because it is the half that lands on the request path. The
    cost half joins it in this block rather than in a second one: they are one
    guarantee, asking one question — what a request, or a command, may spend
    before the behaviour changes.
    """

    request_deadline_ms: float
    max_request_ms: float

    @classmethod
    def from_mapping(cls, raw: Mapping | None) -> BudgetLimits:
        if not raw:
            raise BudgetError(
                "no budget block. Guarantee 5 is held by committed numbers or "
                "it is not held: a ceiling that lives in code is a ceiling "
                "nobody reviews when it moves."
            )
        fields = ("request_deadline_ms", "max_request_ms")
        missing = [name for name in fields if name not in raw]
        if missing:
            raise BudgetError(f"the budget block leaves {', '.join(missing)} unset")
        unknown = sorted(set(raw) - set(fields))
        if unknown:
            raise BudgetError(
                f"the budget block sets {', '.join(unknown)}, which nothing reads"
            )

        limits = cls(**{name: float(raw[name]) for name in fields})
        for name in fields:
            if getattr(limits, name) <= 0:
                raise BudgetError(f"{name} must be positive")

        # The gate's bar has to leave room under the behaviour ceiling. A bar
        # within a factor of two of the deadline is not a gate on the deadline,
        # it is the deadline with extra steps: the build would still be green
        # on the last commit before requests start being abandoned.
        if limits.max_request_ms > limits.request_deadline_ms / 2:
            raise BudgetError(
                f"max_request_ms {limits.max_request_ms} leaves no headroom "
                f"under request_deadline_ms {limits.request_deadline_ms}. The "
                "gate's bar must be at most half the deadline, or a green build "
                "says nothing about whether requests are about to be abandoned."
            )
        return limits

    def budget(self, clock: Clock = time.perf_counter) -> Budget:
        return Budget(deadline_ms=self.request_deadline_ms, clock=clock)


@dataclass(frozen=True)
class Budget:
    """The deadline a request is served under.

    The clock is injected so the deadline can be tested by controlling time
    rather than by burning it. A test that proved this by sleeping would be
    proving that `time.sleep` works, and would have to put a slow path into the
    production code to have something to time — which is the thing this module's
    docstring refuses to do.
    """

    deadline_ms: float
    clock: Clock = time.perf_counter

    def start(self) -> Spend:
        return Spend(deadline_ms=self.deadline_ms, clock=self.clock)


@dataclass
class Spend:
    """One request's wall clock, stage by stage.

    Mutable, and deliberately not shared: it belongs to a single request from
    entry to completion, exactly like the index that request acquired.
    """

    deadline_ms: float
    clock: Clock = time.perf_counter
    stages: dict[str, float] = field(default_factory=dict)
    _started: float = 0.0
    _last: float = 0.0

    def __post_init__(self) -> None:
        self._started = self._last = self.clock()

    @property
    def elapsed_ms(self) -> float:
        return (self.clock() - self._started) * 1000.0

    @property
    def exceeded(self) -> bool:
        return self.elapsed_ms > self.deadline_ms

    def checkpoint(self, stage: str) -> None:
        """Close out a stage and refuse to continue past the deadline.

        Called at stage boundaries and, inside the grounding check, once per
        claim — because the claim loop is the unbounded one, and a deadline
        checked only at the end of it would report the overrun after paying for
        it in full.
        """
        now = self.clock()
        self.stages[stage] = self.stages.get(stage, 0.0) + (now - self._last) * 1000.0
        self._last = now
        if (now - self._started) * 1000.0 > self.deadline_ms:
            raise DeadlineExceeded(self, stage)

    def stage(self, name: str) -> Callable[[], None]:
        """A checkpoint bound to one stage, for handing to code downstream.

        `grounding.check_claims` takes one of these. It is given a callable
        rather than the `Spend` itself so that the grounding check keeps knowing
        nothing about budgets: all it does is call something once per claim.
        """
        return lambda: self.checkpoint(name)

    def to_dict(self) -> dict:
        return {
            "deadline_ms": round(self.deadline_ms, 3),
            "elapsed_ms": round(self.elapsed_ms, 3),
            "exceeded": self.exceeded,
            "stages": {name: round(ms, 3) for name, ms in self.stages.items()},
        }


def measure_request_ms(answer: Callable[[str], object], question_ids, repeats: int = 5):
    """The slowest question's median request time, and where it was spent.

    Median of `repeats` per question, then the worst question — not the worst
    single sample. One sample catches whatever else the machine was doing;
    guarantee 5's bar is about the shape of the request path, not about the
    scheduler.

    This is the one measurement in the gate that is not deterministic, and the
    committed bar it feeds is sized accordingly. See `eval/thresholds.yaml`.
    """
    medians: dict[str, float] = {}
    for question_id in question_ids:
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            answer(question_id)
            samples.append((time.perf_counter() - started) * 1000.0)
        medians[question_id] = statistics.median(samples)
    slowest = max(medians, key=lambda q: medians[q])
    return slowest, medians[slowest], medians
