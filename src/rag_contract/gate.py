"""The regression gate: guarantee 2.

Guarantee 1 measures retrieval quality. This is what makes that measurement
load-bearing — a committed threshold file, a build that fails below it, and a
diff whenever the bar moves.

The gate reads `eval/thresholds.yaml` and a report produced by `evaluate`. It
computes nothing itself and holds no numbers of its own: every floor is in the
committed file, so lowering the bar is an edit someone has to make and a
reviewer can see.

Four kinds of check, and they are independent:

    aggregate     floors on recall@k and MRR over the answerable questions
    per-question  a rank ceiling for each answerable question
    refusal       guarantee 3, from the answer eval — floors on how often the
                  corpus's own questions are answered, a ceiling on how often
                  questions it cannot answer are
    freshness     guarantee 4 — the committed index still describes the
                  committed corpus. The only check here that holds no number

The first three exist because each alone is blind. Aggregates miss compensating
movement — one question improving while another collapses leaves recall@1 flat
and MRR higher. Per-question ceilings miss uniform drift that stays inside every
ceiling while every question gets worse. And both are silent about what the
service does once retrieval has handed it the right passage, which is what the
refusal checks measure. A build passes only when all of them agree.

One more check keeps the file honest: the set of gated questions must be exactly
the set of answerable questions in the report. A question added to the question
set without a recorded ceiling fails the gate rather than slipping in ungated.

Freshness is the odd one out and is not in `eval/thresholds.yaml`, because
there is no bar to set. Either the committed vectors were built from the corpus
in this commit or they were not, and a project that could choose to tolerate
"not" would be quoting eval numbers about a corpus it no longer has. It is the
only part of guarantee 4 a machine can enforce: CI has no key, so it cannot
rebuild an index — it can only refuse to pass a commit that needed one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .lifecycle import IndexStatus

THRESHOLDS_PATH = Path(__file__).resolve().parents[2] / "eval" / "thresholds.yaml"

AGGREGATE_METRICS = ("recall_at_1", "recall_at_5", "recall_at_10", "mrr")

# Guarantee 3. `min_` names a floor and `max_` a ceiling, and both kinds are
# needed: a ceiling on questions the corpus cannot answer is scored perfectly
# by a service that refuses everything, and a floor on the ones it can answer
# is scored perfectly by a service that answers everything.
REFUSAL_METRICS = {
    "min_grounded_rate": "grounded_rate",
    "min_state_agreement": "state_agreement",
    "max_answered_unanswerable": "answered_unanswerable",
}


class ThresholdError(RuntimeError):
    """The threshold file is missing, malformed, or not about this report."""


@dataclass(frozen=True)
class Thresholds:
    aggregate: dict[str, float]
    max_rank: dict[str, int]
    measured: dict
    refusal: dict[str, float]


@dataclass(frozen=True)
class Check:
    name: str
    observed: float | int | str | None
    limit: float | int | str
    passed: bool
    detail: str = ""

    def line(self) -> str:
        mark = "ok  " if self.passed else "FAIL"
        observed = "none" if self.observed is None else f"{self.observed}"
        suffix = f"  {self.detail}" if self.detail else ""
        return f"{mark}  {self.name:<24} {observed:>8}  limit {self.limit}{suffix}"


def load_thresholds(path: Path = THRESHOLDS_PATH) -> Thresholds:
    if not path.exists():
        raise ThresholdError(
            f"{path} does not exist. The regression gate has no bar to hold."
        )
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ThresholdError(f"{path} is not a mapping")

    aggregate = raw.get("aggregate") or {}
    if not aggregate:
        raise ThresholdError(f"{path} sets no aggregate floors")
    unknown = set(aggregate) - set(AGGREGATE_METRICS)
    if unknown:
        raise ThresholdError(
            f"{path} sets floors on metrics the eval does not report: "
            f"{', '.join(sorted(unknown))}"
        )

    max_rank = raw.get("max_rank") or {}
    if not max_rank:
        raise ThresholdError(f"{path} sets no per-question rank ceilings")
    for qid, ceiling in max_rank.items():
        if not isinstance(ceiling, int) or ceiling < 1:
            raise ThresholdError(f"{path}: {qid} has a non-positive rank ceiling")

    refusal = raw.get("refusal") or {}
    if not refusal:
        raise ThresholdError(f"{path} sets no refusal thresholds")
    unknown = set(refusal) - set(REFUSAL_METRICS)
    if unknown:
        raise ThresholdError(
            f"{path} sets refusal thresholds the answer eval does not report: "
            f"{', '.join(sorted(unknown))}"
        )
    missing = set(REFUSAL_METRICS) - set(refusal)
    if missing:
        raise ThresholdError(
            f"{path} leaves {', '.join(sorted(missing))} unset. Guarantee 3 is "
            "held from both sides or not at all: a ceiling alone is passed by a "
            "service that refuses everything."
        )

    return Thresholds(
        aggregate={k: float(v) for k, v in aggregate.items()},
        max_rank={str(k): int(v) for k, v in max_rank.items()},
        measured=raw.get("measured") or {},
        refusal={k: float(v) for k, v in refusal.items()},
    )


def _aggregate_checks(report: dict, thresholds: Thresholds) -> list[Check]:
    metrics = report["metrics"]
    checks = []
    for name, floor in thresholds.aggregate.items():
        if name not in metrics:
            raise ThresholdError(
                f"the report does not carry {name}; the gate cannot hold a "
                "floor on a metric the eval did not produce"
            )
        observed = metrics[name]
        checks.append(
            Check(
                name=name,
                observed=observed,
                limit=floor,
                passed=observed >= floor,
                detail="" if observed >= floor else "below floor",
            )
        )
    return checks


def _coverage_checks(report: dict, thresholds: Thresholds) -> list[Check]:
    """The threshold file must name exactly the answerable questions."""
    answerable = {q["id"] for q in report["questions"] if q["answerable"]}
    gated = set(thresholds.max_rank)

    checks = []
    missing = sorted(answerable - gated)
    if missing:
        checks.append(
            Check(
                name="ungated questions",
                observed=len(missing),
                limit=0,
                passed=False,
                detail=(
                    f"{', '.join(missing)} answerable with no rank ceiling in "
                    "eval/thresholds.yaml"
                ),
            )
        )
    stale = sorted(gated - answerable)
    if stale:
        checks.append(
            Check(
                name="stale ceilings",
                observed=len(stale),
                limit=0,
                passed=False,
                detail=(
                    f"{', '.join(stale)} carry a rank ceiling but are not "
                    "answerable questions in the report"
                ),
            )
        )
    return checks


def _rank_checks(report: dict, thresholds: Thresholds) -> list[Check]:
    checks = []
    for result in report["questions"]:
        if not result["answerable"]:
            continue
        ceiling = thresholds.max_rank.get(result["id"])
        if ceiling is None:
            continue  # reported by the coverage check
        rank = result["rank"]
        passed = rank is not None and rank <= ceiling
        detail = ""
        if rank is None:
            detail = f"no expected section in the top {report['top_k']}"
        elif not passed:
            detail = f"first hit at {rank}; rank 1 was {result['top_section']}"
        checks.append(
            Check(
                name=f"rank {result['id']}",
                observed=rank,
                limit=ceiling,
                passed=passed,
                detail=detail,
            )
        )
    return checks


def refusal_checks(answers: dict, thresholds: Thresholds) -> list[Check]:
    """Guarantee 3, held from both sides.

    Takes the answer eval's report rather than the retrieval one. `min_` names
    a floor and `max_` a ceiling; the prefix is the whole rule, so adding a
    threshold needs no change here beyond naming it in REFUSAL_METRICS.
    """
    metrics = answers["metrics"]
    checks = []
    for name, metric in REFUSAL_METRICS.items():
        if metric not in metrics:
            raise ThresholdError(
                f"the answer report does not carry {metric}; the gate cannot "
                "hold a threshold on a metric the eval did not produce"
            )
        observed = metrics[metric]
        limit = thresholds.refusal[name]
        if name.startswith("min_"):
            passed = observed >= limit
            detail = "" if passed else "below floor"
        else:
            passed = observed <= limit
            detail = "" if passed else "above ceiling"
        if not passed and metric == "answered_unanswerable":
            detail = (
                f"{', '.join(metrics['answered_unanswerable_ids'])} answered "
                "despite the corpus not answering them"
            )
        checks.append(
            Check(
                name=metric,
                observed=observed,
                limit=limit,
                passed=passed,
                detail=detail,
            )
        )
    return checks


def freshness_check(status: IndexStatus) -> Check:
    """Guarantee 4: the committed index describes the committed corpus.

    Holds no threshold. A corpus edit landing without the rebuilt vectors
    turns the build red and the message names both what changed and the
    version the rebuild will produce, because CI cannot run `build-index`
    itself — that needs a key this workflow deliberately does not have.
    """
    return Check(
        name="index freshness",
        observed=status.version,
        limit=status.expected_version,
        passed=status.fresh,
        detail="" if status.fresh else "; ".join(d.line() for d in status.divergences),
    )


def run_gate(report: dict, thresholds: Thresholds) -> list[Check]:
    """Every check, in the order a reader wants them: aggregate first."""
    return (
        _aggregate_checks(report, thresholds)
        + _coverage_checks(report, thresholds)
        + _rank_checks(report, thresholds)
    )


def failures(checks: list[Check]) -> list[Check]:
    return [c for c in checks if not c.passed]
