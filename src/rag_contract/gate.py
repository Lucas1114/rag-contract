"""The regression gate: guarantee 2.

Guarantee 1 measures retrieval quality. This is what makes that measurement
load-bearing — a committed threshold file, a build that fails below it, and a
diff whenever the bar moves.

The gate reads `eval/thresholds.yaml` and a report produced by `evaluate`. It
computes nothing itself and holds no numbers of its own: every floor is in the
committed file, so lowering the bar is an edit someone has to make and a
reviewer can see.

Two kinds of check, and they are independent:

    aggregate   floors on recall@k and MRR over the answerable questions
    per-question  a rank ceiling for each answerable question

Both exist because either alone is blind. Aggregates miss compensating
movement — one question improving while another collapses leaves recall@1 flat
and MRR higher. Per-question ceilings miss uniform drift that stays inside
every ceiling while every question gets worse. A build passes only when both
agree.

A third check keeps the file honest: the set of gated questions must be exactly
the set of answerable questions in the report. A question added to the question
set without a recorded ceiling fails the gate rather than slipping in ungated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

THRESHOLDS_PATH = Path(__file__).resolve().parents[2] / "eval" / "thresholds.yaml"

AGGREGATE_METRICS = ("recall_at_1", "recall_at_5", "recall_at_10", "mrr")


class ThresholdError(RuntimeError):
    """The threshold file is missing, malformed, or not about this report."""


@dataclass(frozen=True)
class Thresholds:
    aggregate: dict[str, float]
    max_rank: dict[str, int]
    measured: dict


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

    return Thresholds(
        aggregate={k: float(v) for k, v in aggregate.items()},
        max_rank={str(k): int(v) for k, v in max_rank.items()},
        measured=raw.get("measured") or {},
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


def run_gate(report: dict, thresholds: Thresholds) -> list[Check]:
    """Every check, in the order a reader wants them: aggregate first."""
    return (
        _aggregate_checks(report, thresholds)
        + _coverage_checks(report, thresholds)
        + _rank_checks(report, thresholds)
    )


def failures(checks: list[Check]) -> list[Check]:
    return [c for c in checks if not c.passed]
