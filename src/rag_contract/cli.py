"""Command line entry point.

Six commands, two of which touch the network:

    sections       inventory the parsed corpus; no network
    build-index    embed chunks and questions, write the committed artefacts;
                   calls the embedding API
    record-drafts  ask an answer model for the claims behind each question and
                   commit them as fixtures; calls the answer API
    eval           score retrieval against eval/questions.yaml; no network
    eval-answers   score failure behaviour against the committed drafts;
                   no network
    answer         answer one question from the committed fixtures, showing the
                   grounding check and the state it lands in; no network
    gate           run the eval and hold it to eval/thresholds.yaml; no network

The split is the point. `gate` is what CI runs, and everything under it is
deterministic — committed vectors, committed drafts, no key, no network, no
live LLM. The two commands that do call an API are run once by hand and their
output is committed as a build artefact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .answer_eval import evaluate_answers
from .answering import decide, passages_from_hits
from .chunking import ChunkParams, chunk_sections
from .corpus import load_documents
from .drafter import (
    FIXTURES_DIR,
    FixtureDrafter,
    record,
    write_recording,
)
from .drafter import MODEL as DRAFT_MODEL
from .evalset import load_questions, questions_fingerprint
from .evaluate import TOP_K, evaluate
from .gate import (
    THRESHOLDS_PATH,
    Check,
    failures,
    load_thresholds,
    refusal_checks,
    run_gate,
)
from .index import INDEX_DIR, load_index, write_index
from .retrieval import search
from .sections import parse_corpus

RESULTS_PATH = (
    Path(__file__).resolve().parents[2] / "eval" / "results" / "retrieval.json"
)
ANSWERS_PATH = Path(__file__).resolve().parents[2] / "eval" / "results" / "answers.json"


def cmd_sections(args: argparse.Namespace) -> int:
    documents = load_documents()
    sections = parse_corpus(documents)
    chunks = chunk_sections(sections)
    if args.json:
        print(
            json.dumps(
                {
                    "documents": len(documents),
                    "sections": len(sections),
                    "chunks": len(chunks),
                    "sections_with_prose": len({c.section_id for c in chunks}),
                },
                indent=2,
            )
        )
        return 0

    for document in documents:
        document_sections = [s for s in sections if s.rfc == document.rfc]
        document_chunks = [c for c in chunks if c.rfc == document.rfc]
        print(
            f"{document.rfc}  {len(document_sections):>4} sections  "
            f"{len(document_chunks):>4} chunks  {document.title}"
        )
    print(
        f"\ntotal: {len(sections)} sections, {len(chunks)} chunks, "
        f"{len({c.section_id for c in chunks})} sections with prose"
    )
    return 0


def cmd_build_index(args: argparse.Namespace) -> int:
    from .embedding import DIMENSIONS, MODEL, embed_texts

    documents = load_documents()
    sections = parse_corpus(documents)
    params = ChunkParams()
    chunks = chunk_sections(sections, params)
    questions = load_questions()

    print(
        f"embedding {len(chunks)} chunks and {len(questions)} questions "
        f"with {MODEL} at {DIMENSIONS} dimensions",
        file=sys.stderr,
    )
    chunk_vectors = embed_texts([c.embedding_text for c in chunks])
    question_vectors = embed_texts([q.question for q in questions])

    meta = write_index(
        documents=documents,
        chunks=chunks,
        vectors=chunk_vectors,
        question_ids=[q.id for q in questions],
        question_vectors=question_vectors,
        questions_fingerprint=questions_fingerprint(),
        section_count=len(sections),
        model=MODEL,
        dimensions=DIMENSIONS,
        params=params,
        directory=args.index_dir,
    )
    print(f"index {meta.version} written to {args.index_dir}", file=sys.stderr)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    index = load_index(args.index_dir)
    report = evaluate(index, load_questions(), top_k=args.top_k)
    payload = json.dumps(report, indent=2) + "\n"

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(f"wrote {args.output}", file=sys.stderr)
    if args.quiet:
        metrics = report["metrics"]
        print(
            f"index {report['index_version']}  "
            f"recall@1 {metrics['recall_at_1']}  "
            f"recall@5 {metrics['recall_at_5']}  "
            f"mrr {metrics['mrr']}  "
            f"misses {len(metrics['misses'])}"
        )
    else:
        sys.stdout.write(payload)
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    """Guarantee 2: a quality regression fails the build.

    The eval is always rerun here rather than read from `eval/results/`. A
    committed report is an artefact of whenever it was last written; the gate
    has to hold this commit's code against the committed thresholds, so it
    scores the index in front of it.
    """
    thresholds = load_thresholds(args.thresholds)
    index = load_index(args.index_dir)
    questions = load_questions()
    report = evaluate(index, questions, top_k=args.top_k)
    answers = evaluate_answers(
        index,
        questions,
        FixtureDrafter.from_directory(args.fixtures_dir),
        top_k=args.top_k,
    )

    checks = run_gate(report, thresholds) + refusal_checks(answers, thresholds)
    if args.check_report:
        checks.append(_report_check(args.report, report))
        checks.append(_report_check(args.answers, answers, "eval-answers"))
    failed = failures(checks)

    print(f"index {report['index_version']} against {args.thresholds.name}")
    for check in checks if not args.quiet else failed:
        print(check.line())

    if failed:
        print(f"\ngate failed: {len(failed)} of {len(checks)} checks below the bar")
        return 1
    print(f"\ngate passed: {len(checks)} checks at or above the committed bar")
    return 0


def _report_check(path: Path, report: dict, command: str = "eval") -> Check:
    """The committed report must still describe what the eval produces.

    Every number in the README is quoted from `eval/results/`. If those files
    can drift from the code that produced them, the README is quoting numbers
    no commit ever measured.
    """
    if not path.exists():
        detail = f"{path} does not exist"
    elif path.read_text() != json.dumps(report, indent=2) + "\n":
        detail = f"{path.name} is stale; rerun `rag-contract {command} -o`"
    else:
        detail = ""
    return Check(
        name=f"committed {path.stem}",
        observed=None if detail else "current",
        limit="current",
        passed=not detail,
        detail=detail,
    )


def cmd_record_drafts(args: argparse.Namespace) -> int:
    """The one command that calls an answer model. Run by hand, output committed.

    Every other step in guarantee 3 is arithmetic. This is not, which is why
    its output is a committed fixture rather than something CI reproduces: the
    grounding check, the state machine and retrieval all rerun on replay, and
    only the model's own words are frozen.
    """
    from .drafter import LiveDrafter

    index = load_index(args.index_dir)
    questions = load_questions()
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q.id in wanted]
        if not questions:
            print(f"no question matches {args.only}", file=sys.stderr)
            return 1

    drafter = LiveDrafter(model=args.model)
    print(
        f"drafting {len(questions)} questions with {args.model} over index "
        f"{index.version}",
        file=sys.stderr,
    )
    for question in questions:
        hits = search(index, index.question_vector(question.id), k=args.top_k)
        passages = passages_from_hits(hits)
        claims = drafter.draft(question.question, passages)
        path = write_recording(
            record(
                question_id=question.id,
                question=question.question,
                passages=passages,
                claims=claims,
                index_version=index.version,
                model=args.model,
            ),
            args.fixtures_dir,
        )
        print(
            f"{question.id}  {len(claims):>2} claims  {len(passages):>2} passages"
            f"  -> {path.name}",
            file=sys.stderr,
        )
    return 0


def cmd_answer(args: argparse.Namespace) -> int:
    """Answer one question from the committed drafts. No network, no key.

    This is the service's behaviour made inspectable: what was retrieved, what
    the model wanted to say, which claims survived the grounding check, and
    which of guarantee 3's states the request ended in.
    """
    index = load_index(args.index_dir)
    questions = {q.id: q for q in load_questions()}
    question = questions.get(args.question_id)
    if question is None:
        print(f"no question {args.question_id}", file=sys.stderr)
        return 1

    hits = search(index, index.question_vector(question.id), k=args.top_k)
    passages = passages_from_hits(hits)
    drafter = FixtureDrafter.from_directory(args.fixtures_dir)
    answer = decide(
        question=question.question,
        index_version=index.version,
        passages=passages,
        claims=drafter.draft(question.question, passages),
    )

    if args.json:
        print(json.dumps(answer.to_dict(), indent=2))
        return 0

    print(f"{question.id}  {question.question}")
    print(
        f"state {answer.state}  http {answer.state.http_status}  "
        f"index {answer.index_version}"
    )
    if answer.message:
        print(f"\n{answer.message}")
    if answer.supported:
        print("\nanswer:")
        for verdict in answer.supported:
            print(f"  {verdict.text} [{verdict.citation}]")
    if answer.withdrawn:
        print("\nwithdrawn:")
        for verdict in answer.withdrawn:
            print(f"  {verdict.text}")
            print(f"    {verdict.rule}: {verdict.detail}")
    return 0


def cmd_eval_answers(args: argparse.Namespace) -> int:
    """Guarantee 3's measurement. Replays committed drafts; no network, no key.

    Only the model's words come from the fixtures. Retrieval, the grounding
    check and the state machine all rerun here, so a change to any of them is
    measured rather than frozen out of the report.
    """
    index = load_index(args.index_dir)
    drafter = FixtureDrafter.from_directory(args.fixtures_dir)
    report = evaluate_answers(index, load_questions(), drafter, top_k=args.top_k)
    payload = json.dumps(report, indent=2) + "\n"

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(f"wrote {args.output}", file=sys.stderr)
    if args.quiet:
        metrics = report["metrics"]
        print(
            f"index {report['index_version']}  "
            f"grounded {metrics['grounded_answerable']}/"
            f"{metrics['answerable_questions']}  "
            f"answered-unanswerable {metrics['answered_unanswerable']}  "
            f"state agreement {metrics['state_agreement']}  "
            f"withdrawn {metrics['withdrawn_claims']}"
        )
    else:
        sys.stdout.write(payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-contract")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sections = subparsers.add_parser(
        "sections", help="inventory the parsed corpus (no network)"
    )
    sections.add_argument("--json", action="store_true", help="machine-readable counts")
    sections.set_defaults(func=cmd_sections)

    build = subparsers.add_parser(
        "build-index",
        help="embed chunks and questions and write the committed index "
        "(the only command that calls the embedding API)",
    )
    build.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    build.set_defaults(func=cmd_build_index)

    evaluate_cmd = subparsers.add_parser(
        "eval", help="score retrieval against eval/questions.yaml (no network)"
    )
    evaluate_cmd.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    evaluate_cmd.add_argument("--top-k", type=int, default=TOP_K)
    evaluate_cmd.add_argument(
        "-o",
        "--output",
        type=Path,
        nargs="?",
        const=RESULTS_PATH,
        help=f"write the report to a file (default {RESULTS_PATH.name} when bare)",
    )
    evaluate_cmd.add_argument(
        "-q", "--quiet", action="store_true", help="print a one-line summary only"
    )
    evaluate_cmd.set_defaults(func=cmd_eval)

    drafts = subparsers.add_parser(
        "record-drafts",
        help="ask an answer model for the claims behind each question and "
        "commit them as fixtures (calls the answer API)",
    )
    drafts.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    drafts.add_argument("--fixtures-dir", type=Path, default=FIXTURES_DIR)
    drafts.add_argument("--top-k", type=int, default=TOP_K)
    drafts.add_argument("--model", default=DRAFT_MODEL)
    drafts.add_argument(
        "--only", help="comma-separated question ids, for re-recording a few"
    )
    drafts.set_defaults(func=cmd_record_drafts)

    answer = subparsers.add_parser(
        "answer",
        help="answer one question from the committed drafts, showing the "
        "grounding check and the state it lands in (no network)",
    )
    answer.add_argument("question_id")
    answer.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    answer.add_argument("--fixtures-dir", type=Path, default=FIXTURES_DIR)
    answer.add_argument("--top-k", type=int, default=TOP_K)
    answer.add_argument("--json", action="store_true", help="the full response")
    answer.set_defaults(func=cmd_answer)

    answers = subparsers.add_parser(
        "eval-answers",
        help="score failure behaviour against the committed drafts (no network)",
    )
    answers.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    answers.add_argument("--fixtures-dir", type=Path, default=FIXTURES_DIR)
    answers.add_argument("--top-k", type=int, default=TOP_K)
    answers.add_argument(
        "-o",
        "--output",
        type=Path,
        nargs="?",
        const=ANSWERS_PATH,
        help=f"write the report to a file (default {ANSWERS_PATH.name} when bare)",
    )
    answers.add_argument(
        "-q", "--quiet", action="store_true", help="print a one-line summary only"
    )
    answers.set_defaults(func=cmd_eval_answers)

    gate = subparsers.add_parser(
        "gate",
        help="run the eval and fail below eval/thresholds.yaml (no network)",
    )
    gate.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    gate.add_argument("--top-k", type=int, default=TOP_K)
    gate.add_argument("--thresholds", type=Path, default=THRESHOLDS_PATH)
    gate.add_argument("--report", type=Path, default=RESULTS_PATH)
    gate.add_argument("--answers", type=Path, default=ANSWERS_PATH)
    gate.add_argument("--fixtures-dir", type=Path, default=FIXTURES_DIR)
    gate.add_argument(
        "--check-report",
        action="store_true",
        help="also fail when the committed report no longer matches this run",
    )
    gate.add_argument(
        "-q", "--quiet", action="store_true", help="print failing checks only"
    )
    gate.set_defaults(func=cmd_gate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
