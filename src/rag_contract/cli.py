"""Command line entry point.

Eight commands, two of which touch the network:

    sections       inventory the parsed corpus; no network
    index-status   whether the committed index still describes the corpus;
                   no network
    build-index    embed chunks and questions, write the committed artefacts;
                   calls the embedding API
    record-drafts  ask an answer model for the claims behind each question and
                   commit them as fixtures; calls the answer API
    eval           score retrieval against eval/questions.yaml; no network
    eval-answers   score failure behaviour against the committed drafts;
                   no network
    answer         answer one question from the committed fixtures, showing the
                   grounding check and the state it lands in; no network
    gate           run both evals and hold them, the index freshness check, the
                   budgets and the served rate limit to eval/thresholds.yaml;
                   no network

The split is the point. `gate` is what CI runs, and everything under it is
deterministic — committed vectors, committed drafts, no key, no network, no
live LLM. The two commands that do call an API are run once by hand and their
output is committed as a build artefact.

Those two are also the only two that can spend money, so they are the two that
carry a spend cap. It is asked before every request and refuses the ones that
would break the day's budget, which means a command stopping with nothing
written rather than a log line after the fact. `spend.py` has the reasoning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .answer_eval import evaluate_answers
from .answering import decide, passages_from_hits
from .budget import measure_request_ms
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
    budget_checks,
    failures,
    freshness_check,
    load_thresholds,
    rate_limit_check,
    refusal_checks,
    run_gate,
    trusted_hop_check,
)
from .index import INDEX_DIR, load_index, write_index
from .lifecycle import index_status
from .registry import IndexRegistry
from .retrieval import search
from .sections import parse_corpus
from .service import Service
from .spend import Cap, Ledger, completion_bound_usd, embedding_usd

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


def cmd_index_status(args: argparse.Namespace) -> int:
    """Guarantee 4: does the committed index still describe the corpus?

    Exits non-zero when it does not, so the same question the gate asks can be
    asked directly, before a commit rather than after CI rejects it. Hashing
    only: no network, no key.
    """
    status = index_status(load_index(args.index_dir))
    if args.json:
        print(json.dumps(status.to_dict(), indent=2))
    else:
        print(status.message)
        for divergence in status.divergences:
            print(f"  {divergence.line()}")
    return 0 if status.fresh else 1


def cmd_build_index(args: argparse.Namespace) -> int:
    """One of the two commands that spend money, and so one of two that are capped.

    The cap is asked before each batch. A corpus that outgrew the budget stops
    here with no index written and the committed one untouched, which is the
    same shape as a failed reload leaving the registry on the last index known
    to be good.
    """
    from .embedding import DIMENSIONS, MODEL, embed_texts

    documents = load_documents()
    sections = parse_corpus(documents)
    params = ChunkParams()
    chunks = chunk_sections(sections, params)
    questions = load_questions()

    cap = Cap(
        daily_usd=load_thresholds(args.thresholds).budget.daily_cap_usd,
        ledger=Ledger(),
        command="build-index",
    )
    texts = [c.embedding_text for c in chunks] + [q.question for q in questions]
    print(
        f"embedding {len(chunks)} chunks and {len(questions)} questions "
        f"with {MODEL} at {DIMENSIONS} dimensions; "
        f"~${embedding_usd(texts, MODEL):.4f} against ${cap.remaining():.4f} "
        f"left of today's ${cap.daily_usd:.2f}",
        file=sys.stderr,
    )
    chunk_vectors = embed_texts([c.embedding_text for c in chunks], cap=cap)
    question_vectors = embed_texts([q.question for q in questions], cap=cap)

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

    checks = (
        run_gate(report, thresholds)
        + refusal_checks(answers, thresholds)
        + [freshness_check(index_status(index))]
        + _budget_checks(index, questions, thresholds, args)
    )
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


def _budget_checks(index, questions, thresholds, args) -> list[Check]:
    """Guarantee 5's numbers, computed here and compared in `gate.py`.

    Two are arithmetic over committed artefacts and need no key: the corpus
    prices an embedding rebuild, and the committed prompts and drafts price a
    re-record from below. The third times the real served request path, because
    there is no other way to know what it costs — and it is timed through
    `Service` rather than through a hand-assembled pipeline so that what the
    gate measures is what the HTTP surface actually runs.

    The timed service is deliberately built without a budget. Timing a request
    that is simultaneously being held to a deadline would make the measurement
    able to fail for the thing it is measuring, and a gate check that can raise
    instead of reporting a number is not a check.
    """
    from .drafter import MAX_TOKENS, SYSTEM, _prompt
    from .drafter import MODEL as DRAFT_MODEL_NAME
    from .embedding import MODEL as EMBED_MODEL

    chunks = chunk_sections(parse_corpus(load_documents()), ChunkParams())
    build_usd = embedding_usd(
        [c.embedding_text for c in chunks] + [q.question for q in questions],
        EMBED_MODEL,
    )

    drafter = FixtureDrafter.from_directory(args.fixtures_dir)
    record_usd = 0.0
    worst_case_usd = 0.0
    for question in questions:
        passages = passages_from_hits(
            search(index, index.question_vector(question.id), k=args.top_k)
        )
        prompt = SYSTEM + _prompt(question.question, passages)
        drafted = " ".join(c.text for c in drafter.draft(question.question, passages))
        # The committed drafts hold the text the model returned and cannot hold
        # the reasoning tokens it was also billed for, so this prices the
        # recorded output as if it were the whole of it: a floor, and named as
        # one wherever it is quoted.
        record_usd += completion_bound_usd(prompt, 0, DRAFT_MODEL_NAME) + (
            completion_bound_usd("", len(drafted) // 3, DRAFT_MODEL_NAME)
        )
        worst_case_usd += completion_bound_usd(prompt, MAX_TOKENS, DRAFT_MODEL_NAME)

    service = Service(
        registry=IndexRegistry(index),
        questions={q.id: q for q in questions},
        drafter=drafter,
        top_k=args.top_k,
    )
    slowest, slowest_ms, _ = measure_request_ms(
        service.answer, [q.id for q in questions]
    )
    return budget_checks(
        slowest_question=slowest,
        slowest_ms=slowest_ms,
        build_index_usd=build_usd,
        record_drafts_usd=record_usd,
        worst_case_record_usd=worst_case_usd,
        thresholds=thresholds,
    ) + [
        _rate_limit_check(service, thresholds),
        _trusted_hop_check(service, thresholds),
    ]


def _rate_limit_check(service: Service, thresholds) -> Check:
    """Guarantee 5's third ceiling, checked through the surface that holds it.

    Composed here from the committed numbers and driven with a frozen clock:
    nothing refills, so one client spends its whole burst and the next request
    is refused, on any machine and in the same number of requests. The clock is
    the only thing injected — the allowance comes from `eval/thresholds.yaml`,
    because a check against a limiter the check configured would be a check on
    nothing.

    `/questions` rather than `/answer`, because the limiter runs before any
    handler and the cheapest route proves the same thing while leaving the
    latency measurement above unperturbed.
    """
    from .app import create_app, probe

    application = create_app(
        service, limiter=thresholds.budget.limiter(clock=lambda: 0.0)
    )
    results = probe(
        application,
        "/questions",
        client_host="10.0.0.1",
        count=thresholds.budget.max_client_burst + 1,
    )
    statuses = [status for status, _ in results]
    refused = next((headers for status, headers in results if status == 429), {})
    return rate_limit_check(
        statuses=statuses,
        retry_after=refused.get("retry-after"),
        thresholds=thresholds,
    )


def _trusted_hop_check(service: Service, thresholds) -> Check:
    """Guarantee 5's third ceiling again, at the other end: who it applies to.

    Two runs through the real surface from one socket, both on a frozen clock
    so nothing refills and the outcome is the same on any machine.

    The first varies the forgeable part of `X-Forwarded-For` and holds the
    trusted hop constant, so every request is the same client however the
    header is dressed up, and the surface has to refuse past the burst. The
    second varies the trusted hop itself, so every request is a different
    client arriving through the same proxy, and none of them may be refused —
    unless nothing is trusted, in which case the header must change nothing and
    the burst must still bite.

    Both are built from `eval/thresholds.yaml`, including the hop count, for
    the reason `_rate_limit_check` is: a check that configured the surface it
    then measured would be a check on nothing.
    """
    from .app import create_app, probe

    hops = thresholds.budget.trusted_proxy_hops
    burst = thresholds.budget.max_client_burst
    count = burst + 1

    def chain(client: str, nonce: str) -> str:
        """One `X-Forwarded-For` whose entry `hops` from the right is `client`.

        Everything left of it is the part a caller writes; everything right of
        it stands for the proxies between that hop and this process.
        """
        entries = [nonce, client] + [f"10.9.9.{i}" for i in range(1, max(hops, 1))]
        return ", ".join(entries)

    def statuses(headers: list[dict[str, str]]) -> list[int]:
        application = create_app(
            service,
            limiter=thresholds.budget.limiter(clock=lambda: 0.0),
            trusted_proxy_hops=hops,
        )
        return [
            status
            for status, _ in probe(
                application,
                "/questions",
                client_host="10.0.0.1",
                count=count,
                headers=headers,
            )
        ]

    return trusted_hop_check(
        forged_statuses=statuses(
            [
                {"X-Forwarded-For": chain("203.0.113.7", f"1.2.3.{i}")}
                for i in range(count)
            ]
        ),
        distinct_statuses=statuses(
            [
                {"X-Forwarded-For": chain(f"203.0.113.{i}", "1.2.3.4")}
                for i in range(count)
            ]
        ),
        thresholds=thresholds,
    )


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

    cap = Cap(
        daily_usd=load_thresholds(args.thresholds).budget.daily_cap_usd,
        ledger=Ledger(),
        command="record-drafts",
    )
    drafter = LiveDrafter(model=args.model, cap=cap)
    print(
        f"drafting {len(questions)} questions with {args.model} over index "
        f"{index.version}; ${cap.remaining():.4f} left of today's "
        f"${cap.daily_usd:.2f}",
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
            f"  -> {path.name}  (${cap.spent_today():.4f} spent today)",
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


DEFAULT_PORT = 8000


def _port_from_environment() -> int:
    """The one thing about this process the environment gets to decide.

    Everything that governs behaviour — the deadline, the day's cap, the
    allowance, who a client is — is committed to `eval/thresholds.yaml`
    precisely so that a deployment cannot quietly change it. The port is the
    exception because it is not behaviour: a platform assigns a socket and the
    process either binds the one it was given or is unreachable. `PORT` is that
    platform convention, and nothing about the contract depends on its value.
    """
    raw = os.environ.get("PORT")
    if not raw:
        return DEFAULT_PORT
    try:
        port = int(raw)
    except ValueError as exc:
        raise SystemExit(f"PORT={raw!r} is not a port number") from exc
    if not 1 <= port <= 65535:
        raise SystemExit(f"PORT={port} is outside 1..65535")
    return port


def cmd_serve(args: argparse.Namespace) -> int:
    """Serve the committed index over HTTP.

    Since P05 the way to run this service has been to name uvicorn and an
    import path in the README, which is a run instruction rather than an
    entry point: it puts the ASGI server, the module path and the bind address
    in the reader's hands, and a container has to repeat all three. This is the
    same process with one name, so the README, the Dockerfile and a developer
    all start it the same way.

    Nothing here composes the service. `create_app` does that from the
    committed thresholds, in the lifespan, so a process started by hand and a
    process started by a platform are the same process — including when there
    is no usable index, which starts anyway and says so on `/health` rather
    than exiting.
    """
    import uvicorn

    from .app import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level=args.log_level)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-contract")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sections = subparsers.add_parser(
        "sections", help="inventory the parsed corpus (no network)"
    )
    sections.add_argument("--json", action="store_true", help="machine-readable counts")
    sections.set_defaults(func=cmd_sections)

    status = subparsers.add_parser(
        "index-status",
        help="whether the committed index still describes the corpus on disk "
        "(no network)",
    )
    status.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    status.add_argument("--json", action="store_true", help="the full status")
    status.set_defaults(func=cmd_index_status)

    build = subparsers.add_parser(
        "build-index",
        help="embed chunks and questions and write the committed index "
        "(the only command that calls the embedding API)",
    )
    build.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    build.add_argument("--thresholds", type=Path, default=THRESHOLDS_PATH)
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
    drafts.add_argument("--thresholds", type=Path, default=THRESHOLDS_PATH)
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

    serve = subparsers.add_parser(
        "serve", help="serve the committed index over HTTP (no network, no key)"
    )
    serve.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address (default 127.0.0.1; a container wants 0.0.0.0)",
    )
    serve.add_argument(
        "--port",
        type=int,
        default=_port_from_environment(),
        help=f"bind port (default $PORT, or {DEFAULT_PORT})",
    )
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=cmd_serve)

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
