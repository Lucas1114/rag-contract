"""Command line entry point.

Three commands, one of which touches the network:

    sections     inventory the parsed corpus; no network
    build-index  embed chunks and questions, write the committed artefacts;
                 the only command that calls the embedding API
    eval         score retrieval against eval/questions.yaml; no network

The split is the point. `eval` is what CI runs, and it is deterministic numpy
over committed vectors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .chunking import ChunkParams, chunk_sections
from .corpus import load_documents
from .evalset import load_questions, questions_fingerprint
from .evaluate import TOP_K, evaluate
from .index import INDEX_DIR, load_index, write_index
from .sections import parse_corpus

RESULTS_PATH = (
    Path(__file__).resolve().parents[2] / "eval" / "results" / "retrieval.json"
)


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
