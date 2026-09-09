# rag-contract

A question-answering service over a small, fixed corpus.

The retrieval pipeline is not the point. Chunking, embedding, retrieving and
stuffing a prompt is commodity work and demonstrates nothing. The point is
treating retrieval as a service with a contract: quality that is measured
rather than asserted, a gate that fails the build when it regresses, defined
behaviour when retrieval fails, a versioned data lifecycle, and budgets that
are enforced rather than logged.

## What this service guarantees

| # | Guarantee | How it is verified |
|---|-----------|--------------------|
| 1 | Retrieval quality is measured, not claimed | A fixed question set annotates the passages that should support each answer. The eval command reports recall over those passages and their rank positions, as machine-readable output. |
| 2 | Quality regressions fail the build | The threshold is committed to `eval/thresholds.yaml`. CI runs the eval and fails below it. Threshold changes are visible in diffs. |
| 3 | The service refuses rather than invents | Empty retrieval, low-confidence retrieval and wrong-passage retrieval are three distinct states with three distinct responses. Every answer carries a grounding check naming the passage that supports each claim. |
| 4 | Every answer is attributable to an index version | Index versions are content-addressed and returned in response metadata. Corpus changes trigger a rebuild, and cutover does not drop in-flight requests. |
| 5 | Latency and cost are ceilings, not advice | Per-request latency and token/cost ceilings change behaviour when exceeded rather than merely recording it. Both are reported per request. |

## Status

This table is updated at each stage commit. `specified` means defined but not
yet implemented, `implemented` means the code exists, `verified` means eval
data or CI results support it.

| Part | Status |
|------|--------|
| 1. Evaluation harness | specified |
| 2. CI regression gate | specified |
| 3. Failure behaviour | specified |
| 4. Index lifecycle | specified |
| 5. Budgets | specified |

## Corpus

Six IETF RFCs, committed verbatim under `corpus/`, covering one coherent layer:
how a resource is named, how a client talks to it, how responses are cached,
how state is carried, and the dominant payload format.

| RFC | Title |
|-----|-------|
| 3986 | Uniform Resource Identifier (URI): Generic Syntax |
| 6265 | HTTP State Management Mechanism |
| 8259 | The JavaScript Object Notation (JSON) Data Interchange Format |
| 9110 | HTTP Semantics |
| 9111 | HTTP Caching |
| 9112 | HTTP/1.1 |

`corpus/manifest.yaml` records the source URL, retrieval date, byte length and
sha256 of each file. Those hashes are inputs to the index version.

RFCs were chosen because answers can be checked against the source text by a
reviewer with no tooling — the entire evaluation story depends on being able to
say whether an answer was right — and because their numbered sections map
directly onto expected-passage annotations.

The set is deliberately narrow rather than broad. A corpus spanning unrelated
protocols would make "this corpus cannot answer that" detectable from
vocabulary alone; keeping every document in the same subject area means the
refusal path has to be exercised by questions that genuinely look in scope.

This is not a general-purpose chatbot. Questions are answered against this
corpus only, and there is no open-ended prompt input.

## Question set

`eval/questions.yaml` defines what "correct" means for the whole project: 20
questions with annotated supporting passages, and 8 questions the corpus cannot
answer, which must be refused rather than answered.

Expected passages are annotated as RFC **section** ids (`rfc9110#9.2.1`), never
chunk ids. The harness that consumes this file is `specified`, not yet
implemented; the contract it will be held to is that each retrieved chunk
resolves to the section it came from, and a question scores a hit when an
expected section appears in the top results. Annotating at section level keeps
the question set valid across changes to chunk size or overlap, which are tuning
knobs the harness measures rather than properties it depends on.

Two annotations are deliberately deferred until the harness has produced a real
score distribution, and are documented as such in the file: multi-passage
questions, and which of the three failure states each refusal question should
land in. The second depends on a similarity threshold that has not been
measured yet, so annotating it now would guarantee re-annotation later.

## Design decisions

**No vector database.** At this corpus size, brute-force cosine similarity over
numpy is the correct choice and stays far below the latency ceiling. A vector
store would add a component to version and operate for no measurable benefit.

**Embedding vectors are committed.** The corpus is fixed, so the vectors are a
deterministic build artifact. Committing them means the retrieval eval runs in
CI with no network calls, no cost, and reproducible results.

**CI never calls a real LLM.** Retrieval evaluation is pure computation.
Answer-level evaluation replays recorded responses. Live calls happen only in
local runs, whose results are written to `eval/results/`.
