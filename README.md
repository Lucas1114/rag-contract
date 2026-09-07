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

Small, fixed, committed to the repository. Chosen so that answers can be
checked against the source text — the entire evaluation story depends on being
able to say whether an answer was right.

This is not a general-purpose chatbot. Questions are answered against this
corpus only, and there is no open-ended prompt input.

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
