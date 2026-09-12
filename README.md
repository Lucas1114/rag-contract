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
| 3 | The service refuses rather than invents | Every answer is a list of claims, each naming the passage that supports it, and a grounding check decides per claim whether that passage actually contains it. Three distinct failure states follow from that check, with three distinct responses. |
| 4 | Every answer is attributable to an index version | Index versions are content-addressed and returned in response metadata. Corpus changes trigger a rebuild, and cutover does not drop in-flight requests. |
| 5 | Latency and cost are ceilings, not advice | Per-request latency and token/cost ceilings change behaviour when exceeded rather than merely recording it. Both are reported per request. |

## Status

This table is updated at each stage commit. `specified` means defined but not
yet implemented, `implemented` means the code exists, `verified` means eval
data or CI results support it.

| Part | Status |
|------|--------|
| 1. Evaluation harness | verified |
| 2. CI regression gate | verified |
| 3. Failure behaviour | verified |
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
chunk ids. Every retrieved chunk resolves to the section it came from, and a
question scores a hit at k when any chunk in the top k carries an expected
section. Annotating at section level keeps the question set valid across changes
to chunk size or overlap, which are tuning knobs the harness measures rather
than properties it depends on.

Two annotations were deliberately deferred until the harness had produced a real
score distribution. One has since been filled in: every refusal question now
carries an `expected_state` naming which failure state it must land in, and the
loader rejects a refusal question that does not — or that claims it should be
answered. The other stands: `expected` is a list but still always holds one
section, pending a `match: all` mode for comparison questions.

The distribution argued against the simplest way to draw those state
boundaries, which is why they are drawn where they are — see below.

## Evaluation harness

The corpus parses into 600 numbered sections, 564 of which carry prose of their
own, and chunks into 868 embedded units at 220 words with 40 words of overlap.
No chunk spans a section, so every hit reports the section it came from.

    rag-contract sections       inventory the parsed corpus            no network
    rag-contract build-index    embed and write the index              calls the API once
    rag-contract record-drafts  record the answer model's claims       calls the API once
    rag-contract eval           score against eval/questions.yaml      no network
    rag-contract eval-answers   score the failure behaviour            no network
    rag-contract answer         answer one question, showing its state no network
    rag-contract gate           hold both evals to eval/thresholds.yaml no network

`eval` is deterministic numpy over committed vectors. Same index, same
questions, same numbers, on any machine — which is the property the CI gate in
part 2 is built on.

### Results

Index `0fc1763d6701`, `text-embedding-3-small` at 1536 dimensions, over the 20
answerable questions. Full report in
[`eval/results/retrieval.json`](eval/results/retrieval.json).

| recall@1 | recall@3 | recall@5 | recall@10 | MRR | misses |
|----------|----------|----------|-----------|-----|--------|
| 0.70 | 0.95 | 1.00 | 1.00 | 0.829 | 0 |

Every annotated section is retrieved within the top 5. The six questions that
miss at rank 1 lose to a plausible neighbour rather than to noise. q17 and q18
rank the Set-Cookie attribute *syntax* in Section 4.1.2 above the user agent
*processing* rules in Section 5 that actually define HttpOnly and domain
matching. The worst placement, q16 at rank 4, asks which request-target form
CONNECT uses and is beaten by RFC 9110's definition of the CONNECT method
itself — the right topic in the wrong document.

### What the score distribution says about refusal

The question set left the three failure states of guarantee 3 unannotated
because their boundaries are a threshold on similarity that had not been
measured. Measured, that threshold does not exist:

| | count | min | median | max |
|---|---|---|---|---|
| answerable | 20 | 0.592 | 0.729 | 0.805 |
| unanswerable | 8 | 0.375 | 0.504 | **0.604** |

The two distributions overlap. q20 — "what are the six structural characters in
JSON?", answered squarely by RFC 8259 Section 2 at rank 1 — scores 0.592, below
the 0.604 that u02 scores asking about HTTP/2 frame layout, which the corpus
does not contain at all. No global cutoff separates them: any threshold that
refuses u02 also refuses a question the corpus answers correctly.

This is evidence for the design guarantee 3 already commits to rather than
against it. Refusal cannot be a similarity threshold alone; it needs the
grounding check that names the passage supporting each claim. Part 3 is built on
that measurement rather than on an assumption, and one of the three states
originally specified did not survive it.

One prediction recorded during annotation held exactly. u01 asks how HTTP/2
multiplexes requests; RFC 9110 Section 1.2 mentions that HTTP/2 introduced a
multiplexed session layer without describing streams or frames, and it was
annotated as a passage retrieval would surface anyway. It ranks first, at 0.548.

## Failure behaviour

An answer is not prose. It is a list of claims, each naming the single section
that supports it, and `rag-contract answer` shows both the claims that survived
and the ones that did not.

### The grounding check

Three rules, applied per claim, all of them arithmetic over strings:

1. **The citation must be among the retrieved passages.** The drafter saw those
   and nothing else, so a claim citing anything else was written from the
   model's own memory whatever it says.
2. **Every literal in the claim must occur in the cited passage.** Literals are
   the tokens that carry the fact and get invented when a model is guessing:
   numbers, and tokens spelled with internal capitals or hyphens. A claim
   asserting a status code the passage never mentions fails here no matter how
   well the rest of it reads.
3. **Content-word coverage must clear a floor.** A claim rephrases its passage
   rather than quoting it, so the floor is two thirds rather than everything.

This is lexical overlap and not entailment, and `grounding.py` says so rather
than hiding it. A model could defeat rule 3 by quoting a passage and negating
it. What the check buys instead is that it is deterministic: it runs in CI with
no key, no cost and no second model to trust, and it catches the failure that
actually occurs — a fluent sentence asserting a specific fact the cited passage
does not contain. A model-graded check would read stronger and would move the
guarantee onto an unverifiable dependency, which is the opposite of the point.

### The three states

They are distinguished by where the failure is detected, which is decidable per
request without consulting a similarity score anywhere.

| State | Reached when | Response |
|-------|--------------|----------|
| `grounded` | every claim survived | the claims, each with its citation |
| `partial` | some claims survived | only those, plus the withdrawn ones named with the rule that dropped them |
| `unsupported` | no claim survived | no answer: a refusal, with the passages consulted and the claims rejected |
| `no_context` | nothing was retrieved | a service error, not a decision — HTTP 503 |

The README previously named the third failure state "low-confidence retrieval".
That state does not exist. It presumes a similarity band separating answerable
questions from unanswerable ones, and the distribution above is the measurement
that says there is none. `partial` replaces it and is the more valuable state:
the case where the corpus *mentions* a subject without answering it. RFC 9110
Section 1.2 says HTTP/2 introduced a multiplexed session layer and never says
how; RFC 9112 Section 9.7 has a client sending a ClientHello and gives no
handshake. Retrieval surfaces those passages at perfectly ordinary scores, a
drafter reads them and supplies the rest from memory, and no confidence number
anywhere in the pipeline is disturbed. That is the failure this guarantee exists
to catch, and no threshold on retrieval could have caught it.

Each of the eight unanswerable questions is annotated with the state it must
land in. The rule: `partial` when the corpus holds a true, relevant, groundable
statement that stops short of the answer, `unsupported` when it holds no
foothold at all. No question is annotated `no_context`, and `questions.yaml`
says why rather than manufacturing one — brute-force cosine over 868 chunks
always returns ten, so that state belongs to an unavailable index and is held by
unit tests instead.

### Keeping the model out of CI

Drafting the claims is the one step in this project that is not arithmetic,
which makes it the one step CI must never run. It sits behind an interface with
two implementations: `LiveDrafter` calls the API, `FixtureDrafter` replays a
committed recording, and only the second is reachable from the gate. This is the
index story again — computed once by hand, committed as a build artefact.

A fixture freezes the model's output and nothing else. Retrieval, the grounding
check and the state machine all rerun on every replay against the live index, so
changing the coverage floor or the chunk parameters is *measured* by the answer
eval rather than frozen out of it. Freezing the verdicts would produce an eval
incapable of ever failing.

### Results

Drafted by `gpt-5.5-2026-04-23` over index `0fc1763d6701`. Full report in
[`eval/results/answers.json`](eval/results/answers.json), drafts in
[`eval/fixtures/drafts/`](eval/fixtures/drafts).

| grounded (of 20 answerable) | state agreement (of 8 refusals) | answered anyway | claims withdrawn |
|---|---|---|---|
| 0.90 | 0.875 | 1 | 3 of 72 |

The two answerable questions that are not fully grounded are `partial`, not
broken: q10 writes "freshness-lifetime" where RFC 9111 writes it as two words,
and two q14 claims sit at 0.64 and 0.667 against the 0.67 coverage floor. Both
questions answer, each having said one hedge sentence less.

Measuring falsified two of the eight state annotations. u03 and u07 were
specified `partial` on footholds that turned out to answer adjacent questions
rather than these — the corpus says which hosts receive a cookie, not how
SameSite treats cross-site requests; it names a ClientHello without giving a
handshake. Nothing was drafted against either, and both are now `unsupported`.
Annotating the expectation before measuring is what made those corrections
visible as corrections rather than as edits.

### What this check does not do

`answered_unanswerable` is 1, and the gate holds it at 1 rather than at 0.

u01 asks how HTTP/2 multiplexes requests over one TCP connection. RFC 9110
Section 1.2 says that HTTP/2 introduced a multiplexed session layer and never
says how. The drafter returned exactly that sentence, cited correctly, and the
response landed in `grounded`. Nothing was invented; every word is in the
passage it names.

What failed is not support but *responsiveness*. The user asked how, and got
that it does. The grounding check verifies that a claim is supported by the
passage it cites; it does not verify that the claim answers the question, and no
deterministic lexical rule can — the two would need a judgement a word-overlap
test cannot make, and making it with a second model would move the guarantee
onto a dependency nothing in CI could check.

So the number is held at the one case that is known, named and explained, and it
may not grow: a second question reaching `grounded` fails the build whatever the
cause. Pinning it at 0 was available and was refused, because it would have
meant gating on a check that does not exist or relabelling u01 until the number
came out right. This is the boundary of what this project verifies, and it is
worth more stated than hidden behind a green zero.

### Gating it from both sides

Either side alone is passed by a broken service, and both directions are
measured rather than argued.

A ceiling on questions the corpus cannot answer is scored perfectly by a service
that refuses everything: raising the coverage floor until all 28 questions land
in `unsupported` scores **0** on `answered_unanswerable` — better than the real
service — and is caught only by the floor on `grounded_rate`. A floor on the
questions the corpus can answer is scored perfectly by a service that answers
everything. `eval/thresholds.yaml` is rejected at load time if it sets one
without the other.

That is the same argument the regression gate makes about aggregate floors and
per-question ceilings, arriving at the same shape from the other direction.

## Regression gate

`eval/thresholds.yaml` is the committed bar. `rag-contract gate` reruns both
evals and exits non-zero below it; CI runs that on every push and pull request.
The gate holds no numbers of its own, so lowering the bar is an edit to that
file and appears in the diff of the commit that does it.

It applies three independent kinds of check — the two below, plus the refusal
thresholds described under failure behaviour above.

**Aggregate floors** on recall@1, recall@5, recall@10 and MRR. Each sits below
the measured value with deliberate headroom: with 20 answerable questions one
question is worth 0.05 of a recall number, and a floor pinned to the measured
value would fail on the first legitimate change to chunk size. The floors absorb
two questions slipping; a third fails the build. recall@10 is the exception and
is pinned at 1.00 — top 10 is the eval window, so anything below that is not a
ranking regression but an annotated section going missing entirely.

**A rank ceiling per question**, which fails the build on its own whatever the
aggregate says. Aggregates hide compensating movement: if q16 improves from rank
4 to 1 while q02 collapses from 1 to 8, recall@1 is unchanged and MRR goes *up*,
and a question the corpus answers squarely has silently broken. Ceilings are the
measured rank plus two, per question rather than global, because the questions
are not equivalent — q16's rank 4 is a specific known confusion, and pinning it
to 4 would gate on that confusion never shifting.

The two are kept separate because either alone is blind. Aggregates miss one
question collapsing; ceilings miss uniform drift that stays inside every ceiling
while every question gets worse.

Every answerable question must carry a ceiling and no others may. Adding a
question without recording what it costs fails the gate, which forces the new
question into the same diff as its bar.

The eight unanswerable questions carry no rank ceiling and no score band. Their
score distribution overlaps the answerable one, as measured above, so a band
here would encode a boundary the data says does not exist. What holds them is
the `refusal` block, whose numbers are outcomes of the grounding check rather
than similarities.

CI declares no secrets and reads none — there is nothing to authenticate
against, because the vectors are committed and the eval is arithmetic. Two tests
assert that rather than asserting it in prose: one checks that the gate's import
graph reaches no HTTP client, and one runs the gate with every `*_API_KEY` in
the environment scrubbed. A third fails if `eval/results/retrieval.json` stops
matching what the eval produces, so the numbers quoted in this README cannot
drift from the commit they describe.

The gate has been observed failing as well as passing. A branch that raised
recall@1's floor to 0.80 and q16's ceiling to 2 turned the build red on both
checks, naming RFC 9110's definition of CONNECT as what outranks q16's expected
section. Tightening `max_answered_unanswerable` to 0 and `min_grounded_rate` to
0.95 turns it red on both refusal checks, naming u01. A gate that has only ever
been green is decoration.

## Running it

Python 3.12, `uv` for dependencies, numpy for retrieval.

```
uv sync
uv run rag-contract gate
```

That is what CI runs. `uv run rag-contract eval` prints the full JSON report
behind it, and `--quiet` reduces it to one line.

The eval needs no API key: the vectors are committed. Rebuilding the index does,
and is the only step that calls an external service:

```
cp .env.example .env    # then set OPENAI_API_KEY
uv run rag-contract build-index
```

## Design decisions

**No vector database.** At this corpus size, brute-force cosine similarity over
numpy is the correct choice and stays far below the latency ceiling. A vector
store would add a component to version and operate for no measurable benefit.

**Embedding vectors are committed.** The corpus is fixed, so the vectors are a
deterministic build artifact. Committing them means the retrieval eval runs in
CI with no network calls, no cost, and reproducible results.

**CI never calls a real LLM.** Retrieval evaluation is pure computation.
Answer-level evaluation replays the drafts committed under `eval/fixtures/`.
Exactly two commands call an external service — `build-index` and
`record-drafts` — both are run by hand, and both commit what they produce. One
API key covers both.

**A fixture freezes the model's output and nothing else.** Retrieval, the
grounding check and the state machine rerun on every replay against the live
index, so changing the coverage floor or the chunk parameters is measured by the
answer eval rather than frozen out of it. This paid for itself immediately: the
first recorded run scored 0.80 with 21 withdrawn claims, 20 of which were the
check wrongly rejecting claims that named their own section in their prose.
Fixing the check and re-measuring cost nothing.
