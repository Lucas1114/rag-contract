# rag-contract

A question-answering service over a small, fixed corpus.

**Live:** <https://rag-contract.fly.dev/> — the fixed question set, and per
question the state it landed in, the claims that survived the grounding check
with their citations, the ones that were withdrawn with the rule that withdrew
them, what the request spent against its deadline, and the index version it was
answered from.

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
| 4 | Every answer is attributable to an index version | Index versions are content-addressed and returned in response metadata. A corpus change fails the build until the index is rebuilt, and a cutover does not drop requests in flight or move them onto the new index. |
| 5 | Latency, cost and request rate are ceilings, not advice | A served request runs under a deadline and is abandoned rather than answered when it spends it. The two commands that call an API ask a committed daily cap before every call and stop rather than break it. A client asking past its committed allowance is refused with a 429 and a `Retry-After` rather than served slowly. All three ceilings are in `eval/thresholds.yaml` and all three are checked by the gate. |

## Status

This table is updated at each stage commit. `specified` means defined but not
yet implemented, `implemented` means the code exists, `verified` means eval
data or CI results support it.

| Part | Status |
|------|--------|
| 1. Evaluation harness | verified |
| 2. CI regression gate | verified |
| 3. Failure behaviour | verified |
| 4. Index lifecycle | verified |
| 5. Budgets | verified |

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
    rag-contract index-status   does the index still match the corpus? no network
    rag-contract build-index    embed and write the index              calls the API once
    rag-contract record-drafts  record the answer model's claims       calls the API once
    rag-contract eval           score against eval/questions.yaml      no network
    rag-contract eval-answers   score the failure behaviour            no network
    rag-contract answer         answer one question, showing its state no network
    rag-contract gate           hold both evals, index freshness, the budgets
                                and the served rate limit to
                                eval/thresholds.yaml                   no network

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
always returns ten, so that state belongs to an unavailable index rather than to
any question. Part 4 is where it becomes reachable.

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

## Index lifecycle

### The version is an address, not a label

    sha256(corpus fingerprint + embedding model + chunk parameters)[:12]

Every input that could change a retrieval result is in that hash. Two indexes
with the same version hold the same vectors over the same text, and an index
whose version no longer matches its inputs is detectable by arithmetic rather
than by convention. `0fc1763d6701` is in `index/meta.json`, in the response
metadata of every answer, in the `X-Index-Version` header, and in
`eval/thresholds.yaml` as the provenance of every number there.

The question set is fingerprinted separately and deliberately left out of that
address. Chunk vectors do not depend on `eval/questions.yaml`; the committed
*question* vectors do. Two artefacts with two reasons to rebuild, and only one
of them changes what the index is called. Folding them together would mean
either a version that moves when nothing about the corpus did, or a stale
question vector that no hash catches.

### A corpus change fails the build

CI has no key, so it cannot rebuild an index. What it can do is refuse to pass a
commit that needed one. `rag-contract index-status` recomputes the address from
what is on disk, and the same check runs inside the gate on every build.

This was verified by doing it. Appending a line to `corpus/rfc8259.txt` and
updating its hash in `corpus/manifest.yaml` turns the build red on exactly one
check:

    FAIL  index freshness    0fc1763d6701  limit 0cadbaf40305
          corpus: index has da38c386b172, disk has 08d885db1dfd

Every other check stays green, and that is the point. The vectors still score
the same questions to the same recall numbers, because they are vectors of a
corpus that is no longer in the repository. Nothing else in the gate can see
that, and the eval reports quality about a corpus that has been edited out from
under it. The check names both what changed and the version the rebuild will
produce, because the person who reads it has to run `build-index` themselves.

Editing a corpus file *without* updating the manifest does not reach this check
at all — `corpus.py` verifies every file against its recorded sha256 at load
time and refuses to go further.

The freshness check holds no threshold and is not in `eval/thresholds.yaml`.
There is no bar to set. Either the committed vectors were built from this
corpus or they were not, and a project that could choose to tolerate "not"
would be quoting eval numbers about a corpus it no longer has.

### Cutover, and why it is almost no code

A registry holds one reference to an `Index`. A request acquires that reference
once, at entry, and uses the object it got for the rest of its work. A cutover
rebinds the registry's reference. A request that acquired before the swap is
already holding the old index, finishes against it, and reports its version;
the old object is collected when the last request holding it returns.

There is no drain, no quiesce, no request counter, no reader-writer lock and no
grace period, and the reason is worth stating rather than hiding. Rebinding a
name cannot produce a half-swapped object, so a reader sees either the whole
old index or the whole new one. What makes that sufficient is that `Index` is
frozen and holds the question vectors, the chunk vectors and the chunks
together: acquiring it once acquires all of them at once. The failure this
design rules out is not a torn read but an answer assembled from a query vector
in one index and chunks in another — attributable to neither version, and
reported under whichever one happened to be current when the response was
written.

So the guarantee is expressed as the absence of a second acquire rather than as
machinery, which is only convincing if the absence is tested. The tests run real
threads and hold a real request open across a real swap, in both windows that
exist: between acquiring the index and first touching it, and mid-draft. Both
were checked against the bug they exist to catch — reading the version from the
registry instead of from the acquired index turns them red, and so does
re-reading the registry for chunks after the query vector came from the
acquired one. The same test then runs over HTTP, with a real request open when
the swap lands, completing with the old version in the body and the header.

A reload that fails leaves the service on the last index known to be good: the
load happens before the swap, so a rebuild that wrote a broken index raises out
of `load_index` and the registry never rebinds.

### `no_context` stops being hypothetical

Part 3 defined four states and could only exercise three. `no_context` is what
the service does when nothing was retrieved, and no question can reach it
against this corpus — brute-force cosine over 868 chunks always returns ten.
`eval/questions.yaml` says so rather than manufacturing a question that
pretends otherwise.

It was never a property of a question. An unavailable index is what reaches it,
and the registry is where an index becomes unavailable: `acquire()` returns
nothing, the request retrieves nothing, and the response is a 503 carrying no
version because there is no index to name. That is a service error rather than
a refusal, and the difference is visible in the response — a refusal is a 200
that names the passages it consulted and the claims it rejected, while this one
consulted nothing and rejected nothing.

A missing index on startup lands in the same state instead of killing the
process. `/health` then says what is wrong, which a container that exits on
boot cannot.

## The HTTP surface

    GET /health                  which index is being served, and whether it is stale
    GET /questions               the fixed question set
    GET /answer/{question_id}    one answer, attributable to one index version
    GET /                        the same question set, for a person
    GET /q/{question_id}         the same answer, for a person

`/answer` returns the state, the claims that survived grounding with their
citations, the ones that were withdrawn with the rule that dropped them, the
passages consulted, and `index_version`. The status code is the state's:
`grounded`, `partial` and `unsupported` are all 200, because a refusal is
something the service decided; `no_context` is 503, because nothing was
consulted. `X-Index-Version` carries the version independently of the body,
which matters precisely where the body has no answer in it.

Every `/answer` response also carries `budget` — the deadline it was held to,
what it spent, and the breakdown by stage. Reporting it on the responses that
carry no answer is the part that matters: a request abandoned for spending its
budget has to say which stage spent it, or an operator is left with a 503 and a
guess. That response is not an answer state at all. It carries `error`,
`detail`, `index_version` and `budget`, and no `state` — the grounding check did
not finish, so the service has no verdict about the corpus to report.

A question id outside the fixed set is a 404. There is no free-text endpoint,
and that constraint is what lets the answer step replay a committed draft
rather than calling a model on the request path.

A client past its committed allowance gets a 429 with a `Retry-After` and no
answer at all. That refusal is the one rule on this surface decided by the
surface, and it is the same rule as everything else here read from the other
end: a rule belongs in `answering.py` when the eval harness can measure it, and
the harness has no notion of a caller — it replays a fixed question set, with no
client and no second client to be crowded out by the first. So the limit runs as
middleware, before a handler and before an index is acquired, and it applies to
every route. Its response carries no `index_version` and no `X-Index-Version`,
because a 429 is not an answer that failed but a request that was never made,
and naming a version on it would attribute a refusal to vectors that never saw
it.

There is no endpoint that reloads or unloads the index. Cutover is an
operational action; an unauthenticated route that swaps the index a public
service answers from would be a worse liability than the feature is worth.

The handlers translate and decide nothing — the state, the status code and the
version are all settled before one runs. A rule expressed in a handler would be
a rule the eval harness cannot measure, because the eval does not speak HTTP.

### The two HTML routes

`/` and `/q/{id}` render what `/questions` and `/answer/{id}` return, from the
same `Service.answer`, under the same deadline and the same allowance. They are
not a second implementation of anything and they decide nothing the JSON routes
do not.

They exist because the failure states are the substance of this service and a
JSON body does not show them to a person. A claim the check withdrew, printed
next to the rule that withdrew it and the number that failed, is guarantee 3 on
one screen — a refusal you can read rather than one you have to take on trust.
The page also carries the state's status code, so a rendered refusal is still a
refusal: a 200 on `no_context` would make the surface disagree with itself
depending on who was reading it.

No page loads anything from another host — no font, no script, no stylesheet.
A test asserts it. The availability of a page that fetches from a CDN is
someone else's, which is an odd thing to depend on in a service whose entire
subject is claims it can hold itself.

## Budgets

### What a latency ceiling here can honestly protect

The obvious answer is retrieval, and measuring says it is wrong. Per stage,
over the served request path:

| stage | median | worst | varies with |
|-------|--------|-------|-------------|
| retrieval | 0.026 ms | 0.026 ms | nothing — flat across all 28 questions |
| collapsing hits to passages | 0.004 ms | 0.005 ms | nothing |
| grounding check | 0.162 ms | **7.400 ms** | claims x cited passage size |

Retrieval is one matrix-vector product over 868 chunks and there is no question
this corpus makes slow. The answer step replays a committed fixture and is a
dictionary lookup. The request is spent almost entirely in the grounding check
— the one part of the pipeline this project added — because `check_claim`
tokenises the whole cited passage once per claim, twice over.

The corpus bounds one side of that: ten retrieved sections, 13k characters at
worst. Nothing bounds the other. **The number of claims comes from the model**,
not from the corpus, the question set or the request, and it is the only input
on the request path the service does not control. That is what the ceiling is
for. It is not a slow path invented so that there would be something to guard;
it is the one place where an input the service does not own multiplies work the
service does.

### Exceeding it abandons the request

The tempting behaviour is to stop checking and answer from the claims checked so
far, and it is exactly wrong — for guarantee 3's reasons rather than guarantee
5's. Claims the check never reached are not claims that passed. An answer
assembled from a truncated grounding check would report `grounded` or `partial`
on the strength of claims nobody verified, which is the precise failure the
grounding check exists to prevent. A partially applied check is not a weaker
check, it is an unsound one.

So the deadline is checked *inside* the per-claim loop and raises out of it.
There is no half-checked verdict list for a later stage to be tempted by, and
the response says the request was abandoned rather than answered.

It is deliberately not a member of `AnswerState`. `decide` never returns it,
because reaching it means `decide` never finished — the service has no verdict
about the corpus to report, only what it spent getting nowhere. A service error
like `no_context`, and a 503 for the same reason.

    request_deadline_ms: 150.0   what the service enforces per request
    max_request_ms:       50.0   the bar the gate holds the measured request to

Two numbers, because one would not do. The gate sees the index every build is
made against and fails before a deployment where the ceiling would fire; it
cannot see an index installed at runtime through `registry.install`, which the
cutover machinery can be handed anything. The per-request deadline is the
backstop for exactly that, and is the only one of the two a served request ever
reaches. `budget.py` refuses the threshold file when the gate's bar is not at
most half the deadline — a bar at the deadline is the deadline with extra steps,
and would leave the build green on the last commit before requests start being
abandoned.

Nothing served today comes close to either. The tests therefore drive an
injected clock rather than the workload: making the deadline fire by sleeping
would test `time.sleep`, and making it fire with real work would mean adding a
slow path to the service to have something to catch.

### The cost ceiling has one real consumer

Nothing on the request path spends money — committed vectors, committed drafts,
no key — so a token ceiling there would be a ceiling on zero. The spend is in
the two commands that are run by hand, and it is small and now known:

| command | priced from committed artefacts | actually billed |
|---------|-------------------------------|-----------------|
| `build-index` | $0.0059 — 293,131 tokens at $0.02/1M | — |
| `record-drafts` | $0.6621 floor | **$0.57**, 29 requests, 2026-09-12 |

The cap is asked *before* every call and refuses the ones that would break it,
because a cap that notices afterwards is a report. That needs an upper bound on
a call before making it, which sounds impossible for a completion and is not:
embeddings cost a function of text already on disk, and a completion is bounded
because the request itself sets `max_completion_tokens`, which reasoning tokens
count against. The ledger then records reported usage, so the running total is
real spend rather than accumulated worst cases.

`.env.example` carried `DAILY_SPEND_CAP_USD=2.00` and nothing read it. Wiring it
up where it stood was the obvious fix and the wrong one — a cap set by whoever
runs the command is invisible in review and different on every machine. It is
`budget.daily_cap_usd` in `eval/thresholds.yaml` now, with no environment
override, so raising it is a diff like every other bar here.

### What pricing the commands found

`drafter.MAX_TOKENS` was 4000, a generous default with no reason behind it.
Priced: 28 questions at 4000 output tokens is $3.36 at $30/1M, $3.90 with input,
against a $2.00 cap. **The cap would have refused to authorise the run it exists
to permit.** Two numbers in two files, set by different people for different
reasons, that nothing had ever compared.

It is 1500 now, which brings the worst case to $1.80. The size comes from
backing the output side out of the invoice rather than from guessing: the run
billed $0.57 in total and 75,250 input tokens, so at $5/1M input and $30/1M
output the output side is about 6,500 tokens across 29 requests — roughly 220 a
question, reasoning tokens included, since those are billed as output and count
against this same ceiling. A ceiling of 1500 is nearly seven times that. Cutting it is safe because truncation here
is loud rather than silent: `LiveDrafter` refuses any response that did not
finish on `stop`, so a draft needing more room fails the command instead of
committing a half-written fixture. `MAX_TOKENS` stays out of the prompt
fingerprint, because it cannot change what was asked, only truncate — so the
committed drafts did not have to be bought again to lower it.

The token estimator is left deliberately over-estimating, and this is where that
got checked rather than asserted: it scores 108,693 input tokens for a run the
invoice billed at 75,250, a factor of 1.44. Every priced figure above is
therefore conservative, which is the only safe direction for a cap — one built
on an optimistic estimate authorises the call that breaks it.

### The third ceiling protects the first one

A rate limit is the last of the three and the one whose justification had to be
found rather than borrowed. The usual argument is cost, and there is none here:
a served request uses committed vectors and a committed draft, holds no key and
spends nothing. The second usual argument is per-request work, and that is what
the deadline above already does. So the question was what is left, and measuring
answered it more sharply than "availability" would have.

One process, one client per thread, all of them asking for `/answer/q14`:

| clients | req/s | observed p99 | service reported (median) | abandoned |
|---------|-------|--------------|---------------------------|-----------|
| 1 | 131 | 8.6 ms | 7.0 ms | 0 |
| 4 | 134 | 44.7 ms | 8.5 ms | 0 |
| 8 | 134 | 82.2 ms | 14.9 ms | 0 |
| 16 | 134 | 179.4 ms | 27.2 ms | 0 |
| 32 | 134 | 297.2 ms | 39.2 ms | 0 |
| 64 | 139 | 481.2 ms | 57.2 ms | **10.2%** |

Three things are in that table.

**Throughput is flat.** 131 requests a second at one client and 139 at
sixty-four. The answer path is synchronous CPU work under one interpreter lock,
so concurrency buys nothing — a process serves about 130 requests a second and
extra callers only add queue.

**The deadline cannot see the queue.** At 32 clients a caller waits 297 ms and
the service reports having spent 39. `Spend` starts when `Service.answer`
begins, which is after the request was accepted, parsed and handed to a worker
thread. Guarantee 5's ceiling is a ceiling on *service* time, and until this
measurement nothing in the repository bounded the time a client actually waits.

**And past a point the deadline fires on the wrong requests.** At 64 clients one
request in ten is abandoned with a 503, and the spend reports on them look like
this:

    "stages": {"retrieval": 124.957, "drafting": 0.001, "grounding": 28.462}

125 ms in retrieval — a stage that does 0.026 ms of work. Nothing was retrieved
slowly. The thread holding that request was descheduled and the wall clock kept
running; the request did nothing unusual and was abandoned anyway, because
someone else was loud.

That is the argument, and it is not the one this section set out to make.
Without a limit on how fast one client may ask, **the mechanism that holds
guarantee 5 becomes a way to deny service**: a single caller can push the
process into contention until other callers' requests are abandoned for
spending a budget they never spent on work. The rate limit is what keeps the
deadline pointed at the thing it was built to catch — an unbounded number of
claims — rather than at whoever happened to be sharing the process.

    max_requests_per_minute: 60   how fast one client may ask
    max_client_burst:        10   how many of those it may spend at once

Two numbers, because a rate alone does not describe the failure above: 60 a
minute permits 60 at once, and simultaneous is exactly what produced the
abandoned requests. The depth is what bounds that, and 10 sits well inside the
region where the table shows nothing being abandoned.

The rate is bounded from both directions, which is what makes it a number rather
than a preference. From above by arithmetic: a client allowed R requests a
minute, each entitled to the 150 ms deadline, can demand R x 150 ms of a
process-minute, and `budget.py` refuses the threshold file if that exceeds a
quarter of one — so no allowance above 100 a minute can be committed while the
deadline stands. Lengthening the deadline tightens the allowance automatically,
because the two are one statement. From below by a workload that cannot grow:
the corpus is fixed, the drafts are committed, and the answer to q14 today is
the answer to q14 tomorrow byte for byte, so 60 a minute is the entire
28-question set twice over, every minute, and no honest caller has a reason to
ask again.

### Refusing, not waiting — and what this does not cover

`embedding.py` already contained a rate limiter and it is deliberately not
reused. It solves the opposite problem from the opposite side: it schedules
*our* requests against a vendor's published allowance, and sleeps when one does
not fit. Blocking is right there — one caller, a batch job, waiting cheaper than
failing. Every one of those properties is inverted on a served request, and a
limiter that made a client wait would be holding a worker thread open for
exactly the caller it has decided is asking for too much, which is the
contention above with extra steps. This one refuses immediately and says when to
come back.

### Who the allowance applies to

An allowance is half of a per-client limit. The other half is what "client"
means, and `trusted_proxy_hops` in `eval/thresholds.yaml` is that half:
committed to the same reviewed file as the numbers, because an allowance whose
subject comes from a deployment's environment is a ceiling nobody reviewed.

A client is the socket peer, or — where the deployment declares proxies in
front of it — that many entries in from the **right** of `X-Forwarded-For`. The
direction is the entire security property. Each proxy appends the peer it saw,
so the last N entries are the testimony of the N machines the deployment
trusts, and everything to their left was typed by whoever was calling. Reading
the leftmost entry, which is the usual mistake and reads like "the original
client", lets any caller mint a fresh identity per request and walk past the
limiter by setting a header.

Counting from the right also fails safe. Declare more hops than the chain has
and the header can never satisfy them, so the request falls back to the socket
peer: every caller collapses into one bucket, which refuses too much rather
than too little. A caller can only add entries on the left, so there is no
header that makes the list long enough to be believed.

This was 0 through P06, which was correct while nothing was deployed and became
wrong the moment something was: behind a proxy every request arrives from the
proxy, so all visitors would have shared one bucket and the second visitor to
the public deployment would have been refused. It is 1 now, for a chain with a
single Fly edge in it — that proxy appends the peer it saw, so the rightmost
entry is the visitor — and the gate holds the surface to it from both sides:
one caller varying the forgeable entries must still be refused past the burst,
and requests differing only at the trusted hop must not be. What no gate can
check is that the committed count matches the real chain, which is the reason
the fail-safe direction matters and the reason the number is read in a diff.

The limit covers every route, `/health` included. Exempting it is the obvious
kindness and the wrong one: `/health` measures 7.6 ms — as much as the slowest
question — because it re-reads and re-hashes the whole corpus to say whether the
index still describes it, and it is the only route with no deadline. An
exemption there would be a hole shaped exactly like the cheapest way to take the
process. The consequence is named rather than engineered around: a liveness
probe is a client like any other here, so it belongs on a path this limiter does
not see rather than on a public route carving an exemption anyone can use.
Measuring also falsified a claim already in the repository — `lifecycle.py` said
`index_status` was "cheap enough to run on every health check" — and the
docstring now says what it costs and why it is still not cached.

The deployment takes that branch rather than reopening it. `fly.toml` declares
a TCP check and no HTTP check anywhere, so the platform probes the port and
never enters the process: an HTTP probe every ten seconds would be a client
spending 6 of the allowance a minute forever, and the visitor it eventually
crowded out would be refused for someone else's monitoring. What a TCP check
gives up is real — it proves the process is listening, not that it can answer.
That is the right trade here because the failure it cannot see is one this
service already handles deliberately: a process with no usable index starts
anyway and serves `no_context` with a 503 so that `/health` can say what is
wrong, and restarting it would not produce an index. Only a commit does.

What a per-client limit cannot do is bound many clients rather than one. That
needs a global concurrency limit or something upstream of the process, and
neither is in this repository. Guarantee 5 is about ceilings this service
enforces on itself, and the edge of that is worth stating rather than implying.

Deploying it found a sharper version of the same edge. The bucket lives in the
process, so *N* machines behind the edge is *N* times the committed allowance —
and the first deploy created a high-availability pair, which is Fly's default.
26 requests from one visitor came back refused from the twentieth rather than
the eleventh, interleaved, because the edge was alternating between two buckets
that could not see each other: the file said 60 a minute and 10 at once, and a
visitor met 120 and 20. So the app runs a single machine, and `fly.toml`
records that as a decision about this guarantee rather than about cost. The
number in the committed file is the number a visitor actually meets, and
nothing in CI can hold that — only the deployment config can.

## Regression gate

`eval/thresholds.yaml` is the committed bar. `rag-contract gate` reruns both
evals and exits non-zero below it; CI runs that on every push and pull request.
The gate holds no numbers of its own, so lowering the bar is an edit to that
file and appears in the diff of the commit that does it.

It applies five independent kinds of check — the two below, plus the refusal
thresholds described under failure behaviour above, the index freshness check
described under index lifecycle, and the budgets described above. Thirty-four
checks in total on this commit, thirty-six with `--check-report`.

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
while every question gets worse. Both are blind to the index having stopped
describing the corpus, which is why the freshness check is there and holds no
number at all.

Every answerable question must carry a ceiling and no others may. Adding a
question without recording what it costs fails the gate, which forces the new
question into the same diff as its bar.

**Five budget checks**, described above: the slowest measured request against
the bar under the served deadline, the priced cost of a rebuild and of a
re-record against their ceilings, the worst-case drafting run against the day's
cap, and the rate limit.

The last one is shaped like the freshness check rather than like the other four,
because it holds no threshold. The allowance is already held at load time —
`budget.py` refuses a file whose rate lets one client demand more than a quarter
of a process-minute — so comparing it again here would hold nothing new. What is
held nowhere else is that the limiter is still *installed*, which is how rate
limits actually die: nobody edits them to zero, they get lifted out of a
middleware stack during unrelated work and no test notices because every test
sends one request. So the gate drives the real ASGI application, from one
client, on a frozen clock, exactly one request past the committed burst. The
frozen clock is what makes a wall-clock control deterministic enough to gate on:
nothing refills, so the allowance is spent in a fixed number of requests and the
check produces the same result on any machine. It does not use a test client to
do that — `starlette.testclient` imports httpx, and a test asserts the gate's
import graph reaches no HTTP client — so it calls the application the way a
server would, which is all a test client does underneath.

The latency one is the first check in this gate that is not deterministic, and
that is worth stating rather than burying. Every other number here is numpy over
committed vectors and comes out identical on any machine, which is the property
the whole project rests on; a wall clock does not. So its bar is sized for an
order of magnitude — 7.4 ms measured, perhaps three times that on a loaded
runner, against a 50 ms bar — and not for a measurement. It earns its place
because the regression it catches is also an order of magnitude: the request
path acquiring work that scales with something nothing bounds. The two priced
checks are arithmetic over committed text and cost nothing to run; a further
test asserts the gate leaves the spend ledger untouched, since it prices an API
call and holds the cap that refuses one.

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
0.95 turns it red on both refusal checks, naming u01. Editing a line into
`corpus/rfc8259.txt` turns it red on freshness alone, with every quality check
still green. Restoring `MAX_TOKENS` to 4000 turns it red on worst-case drafting
alone, at $3.9035 against the $2.00 cap; tightening the latency bar to 1 ms
names q14; tightening the rebuild ceiling below the corpus names `build-index`.
Lifting the rate limiter out of the middleware stack turns it red on the rate
limit check alone, at eleven requests from one client and none refused, and so
does a 429 that carries no `Retry-After`; raising the allowance to 400 a minute
is refused at load instead, naming the whole process-minute one client would
then be entitled to. A gate that has only ever been green is decoration.

## Running it

Python 3.12, `uv` for dependencies, numpy for retrieval.

```
uv sync
uv run rag-contract gate
```

That is what CI runs. `uv run rag-contract eval` prints the full JSON report
behind it, and `--quiet` reduces it to one line.

Serving it:

```
uv run rag-contract serve
curl -i localhost:8000/answer/q04
```

Then open <http://localhost:8000/> for the same thing as pages: the question
set, and per question the state it landed in, the claims that survived the
grounding check with their citations, the ones that were withdrawn with the
rule that withdrew them, what the request spent against its deadline, and the
index version it was answered from.

One entry point rather than an ASGI server plus an import path, so a
container, a platform and a developer all start the same process. `--host` and
`--port` override the defaults, and `PORT` is read from the environment because
a platform assigns the socket — it is the only thing about this process the
environment decides, and it is not behaviour. Everything that is behaviour is
committed to `eval/thresholds.yaml`.

No key, no network: the answer replays a committed draft and reruns retrieval,
the grounding check and the state machine against the committed index.

That server enforces the committed allowance per client — 60 requests a minute,
10 at once — so a loop over `/questions` fast enough to notice gets a 429 and a
`Retry-After` rather than a slower answer. Run this way the client is the
socket peer; the committed `trusted_proxy_hops` is what makes it the visitor
rather than the proxy when the same process runs behind one.

The eval needs no API key: the vectors are committed. Rebuilding the index does,
and is one of exactly two steps that call an external service:

```
cp .env.example .env    # then set OPENAI_API_KEY
uv run rag-contract build-index
```

Both of those steps run under the daily spend cap in `eval/thresholds.yaml`.
The cap is asked before every call and refuses the ones that would break it, so
a command that runs out of budget stops with nothing written rather than
finishing and reporting what it cost. Spend is recorded to a local, gitignored
ledger; nothing about it is committed, because it is machine state rather than a
property of the repository.

### Deploying it

```
docker build -t rag-contract .
docker run --rm -p 8000:8000 rag-contract
```

The image copies the package, the templates, the corpus, the committed vectors
and the question set, thresholds and drafts — by name, not with `COPY . .` and
a list of exclusions, because the two are equivalent right up until someone
adds a file. It runs as a non-root user, needs no key and no volume, writes
nothing, and starts in about 0.3 s into 58 MB resident — measured in the
container, against the 256 MB that is the smallest machine Fly offers.

There is no `HEALTHCHECK` in the image and no HTTP check in `fly.toml`, for the
reason under Budgets above: the probe belongs on a path the rate limiter cannot
see, so the platform connects to the port instead.

The one thing the environment decides is `PORT`. Everything that governs
behaviour — the deadline, the day's cap, the allowance, who a client is — is in
`eval/thresholds.yaml`, so a deployment can move the socket and nothing else.

On Fly, from `fly.toml`:

```
fly deploy --ha=false
```

`--ha=false` is not optional here. The rate limiter holds one bucket per
process, so the default high-availability pair would double the committed
allowance — see Budgets above for the measurement that says so.

## Design decisions

**No vector database.** At this corpus size, brute-force cosine similarity over
numpy is the correct choice, and this is now measured rather than assumed: 0.026
ms per query, flat across all 28 questions, against a 150 ms request deadline. A
vector store would add a component to version and operate for no measurable
benefit. The measurement also settled where the request's time actually goes,
which was not where this section used to imply — see Budgets.

**Embedding vectors are committed.** The corpus is fixed, so the vectors are a
deterministic build artifact. Committing them means the retrieval eval runs in
CI with no network calls, no cost, and reproducible results.

**CI never calls a real LLM.** Retrieval evaluation is pure computation.
Answer-level evaluation replays the drafts committed under `eval/fixtures/`.
Exactly two commands call an external service — `build-index` and
`record-drafts` — both are run by hand, and both commit what they produce. One
API key covers both, and one committed daily cap refuses either of them a call
it cannot afford.

**A fixture freezes the model's output and nothing else.** Retrieval, the
grounding check and the state machine rerun on every replay against the live
index, so changing the coverage floor or the chunk parameters is measured by the
answer eval rather than frozen out of it. This paid for itself immediately: the
first recorded run scored 0.80 with 21 withdrawn claims, 20 of which were the
check wrongly rejecting claims that named their own section in their prose.
Fixing the check and re-measuring cost nothing.
