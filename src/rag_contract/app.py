"""The HTTP surface. Thin on purpose.

Every decision this service makes has already been made by the time a handler
runs: `service.py` acquires the index and produces an `Answer`, and `Answer`
already carries the state, the version it was built from, and the HTTP status
that state maps to. So these handlers translate and nothing else. A rule
expressed here rather than in `answering.py` would be a rule the eval harness
cannot measure, because the eval does not speak HTTP.

    GET /health                  what index is being served, and whether it is stale
    GET /questions               the fixed question set
    GET /answer/{question_id}    one answer, attributable to one index version

The index version is in the body *and* in an `X-Index-Version` header, because
the body is absent from exactly the case where attribution matters most — a
503 carrying no answer still has to say which index, or the absence of one,
produced it.

Every `/answer` response carries what the request spent against its deadline,
including the two that carry no answer. That is guarantee 5's "reported per
request", and reporting it on the failures is the part that matters: a request
abandoned for spending its budget has to say which stage spent it, or an
operator is left with a 503 and a guess.

Where the rate limit lives, and why it is the exception
------------------------------------------------------

Everything above says decisions do not belong here. The rate limit is the one
that does, and the reason is the same rule read from the other end: a rule
belongs in `answering.py` when the eval harness can measure it, and the eval
harness has no notion of a caller. It replays a fixed question set through
`decide()`; there is no client, no socket and no second client to be crowded
out by the first. The rate limit is a property of who is asking rather than of
what the corpus can support, so the HTTP surface is the only layer that can
see it, and it runs as middleware — before a handler, before an index is
acquired, before anything is spent.

Which is also why its response carries no `index_version` and no
`X-Index-Version`. A 429 is not an answer that failed; it is a request that was
never made, and naming an index on it would attribute something to a version
that never saw it.

What is not here
----------------

There is no endpoint that accepts free text, because the question set is fixed
and the drafts are committed; a question outside the set is a 404 rather than
something to improvise on. There is no endpoint that reloads or unloads the
index either. Cutover is an operational action, and an unauthenticated route
that swaps the index a public service is answering from would be a worse
liability than the feature is worth. `IndexRegistry` exposes it to the process
that owns the registry, and that is the whole audience for it.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from .budget import DeadlineExceeded
from .gate import load_thresholds
from .index import IndexError_
from .lifecycle import index_status
from .ratelimit import FORWARDED_FOR, RateLimiter, resolve_client
from .registry import IndexRegistry
from .service import Service, UnknownQuestion


def _startup_registry() -> IndexRegistry:
    """Load the committed index, or start without one.

    A missing or inconsistent index is not a reason to refuse to start. The
    service has a defined behaviour for having no index — `no_context`, 503 —
    and starting up into that state is more useful than crashing: `/health`
    then says what is wrong, which a container that exits on boot cannot.
    """
    registry = IndexRegistry()
    try:
        registry.reload(reason="startup")
    except IndexError_:
        pass
    return registry


def client_key(request: Request, trusted_proxy_hops: int) -> str:
    """Who is asking, translated off the wire and decided in `ratelimit.py`.

    The socket peer, or — when the deployment has committed a hop count —
    that many entries in from the right of `X-Forwarded-For`. Reading the
    header rather than the socket is only safe because of the direction:
    entries are appended by each proxy, so the rightmost N were written by the
    N machines in front of this process and everything to their left was
    written by the caller. `resolve_client` holds that argument and the
    fallbacks; this function does what the rest of the module does, which is
    take something out of an HTTP request and decide nothing.
    """
    peer = request.client.host if request.client else None
    return resolve_client(peer, request.headers.get(FORWARDED_FOR), trusted_proxy_hops)


def create_app(
    service: Service | None = None,
    limiter: RateLimiter | None = None,
    trusted_proxy_hops: int | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.service is None:
            app.state.service = Service.from_committed(
                _startup_registry(), load_thresholds().budget
            )
        yield

    app = FastAPI(
        title="rag-contract",
        summary="Question answering over a fixed corpus, with a measured contract.",
        lifespan=lifespan,
    )
    app.state.service = service
    # Both come from the committed budget block, and both are read here for the
    # same reason: the process that composes the service is where
    # `eval/thresholds.yaml` is read, so the gate and the runtime cannot
    # disagree about the allowance or about who it applies to.
    committed = (
        load_thresholds().budget
        if limiter is None or trusted_proxy_hops is None
        else None
    )
    app.state.limiter = limiter or committed.limiter()
    app.state.trusted_proxy_hops = (
        committed.trusted_proxy_hops
        if trusted_proxy_hops is None
        else trusted_proxy_hops
    )

    def current() -> Service:
        return app.state.service

    @app.middleware("http")
    async def rate_limit(request: Request, call_next):
        """Guarantee 5's third ceiling, applied before anything is served.

        Every route, including `/health`. Exempting it is the obvious kindness
        and the wrong one: measured, `/health` costs 7.6 ms — as much as the
        slowest question — because it re-reads and re-hashes the whole corpus
        to answer whether the index still describes it, and it is the one route
        that carries no deadline. An exemption there would be a hole shaped
        exactly like the cheapest way to take the process.

        The consequence is named rather than engineered around: a deployment's
        liveness probe is a client like any other here, so it belongs on a path
        this limiter does not see — the container directly, or a separate port
        — rather than on a public route carving an exemption anyone can use.
        """
        decision = app.state.limiter.check(
            client_key(request, app.state.trusted_proxy_hops)
        )
        if decision.allowed:
            return await call_next(request)
        return JSONResponse(
            content={
                "error": "rate limited",
                "detail": (
                    f"this client may make {app.state.limiter.requests_per_minute} "
                    f"requests a minute, up to {app.state.limiter.burst} at once. "
                    "The corpus is fixed and every answer is committed, so a "
                    "repeated question has a repeated answer: retry after "
                    f"{decision.retry_after_s}s."
                ),
                "limit": {
                    "requests_per_minute": app.state.limiter.requests_per_minute,
                    "burst": app.state.limiter.burst,
                },
                "retry_after_s": decision.retry_after_s,
            },
            status_code=429,
            headers={"Retry-After": str(decision.retry_after_s)},
        )

    @app.get("/health")
    def health() -> dict:
        """What is being served, and whether it still describes the corpus."""
        registry = current().registry
        index = registry.acquire()
        payload = registry.to_dict()
        payload["status"] = "serving" if index is not None else "no index"
        payload["index"] = None if index is None else index_status(index).to_dict()
        return payload

    @app.get("/questions")
    def questions() -> dict:
        """The fixed set. There is no other way to ask this service anything."""
        return {
            "questions": [
                {"id": q.id, "question": q.question, "answerable": q.answerable}
                for q in current().questions.values()
            ]
        }

    @app.get("/answer/{question_id}")
    def answer(question_id: str, response: Response) -> Response:
        del response
        try:
            answered = current().answer(question_id)
        except UnknownQuestion as exc:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"no question {question_id}. This service answers the fixed "
                    "set at /questions and takes no free-text input."
                ),
            ) from exc
        except DeadlineExceeded as exc:
            # Not an answer state, so not `answered.to_dict()`. The request was
            # abandoned mid-check, which means the service has no verdict on
            # the corpus to report — only what it spent getting nowhere. A
            # service error, like `no_context`, and a 503 for the same reason.
            version = current().registry.version
            return JSONResponse(
                content={
                    "error": "deadline exceeded",
                    "detail": exc.detail,
                    "index_version": version,
                    "budget": exc.spend.to_dict(),
                },
                status_code=503,
                headers={"X-Index-Version": version or "none"},
            )

        return JSONResponse(
            content=answered.to_dict(),
            status_code=answered.state.http_status,
            headers={"X-Index-Version": answered.index_version or "none"},
        )

    return app


def probe(
    application,
    path: str,
    *,
    client_host: str,
    count: int,
    headers: list[dict[str, str]] | None = None,
) -> list[tuple[int, dict[str, str]]]:
    """Send `count` GETs through an ASGI app from one client. No HTTP client.

    `headers`, when given, is one mapping per request, which is what lets the
    gate vary `X-Forwarded-For` across a run: the trusted-hop check is entirely
    about which requests share a bucket, and that cannot be driven from one
    fixed header set.

    This exists for the gate. The regression worth catching about a rate limit
    is not that the arithmetic in `ratelimit.py` is wrong — the tests hold that
    — but that a correct limiter stops being *installed*, which is how rate
    limits actually die. Catching it means going through the surface rather
    than asking the limiter directly.

    It cannot use a test client to do that: `starlette.testclient` imports
    httpx, and `tests/test_ci_contract.py` holds the gate's import graph clean
    of every HTTP client so that CI is provably never one call away from a real
    API. So this calls the ASGI application the way a server would, which is
    all a test client does underneath.
    """

    per_request = headers or [{} for _ in range(count)]

    async def send_one(extra: dict[str, str]) -> tuple[int, dict[str, str]]:
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [(b"host", b"gate")]
            + [(k.lower().encode(), v.encode()) for k, v in extra.items()],
            "client": (client_host, 50000),
            "server": ("gate", 80),
        }

        async def receive() -> dict:
            return {"type": "http.request", "body": b"", "more_body": False}

        captured: dict = {}

        async def send(message: dict) -> None:
            if message["type"] == "http.response.start":
                captured["status"] = message["status"]
                captured["headers"] = {
                    k.decode().lower(): v.decode() for k, v in message["headers"]
                }

        await application(scope, receive, send)
        return captured.get("status", 0), captured.get("headers", {})

    async def all_of_them() -> list[tuple[int, dict[str, str]]]:
        return [await send_one(extra) for extra in per_request]

    return asyncio.run(all_of_them())


app = create_app()
