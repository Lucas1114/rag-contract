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

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse

from .index import IndexError_
from .lifecycle import index_status
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


def create_app(service: Service | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.service is None:
            app.state.service = Service.from_committed(_startup_registry())
        yield

    app = FastAPI(
        title="rag-contract",
        summary="Question answering over a fixed corpus, with a measured contract.",
        lifespan=lifespan,
    )
    app.state.service = service

    def current() -> Service:
        return app.state.service

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

        del response
        return JSONResponse(
            content=answered.to_dict(),
            status_code=answered.state.http_status,
            headers={"X-Index-Version": answered.index_version or "none"},
        )

    return app


app = create_app()
