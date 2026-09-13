# The service, and nothing that is not the service.
#
# Every path copied below is one the running process actually reads: the
# package, the templates, the committed vectors, the corpus `/health` re-hashes
# to say whether those vectors still describe it, and the question set,
# thresholds and drafts that make an answer reproducible without a key. Nothing
# else is copied — not the tests, not `eval/results/`, and not `.env`, which
# the running service has no use for because the served path spends nothing.
#
# Copied by name rather than with `COPY . .` plus exclusions. The two are
# equivalent until someone adds a file, and then one of them ships it.

FROM python:3.12-slim-bookworm AS build

# Pinned, and the same uv the lockfile was resolved with. Both stages use the
# image's own interpreter: a virtualenv built against uv's standalone Python
# would point at a binary the runtime stage does not have.
COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Dependencies from the lockfile alone, before any source, so editing the
# service does not re-resolve or re-download anything.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev


FROM python:3.12-slim-bookworm

# Nothing here runs as root. The process reads committed artefacts and writes
# nothing at all — there is no ledger on this path, because the two commands
# that spend money are not in this image.
RUN useradd --create-home --uid 10001 service
WORKDIR /app

COPY --from=build --chown=service:service /app/.venv /app/.venv
COPY --chown=service:service src/ src/
COPY --chown=service:service templates/ templates/
COPY --chown=service:service corpus/ corpus/
COPY --chown=service:service index/ index/
COPY --chown=service:service eval/questions.yaml eval/thresholds.yaml eval/
COPY --chown=service:service eval/fixtures/drafts/ eval/fixtures/drafts/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PORT=8000

USER service
EXPOSE 8000

# No HEALTHCHECK. `/health` is the most expensive route on the surface — it
# re-reads and re-hashes the whole corpus — it carries no deadline, and it is
# covered by the per-client rate limit like every other route. A probe hitting
# it every few seconds is a client spending the allowance, so liveness is
# checked by connecting to the port instead, which the platform does without
# entering this process. `fly.toml` carries that decision and the reasoning.
CMD ["rag-contract", "serve", "--host", "0.0.0.0"]
