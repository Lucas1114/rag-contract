"""The one network-calling module in the project.

Embeddings are computed once, by `build-index`, and the vectors are committed.
Nothing else — not the eval, not CI — calls this. Keeping the dependency in one
module makes that claim checkable by import graph rather than asserted in prose.

`httpx` is imported inside the function that uses it rather than at module
scope, the way `drafter.py` does it. The model name and dimension count live
here, and the index lifecycle has to read them to say whether the committed
index still describes the corpus — so this module has to be importable from the
gate, and importing it must not put an HTTP client in that graph.

Spend is capped here rather than reported. `embed_texts` takes a `Cap` and asks
it before every batch whether that batch fits inside what is left of the day,
so a corpus that grew past the budget stops the build with no index written
instead of producing one and an invoice. Embeddings are the easy half of
guarantee 5's cost side: there is no output to bound, so a batch's cost is a
function of text that is already on disk.

The build has to survive whatever rate limit the account it runs under happens
to have, because the point of committing vectors is that anyone can rebuild
them. Rather than hard-coding one tier's allowances, the client schedules
against generous defaults and treats a 429 as authoritative, honouring the
Retry-After the server sends back.
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .spend import Cap, estimate_tokens, price_for

if TYPE_CHECKING:  # pragma: no cover - annotations only
    import httpx

MODEL = "text-embedding-3-small"
DIMENSIONS = 1536
ENDPOINT = "https://api.openai.com/v1/embeddings"

# Hard per-request limits: the API accepts at most 2048 inputs, and batches are
# kept well under that so a failed request is cheap to retry.
MAX_INPUTS_PER_REQUEST = 256
MAX_TOKENS_PER_REQUEST = 100_000

# Per-minute allowances the client schedules against. Deliberately generous:
# the corpus is around 170k tokens, so a build takes a couple of minutes on any
# account that is not throttled, and a throttled one falls back on the 429 path
# below rather than on a number guessed here.
REQUESTS_PER_MINUTE = 60
TOKENS_PER_MINUTE = 150_000

# Retry budget for a 429. The server's Retry-After wins when it sends one.
MAX_ATTEMPTS = 8
BACKOFF_SECONDS = 15
MAX_BACKOFF_SECONDS = 120

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
_KEY_NAME = "OPENAI_API_KEY"


class EmbeddingError(RuntimeError):
    """The embedding API could not be reached or answered unusably."""


def _api_key() -> str:
    key = os.environ.get(_KEY_NAME, "").strip()
    if not key and _ENV_PATH.is_file():
        for line in _ENV_PATH.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == _KEY_NAME:
                key = value.strip()
                break
    if not key:
        raise EmbeddingError(
            f"{_KEY_NAME} is not set. Embeddings are built once and committed; "
            "the eval runs off those committed vectors and needs no key."
        )
    return key


class _RateLimiter:
    """Client-side throttle for the request and token allowances."""

    def __init__(self, requests_per_minute: int, tokens_per_minute: int) -> None:
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self._recent: deque[tuple[float, int]] = deque()

    def acquire(self, tokens: int) -> None:
        """Block until a request of this size fits inside both allowances."""
        while True:
            now = time.monotonic()
            while self._recent and now - self._recent[0][0] >= 60.0:
                self._recent.popleft()
            fits = (
                len(self._recent) < self.requests_per_minute
                and sum(t for _, t in self._recent) + tokens <= self.tokens_per_minute
            )
            if fits:
                self._recent.append((now, tokens))
                return
            # The oldest entry is what frees capacity, so wait for it to age out.
            time.sleep(max(60.0 - (now - self._recent[0][0]) + 0.1, 0.1))


def _batch(texts: list[str], token_budget: int) -> list[tuple[int, list[str]]]:
    """Group texts into requests that fit the per-request limits."""
    batches: list[tuple[int, list[str]]] = []
    current: list[str] = []
    current_tokens = 0
    for text in texts:
        tokens = estimate_tokens(text)
        full = len(current) >= MAX_INPUTS_PER_REQUEST
        over_budget = current and current_tokens + tokens > token_budget
        if full or over_budget:
            batches.append((current_tokens, current))
            current, current_tokens = [], 0
        current.append(text)
        current_tokens += tokens
    if current:
        batches.append((current_tokens, current))
    return batches


def normalise(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise row-wise, so cosine similarity is a plain dot product."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # A zero vector cannot be normalised; leave it as-is rather than dividing.
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    """How long to wait after a 429: the server's answer, or a backoff."""
    header = response.headers.get("retry-after")
    if header:
        try:
            return min(max(float(header), 1.0), MAX_BACKOFF_SECONDS)
        except ValueError:
            pass  # Retry-After may be an HTTP date; fall back on backoff.
    return min(BACKOFF_SECONDS * attempt, MAX_BACKOFF_SECONDS)


def _post_with_retry(client: httpx.Client, headers: dict, body: dict) -> dict:
    """POST one batch, retrying while the account's real rate limit allows."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = client.post(ENDPOINT, headers=headers, json=body)
        if response.status_code == 200:
            return response.json()
        if response.status_code == 429 and attempt < MAX_ATTEMPTS:
            delay = _retry_delay(response, attempt)
            print(
                f"    rate limited, retrying in {delay:.0f}s "
                f"(attempt {attempt}/{MAX_ATTEMPTS})",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
            continue
        raise EmbeddingError(
            f"embedding request failed with {response.status_code}: "
            f"{response.text[:300]}"
        )
    raise EmbeddingError("embedding request exhausted its retry budget")


def embed_texts(
    texts: list[str],
    *,
    model: str = MODEL,
    dimensions: int = DIMENSIONS,
    cap: Cap | None = None,
) -> np.ndarray:
    """Embed texts in order, returning an L2-normalised (n, dimensions) array.

    This model embeds queries and documents into one space, so questions and
    chunks go through the same call with no asymmetric hint.

    `cap` is asked before each batch and told after it. Passing `None` runs
    uncapped, which is for tests that stub the transport and therefore spend
    nothing; every path that can reach the real endpoint passes one.
    """
    if not texts:
        return np.zeros((0, dimensions), dtype=np.float32)

    import httpx

    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }
    batches = _batch(texts, MAX_TOKENS_PER_REQUEST)
    limiter = _RateLimiter(REQUESTS_PER_MINUTE, TOKENS_PER_MINUTE)
    embedded = []

    with httpx.Client(timeout=120) as client:
        for number, (tokens, batch) in enumerate(batches, start=1):
            if cap is not None:
                # Before the request, not after. A batch that does not fit
                # stops the build here, with no index written and the previous
                # one still on disk.
                cap.authorise(price_for(model).usd(tokens), f"embedding batch {number}")
            limiter.acquire(tokens)
            print(
                f"  batch {number}/{len(batches)} "
                f"({len(batch)} texts, ~{tokens} tokens)",
                file=sys.stderr,
                flush=True,
            )
            payload = _post_with_retry(
                client,
                headers,
                {
                    "input": batch,
                    "model": model,
                    "dimensions": dimensions,
                    "encoding_format": "float",
                },
            )
            if cap is not None:
                # Billed at what the API says it used, not at the estimate the
                # authorisation was made against.
                billed = (payload.get("usage") or {}).get("prompt_tokens", tokens)
                entry = cap.record(model, int(billed))
                print(
                    f"    {billed} tokens, ${entry.usd:.4f}; "
                    f"${cap.spent_today():.4f} of ${cap.daily_usd:.2f} spent today",
                    file=sys.stderr,
                    flush=True,
                )
            data = sorted(payload["data"], key=lambda item: item["index"])
            if len(data) != len(batch):
                raise EmbeddingError(
                    f"asked for {len(batch)} embeddings, received {len(data)}"
                )
            embedded.append(
                np.array([item["embedding"] for item in data], dtype=np.float32)
            )

    vectors = np.vstack(embedded)
    if vectors.shape != (len(texts), dimensions):
        raise EmbeddingError(f"unexpected embedding shape {vectors.shape}")
    return normalise(vectors)
