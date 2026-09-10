"""The one network-calling module in the project.

Embeddings are computed once, by `build-index`, and the vectors are committed.
Nothing else — not the eval, not CI — calls this. Keeping the dependency in one
module makes that claim checkable by import graph rather than asserted in prose.
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from pathlib import Path

import httpx
import numpy as np

MODEL = "voyage-4-lite"
DIMENSIONS = 1024
ENDPOINT = "https://api.voyageai.com/v1/embeddings"

# Voyage accepts up to 1000 inputs and 1M tokens per request, so request size is
# never the binding constraint. The account rate limit is: with no payment
# method on file, 3 requests and 10k tokens per minute. The corpus is around
# 170k tokens, so a build from a fresh account takes roughly twenty minutes.
#
# That is acceptable for something run once whose output is committed, and
# encoding the free-tier limits here keeps the build reproducible on any
# account. An account with billing enabled can raise both numbers.
REQUESTS_PER_MINUTE = 3
TOKENS_PER_MINUTE = 10_000
BATCH_SIZE = 64
TIMEOUT_SECONDS = 120

# Retry budget for a 429 the client-side limiter did not prevent.
MAX_ATTEMPTS = 6
BACKOFF_SECONDS = 20

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


class EmbeddingError(RuntimeError):
    """The embedding API could not be reached or answered unusably."""


def _api_key() -> str:
    key = os.environ.get("VOYAGE_API_KEY", "").strip()
    if not key and _ENV_PATH.is_file():
        for line in _ENV_PATH.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "VOYAGE_API_KEY":
                key = value.strip()
                break
    if not key:
        raise EmbeddingError(
            "VOYAGE_API_KEY is not set. Embeddings are built once and committed; "
            "the eval runs off those committed vectors and needs no key."
        )
    return key


def estimate_tokens(text: str) -> int:
    """A deliberate over-estimate of a text's token count.

    Used only to stay inside the per-minute token allowance. Over-estimating
    costs a little wall clock on a one-time build; under-estimating costs a 429.
    """
    return max(1, len(text) // 3)


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
    """Group texts into requests that fit the per-minute token allowance."""
    batches: list[tuple[int, list[str]]] = []
    current: list[str] = []
    current_tokens = 0
    for text in texts:
        tokens = estimate_tokens(text)
        full = len(current) >= BATCH_SIZE
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


def _post_with_retry(client: httpx.Client, headers: dict, body: dict) -> dict:
    """POST one batch, retrying a 429 the client-side limiter did not prevent."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = client.post(ENDPOINT, headers=headers, json=body)
        if response.status_code == 200:
            return response.json()
        if response.status_code == 429 and attempt < MAX_ATTEMPTS:
            delay = BACKOFF_SECONDS * attempt
            print(
                f"    rate limited, retrying in {delay}s "
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
    input_type: str,
    *,
    model: str = MODEL,
    dimensions: int = DIMENSIONS,
) -> np.ndarray:
    """Embed texts in order, returning an L2-normalised (n, dimensions) array.

    `input_type` is "document" for corpus chunks and "query" for questions;
    Voyage embeds the two asymmetrically, and mixing them costs recall.
    """
    if input_type not in {"document", "query"}:
        raise ValueError(f"input_type must be document or query, not {input_type!r}")
    if not texts:
        return np.zeros((0, dimensions), dtype=np.float32)

    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }
    batches = _batch(texts, TOKENS_PER_MINUTE)
    limiter = _RateLimiter(REQUESTS_PER_MINUTE, TOKENS_PER_MINUTE)
    embedded = []

    with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
        for number, (tokens, batch) in enumerate(batches, start=1):
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
                    "input_type": input_type,
                    "output_dimension": dimensions,
                },
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
