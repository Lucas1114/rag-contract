"""The one network-calling module in the project.

Embeddings are computed once, by `build-index`, and the vectors are committed.
Nothing else — not the eval, not CI — calls this. Keeping the dependency in one
module makes that claim checkable rather than asserted.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import numpy as np

MODEL = "voyage-3.5-lite"
DIMENSIONS = 1024
ENDPOINT = "https://api.voyageai.com/v1/embeddings"

# Voyage accepts up to 1000 inputs per request; the practical limit is the
# per-request token budget, so batches stay well under it.
BATCH_SIZE = 64
TIMEOUT_SECONDS = 120

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


def normalise(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise row-wise, so cosine similarity is a plain dot product."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # A zero vector cannot be normalised; leave it as-is rather than dividing.
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


def embed_texts(
    texts: list[str],
    input_type: str,
    *,
    model: str = MODEL,
    dimensions: int = DIMENSIONS,
) -> np.ndarray:
    """Embed texts in order, returning an L2-normalised (n, dimensions) array.

    `input_type` is "document" for corpus chunks and "query" for questions;
    Voyage embeds the two asymmetrically and mixing them costs recall.
    """
    if input_type not in {"document", "query"}:
        raise ValueError(f"input_type must be document or query, not {input_type!r}")
    if not texts:
        return np.zeros((0, dimensions), dtype=np.float32)

    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }
    batches = []
    with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            response = client.post(
                ENDPOINT,
                headers=headers,
                json={
                    "input": batch,
                    "model": model,
                    "input_type": input_type,
                    "output_dimension": dimensions,
                },
            )
            if response.status_code != 200:
                raise EmbeddingError(
                    f"embedding request failed with {response.status_code}: "
                    f"{response.text[:300]}"
                )
            payload = response.json()
            data = sorted(payload["data"], key=lambda item: item["index"])
            if len(data) != len(batch):
                raise EmbeddingError(
                    f"asked for {len(batch)} embeddings, received {len(data)}"
                )
            batches.append(
                np.array([item["embedding"] for item in data], dtype=np.float32)
            )

    vectors = np.vstack(batches)
    if vectors.shape != (len(texts), dimensions):
        raise EmbeddingError(f"unexpected embedding shape {vectors.shape}")
    return normalise(vectors)
