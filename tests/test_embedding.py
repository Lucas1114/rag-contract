"""The build client never runs in CI, but its throttle decides whether a build
from a fresh account completes or dies half way through 868 chunks."""

import pytest

from rag_contract.embedding import (
    BATCH_SIZE,
    TOKENS_PER_MINUTE,
    _batch,
    _RateLimiter,
    embed_texts,
    estimate_tokens,
)


def test_batches_stay_inside_the_token_allowance():
    texts = ["x" * 3000] * 40  # ~1000 estimated tokens each
    for tokens, batch in _batch(texts, TOKENS_PER_MINUTE):
        assert tokens <= TOKENS_PER_MINUTE
        assert batch


def test_batches_stay_inside_the_input_count_limit():
    for _, batch in _batch(["x"] * (BATCH_SIZE * 3), TOKENS_PER_MINUTE):
        assert len(batch) <= BATCH_SIZE


def test_batching_preserves_every_text_in_order():
    texts = [f"text {i}" for i in range(200)]
    flattened = [t for _, batch in _batch(texts, TOKENS_PER_MINUTE) for t in batch]
    assert flattened == texts


def test_a_single_oversized_text_still_gets_its_own_batch():
    # Splitting a text is not this function's job; truncation is the API's.
    batches = _batch(["x" * (TOKENS_PER_MINUTE * 10)], TOKENS_PER_MINUTE)
    assert len(batches) == 1


def test_token_estimate_errs_high():
    # Roughly four characters per token in English, so a third is a safe margin.
    text = "The quick brown fox jumps over the lazy dog. " * 20
    assert estimate_tokens(text) > len(text.split())


def test_limiter_admits_up_to_the_request_allowance_without_waiting():
    limiter = _RateLimiter(requests_per_minute=3, tokens_per_minute=10_000)
    for _ in range(3):
        limiter.acquire(100)  # would block on the fourth
    assert len(limiter._recent) == 3


def test_limiter_blocks_once_an_allowance_is_spent(monkeypatch):
    slept = []
    monkeypatch.setattr("rag_contract.embedding.time.sleep", slept.append)

    clock = iter([0.0, 0.0, 0.0, 61.0, 61.0])
    monkeypatch.setattr("rag_contract.embedding.time.monotonic", lambda: next(clock))

    limiter = _RateLimiter(requests_per_minute=2, tokens_per_minute=10_000)
    limiter.acquire(100)
    limiter.acquire(100)
    limiter.acquire(100)  # over the request allowance: sleeps, then the window clears
    assert slept and slept[0] > 0


def test_the_token_allowance_binds_as_well_as_the_request_allowance(monkeypatch):
    slept = []
    monkeypatch.setattr("rag_contract.embedding.time.sleep", slept.append)
    clock = iter([0.0, 0.0, 61.0, 61.0])
    monkeypatch.setattr("rag_contract.embedding.time.monotonic", lambda: next(clock))

    limiter = _RateLimiter(requests_per_minute=10, tokens_per_minute=1000)
    limiter.acquire(900)
    limiter.acquire(900)  # inside the request allowance, over the token one
    assert slept


def test_input_type_must_be_document_or_query():
    with pytest.raises(ValueError, match="input_type"):
        embed_texts(["anything"], "passage")


def test_no_texts_makes_no_request():
    # No key needed, and no network: the empty case short-circuits.
    assert embed_texts([], "document").shape == (0, 1024)
