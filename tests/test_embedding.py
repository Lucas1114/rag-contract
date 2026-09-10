"""The build client never runs in CI, but its scheduling decides whether a
build survives whatever rate limit the account it runs under happens to have."""

import httpx

from rag_contract.embedding import (
    DIMENSIONS,
    MAX_BACKOFF_SECONDS,
    MAX_INPUTS_PER_REQUEST,
    MAX_TOKENS_PER_REQUEST,
    _batch,
    _RateLimiter,
    _retry_delay,
    embed_texts,
    estimate_tokens,
)


def response(status: int, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, headers=headers or {})


def test_batches_stay_inside_the_per_request_token_limit():
    texts = ["x" * 3000] * 200  # ~1000 estimated tokens each
    for tokens, batch in _batch(texts, MAX_TOKENS_PER_REQUEST):
        assert tokens <= MAX_TOKENS_PER_REQUEST
        assert batch


def test_batches_stay_inside_the_per_request_input_limit():
    batches = _batch(["x"] * (MAX_INPUTS_PER_REQUEST * 3), MAX_TOKENS_PER_REQUEST)
    assert all(len(batch) <= MAX_INPUTS_PER_REQUEST for _, batch in batches)


def test_batching_preserves_every_text_in_order():
    texts = [f"text {i}" for i in range(1000)]
    flattened = [t for _, batch in _batch(texts, MAX_TOKENS_PER_REQUEST) for t in batch]
    assert flattened == texts


def test_a_single_oversized_text_still_gets_its_own_batch():
    # Splitting a text is not this function's job; truncation is the API's.
    assert (
        len(_batch(["x" * (MAX_TOKENS_PER_REQUEST * 10)], MAX_TOKENS_PER_REQUEST)) == 1
    )


def test_token_estimate_errs_high():
    # Roughly four characters per token in English, so a third is a safe margin.
    text = "The quick brown fox jumps over the lazy dog. " * 20
    assert estimate_tokens(text) > len(text.split())


def test_limiter_admits_up_to_the_request_allowance_without_waiting():
    limiter = _RateLimiter(requests_per_minute=3, tokens_per_minute=10_000)
    for _ in range(3):
        limiter.acquire(100)
    assert len(limiter._recent) == 3


def test_limiter_blocks_once_the_request_allowance_is_spent(monkeypatch):
    slept = []
    monkeypatch.setattr("rag_contract.embedding.time.sleep", slept.append)
    clock = iter([0.0, 0.0, 0.0, 61.0, 61.0])
    monkeypatch.setattr("rag_contract.embedding.time.monotonic", lambda: next(clock))

    limiter = _RateLimiter(requests_per_minute=2, tokens_per_minute=10_000)
    limiter.acquire(100)
    limiter.acquire(100)
    limiter.acquire(100)  # over the request allowance
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


def test_a_server_supplied_retry_after_wins_over_the_backoff():
    assert _retry_delay(response(429, {"retry-after": "7"}), attempt=1) == 7.0


def test_retry_after_is_capped():
    delay = _retry_delay(response(429, {"retry-after": "99999"}), attempt=1)
    assert delay == MAX_BACKOFF_SECONDS


def test_a_date_formatted_retry_after_falls_back_on_the_backoff():
    # Retry-After may be an HTTP date, which is not a delay in seconds.
    delay = _retry_delay(
        response(429, {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}), attempt=2
    )
    assert delay > 0


def test_backoff_grows_with_the_attempt_and_is_capped():
    delays = [_retry_delay(response(429), attempt=n) for n in range(1, 12)]
    assert delays == sorted(delays)
    assert max(delays) == MAX_BACKOFF_SECONDS


def test_no_texts_makes_no_request():
    # No key needed, and no network: the empty case short-circuits.
    assert embed_texts([]).shape == (0, DIMENSIONS)
