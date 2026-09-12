"""Guarantee 5's cost half: the cap, and that it refuses before the call.

A cap that notices afterwards is a report. Most of what is pinned here is
therefore about *ordering* — that the request is never sent — rather than about
arithmetic, which is why several tests assert on a transport that records
whether it was reached at all.

Nothing here touches the network. The two commands that can are stubbed, and
the ledger is a `tmp_path` file, so the tests cost what the rest of the suite
costs: nothing.
"""

import json

import httpx
import pytest

from rag_contract.drafter import MAX_TOKENS, LiveDrafter
from rag_contract.embedding import embed_texts
from rag_contract.gate import load_thresholds
from rag_contract.spend import (
    PRICES,
    Cap,
    Entry,
    Ledger,
    SpendCapExceeded,
    SpendError,
    completion_bound_usd,
    estimate_tokens,
    price_for,
    today,
)


def cap(tmp_path, daily_usd=2.00, command="test"):
    return Cap(
        daily_usd=daily_usd,
        ledger=Ledger(tmp_path / "ledger.jsonl"),
        command=command,
    )


# --- Prices ---------------------------------------------------------------


def test_a_known_model_prices_from_the_committed_table():
    assert price_for("text-embedding-3-small").input_per_m == 0.02


def test_a_dated_snapshot_prices_as_its_family():
    """The snapshot is pinned for reproducibility, not billed differently."""
    assert price_for("gpt-5.5-2026-04-23") is PRICES["gpt-5.5"]


def test_an_unpriced_model_raises_rather_than_costing_nothing():
    """Pricing the unknown at zero is how a spend cap fails silently."""
    with pytest.raises(SpendError, match="no committed price"):
        price_for("some-model-nobody-priced")


def test_embeddings_have_no_output_side():
    assert price_for("text-embedding-3-small").usd(1_000_000, 999) == pytest.approx(
        0.02
    )


def test_output_tokens_cost_six_times_input_for_the_drafting_model():
    price = price_for("gpt-5.5")
    assert price.usd(0, 1_000_000) == pytest.approx(30.0)
    assert price.usd(1_000_000, 0) == pytest.approx(5.0)


def test_the_token_estimate_errs_high_against_a_real_invoice():
    """The measured run billed 75,250 input tokens where this scores 108,693.

    Over-estimating is the safe direction: a cap built on an optimistic
    estimate authorises the call that breaks it.
    """
    text = "The quick brown fox jumps over the lazy dog. " * 100
    assert estimate_tokens(text) > len(text) / 4.5


# --- Bounding a call before making it -------------------------------------


def test_a_completion_is_bounded_by_the_output_ceiling_the_request_sets():
    """Which is what makes pre-authorising a completion possible at all."""
    bound = completion_bound_usd("prompt", 1000, "gpt-5.5")
    assert bound > price_for("gpt-5.5").usd(0, 1000)
    assert bound == pytest.approx(
        price_for("gpt-5.5").usd(estimate_tokens("prompt"), 1000)
    )


def test_raising_the_output_ceiling_raises_the_bound():
    small = completion_bound_usd("prompt", 100, "gpt-5.5")
    large = completion_bound_usd("prompt", 4000, "gpt-5.5")
    assert large > small


# --- The ledger -----------------------------------------------------------


def test_an_empty_ledger_has_spent_nothing(tmp_path):
    assert Ledger(tmp_path / "none.jsonl").spent_on(today()) == 0.0


def test_entries_round_trip(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append(Entry("2026-09-13T10:00:00Z", "build-index", "m", 100, 0, 0.5))
    ledger.append(Entry("2026-09-13T11:00:00Z", "build-index", "m", 200, 0, 0.25))
    assert ledger.spent_on("2026-09-13") == pytest.approx(0.75)


def test_spend_is_totalled_per_day_not_for_all_time(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append(Entry("2026-09-12T10:00:00Z", "c", "m", 1, 0, 1.50))
    ledger.append(Entry("2026-09-13T10:00:00Z", "c", "m", 1, 0, 0.25))
    assert ledger.spent_on("2026-09-13") == pytest.approx(0.25)


def test_a_ledger_that_cannot_be_totalled_is_refused(tmp_path):
    """Rather than skipping the line and under-reporting what has been spent."""
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"at": "2026-09-13T10:00:00Z"}\n')
    with pytest.raises(SpendError, match="not a ledger entry"):
        Ledger(path).spent_on("2026-09-13")


# --- The cap --------------------------------------------------------------


def test_a_call_inside_the_remaining_budget_is_authorised(tmp_path):
    cap(tmp_path).authorise(0.50, "a call")


def test_a_call_that_would_break_the_cap_is_refused(tmp_path):
    with pytest.raises(SpendCapExceeded, match="daily cap"):
        cap(tmp_path, daily_usd=1.00).authorise(1.01, "a call")


def test_what_has_been_spent_today_narrows_what_is_left(tmp_path):
    spender = cap(tmp_path, daily_usd=1.00)
    spender.record("gpt-5.5", 100_000, 10_000)  # 0.50 + 0.30
    assert spender.spent_today() == pytest.approx(0.80)
    assert spender.remaining() == pytest.approx(0.20)
    with pytest.raises(SpendCapExceeded):
        spender.authorise(0.25, "one more")


def test_a_recorded_call_is_billed_at_reported_usage(tmp_path):
    entry = cap(tmp_path).record("gpt-5.5", 1_000_000, 1_000_000)
    assert entry.usd == pytest.approx(35.0)
    assert entry.input_tokens == 1_000_000


def test_the_refusal_says_where_the_cap_lives(tmp_path):
    with pytest.raises(SpendCapExceeded, match="eval/thresholds.yaml"):
        cap(tmp_path, daily_usd=0.01).authorise(1.0, "a call")


# --- The cap reaches the two commands that spend ---------------------------


class Transport:
    """Records whether the request was ever made."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return httpx.Response(200, json=self.payload)


def stub_client(monkeypatch, transport: Transport) -> None:
    """Point `embed_texts`'s client at a mock transport.

    The real class is captured first: patching `httpx.Client` with something
    that calls `httpx.Client` recurses.
    """
    real = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: real(transport=httpx.MockTransport(transport.handler)),
    )


def embedding_payload(count: int, dimensions: int) -> dict:
    return {
        "data": [
            {"index": i, "embedding": [0.0] * (dimensions - 1) + [1.0]}
            for i in range(count)
        ],
        "usage": {"prompt_tokens": 42},
    }


def test_embedding_stops_before_the_request_when_the_cap_is_spent(
    tmp_path, monkeypatch
):
    """The behaviour change: no index written, and no money spent finding out."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    transport = Transport(embedding_payload(1, 4))
    stub_client(monkeypatch, transport)

    with pytest.raises(SpendCapExceeded):
        embed_texts(
            ["x" * 300_000],
            dimensions=4,
            cap=cap(tmp_path, daily_usd=0.000001),
        )
    assert transport.calls == 0


def test_embedding_inside_the_cap_proceeds_and_bills_reported_usage(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    transport = Transport(embedding_payload(1, 4))
    stub_client(monkeypatch, transport)

    spender = cap(tmp_path)
    embed_texts(["hello"], dimensions=4, cap=spender)

    assert transport.calls == 1
    entries = spender.ledger.entries()
    assert len(entries) == 1
    # 42 is what the stubbed API reported, not what the estimator guessed.
    assert entries[0].input_tokens == 42
    assert entries[0].command == "test"


def test_embedding_without_a_cap_still_works(tmp_path, monkeypatch):
    """Tests that stub the transport spend nothing and need no cap."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    transport = Transport(embedding_payload(1, 4))
    stub_client(monkeypatch, transport)
    assert embed_texts(["hello"], dimensions=4).shape == (1, 4)


def test_drafting_stops_before_the_request_when_the_cap_is_spent(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls = []
    monkeypatch.setattr(httpx, "post", lambda *a, **k: calls.append(1))

    drafter = LiveDrafter(cap=cap(tmp_path, daily_usd=0.000001))
    with pytest.raises(SpendCapExceeded):
        drafter.draft("a question", [])
    assert calls == []


def test_drafting_is_authorised_against_the_output_ceiling_it_will_send(
    tmp_path, monkeypatch
):
    """The bound is real because the request itself sets `max_completion_tokens`.

    A cap sized just under that bound refuses; the same cap sized just over it
    permits, which is what shows the ceiling is what is being priced.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps({"claims": []})},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            },
        ),
    )
    bound = completion_bound_usd("", MAX_TOKENS, "gpt-5.5")

    with pytest.raises(SpendCapExceeded):
        LiveDrafter(cap=cap(tmp_path, daily_usd=bound / 2)).draft("q", [])

    spender = cap(tmp_path, daily_usd=bound * 2)
    assert LiveDrafter(cap=spender).draft("q", []) == []
    # Billed at what came back, not at the worst case it was authorised for.
    assert spender.spent_today() == pytest.approx(price_for("gpt-5.5").usd(10, 20))


# --- The committed numbers stay consistent with each other -----------------


def test_a_full_record_drafts_run_fits_inside_the_committed_cap():
    """The check that found something.

    `MAX_TOKENS` was 4000, which put a worst-case run at $3.90 against a $2.00
    cap — the cap would have refused to authorise the run it exists to permit.
    Two numbers in two files that nothing compared until guarantee 5 did.
    """
    limits = load_thresholds().budget
    questions = 28
    worst_case = questions * completion_bound_usd("", MAX_TOKENS, "gpt-5.5")
    assert worst_case < limits.daily_cap_usd
