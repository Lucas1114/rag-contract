"""Guarantee 5's third ceiling: what one client is allowed to ask for.

Every test here drives the clock rather than the calendar, for the reason
`test_budget.py` drives one rather than sleeping: the committed allowance
refills over a minute, and a test that proved the refill by waiting sixty
seconds would be proving that `time.monotonic` advances.

What is worth pinning is not the arithmetic of a token bucket, which is
standard, but the choices in it that are not: that a refusal costs nothing,
that an idle client cannot bank a deeper burst, that clients cannot reach each
other's buckets, and that running out of table space forgives rather than
refuses. Each of those is a way a rate limiter turns into an outage.
"""

import threading

import pytest

from rag_contract.budget import MAX_CLIENT_SHARE, PROCESS_MINUTE_MS, BudgetLimits
from rag_contract.gate import load_thresholds
from rag_contract.ratelimit import UNKNOWN_CLIENT, RateLimiter, resolve_client


class Clock:
    """Seconds, and only when told."""

    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def limiter(requests_per_minute=60, burst=10, clock=None, **kwargs):
    return RateLimiter(requests_per_minute, burst, clock=clock or Clock(), **kwargs)


# --- The allowance, and its depth -----------------------------------------


def test_a_client_inside_its_burst_is_allowed():
    limit = limiter(burst=10)
    assert all(limit.check("10.0.0.1").allowed for _ in range(10))


def test_the_burst_is_the_depth_and_the_next_request_is_refused():
    """The number that bounds simultaneity, which the rate alone does not.

    60 a minute permits 60 at once, and 60 at once is what the load measurement
    in `ratelimit.py` found abandoning other clients' requests.
    """
    limit = limiter(burst=10)
    for _ in range(10):
        limit.check("10.0.0.1")
    assert not limit.check("10.0.0.1").allowed


def test_a_refusal_says_when_to_come_back():
    limit = limiter(requests_per_minute=60, burst=10)
    for _ in range(10):
        limit.check("10.0.0.1")
    assert limit.check("10.0.0.1").retry_after_s == 1


def test_retry_after_is_rounded_up_rather_than_down():
    """A 429 whose Retry-After rounds down invites a retry that is still early.

    At 6 requests a minute a token takes 10 seconds, so a client refused
    immediately after spending one waits the whole ten rather than nine.
    """
    limit = limiter(requests_per_minute=6, burst=1)
    limit.check("10.0.0.1")
    assert limit.check("10.0.0.1").retry_after_s == 10


def test_retry_after_is_never_zero():
    """Whatever the rate, a refusal that says "retry now" is a retry loop."""
    limit = limiter(requests_per_minute=6000, burst=1)
    limit.check("10.0.0.1")
    assert limit.check("10.0.0.1").retry_after_s >= 1


# --- Refilling -------------------------------------------------------------


def test_the_bucket_refills_at_the_committed_rate():
    clock = Clock()
    limit = limiter(requests_per_minute=60, burst=10, clock=clock)
    for _ in range(10):
        limit.check("10.0.0.1")
    assert not limit.check("10.0.0.1").allowed
    clock.advance(1.0)  # 60 a minute is one a second
    assert limit.check("10.0.0.1").allowed


def test_an_idle_client_cannot_bank_a_deeper_burst():
    """Otherwise the burst is whatever a client is willing to wait for.

    A bucket that accumulated past its depth would let a caller sit quiet for
    an hour and then open three thousand requests at once, which is precisely
    the contention the depth exists to bound.
    """
    clock = Clock()
    limit = limiter(requests_per_minute=60, burst=10, clock=clock)
    clock.advance(3600.0)
    allowed = sum(1 for _ in range(20) if limit.check("10.0.0.1").allowed)
    assert allowed == 10


def test_a_refusal_does_not_spend_a_token():
    """A limiter that charged for knocking would be a ban nobody committed.

    The client below is refused nine times and then waits exactly one token's
    worth of time. If refusals were charged, its recovery would have moved
    nine seconds further away each time it asked.
    """
    clock = Clock()
    limit = limiter(requests_per_minute=60, burst=1, clock=clock)
    limit.check("10.0.0.1")
    for _ in range(9):
        assert not limit.check("10.0.0.1").allowed
    clock.advance(1.0)
    assert limit.check("10.0.0.1").allowed


# --- One client is one client ---------------------------------------------


def test_one_client_exhausting_its_allowance_does_not_refuse_another():
    """The whole point. The limit is per client or it is an outage."""
    limit = limiter(burst=10)
    for _ in range(20):
        limit.check("10.0.0.1")
    assert limit.check("10.0.0.2").allowed


def test_concurrent_requests_from_one_client_spend_one_bucket():
    """The handlers are synchronous, so two requests do reach this at once.

    Without the lock this is the classic read-modify-write race and a client
    gets more than its burst — quietly, and only under the load the limit
    exists for.
    """
    limit = limiter(burst=10)
    allowed = []
    lock = threading.Lock()

    def ask():
        decision = limit.check("10.0.0.1")
        with lock:
            allowed.append(decision.allowed)

    threads = [threading.Thread(target=ask) for _ in range(50)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(allowed) == 10


# --- The table is bounded, and gives way in the safe direction -------------


def test_the_client_table_does_not_grow_without_bound():
    """The limiter's own memory is an availability surface too."""
    limit = limiter(burst=10, max_clients=100)
    for n in range(1000):
        limit.check(f"10.0.0.{n}")
    assert limit.tracked == 100


def test_running_out_of_table_space_forgives_rather_than_refuses():
    """A limiter that refused callers it had no record of would be the outage.

    The evicted client here is the one that has gone longest without asking,
    which is also the one whose bucket is closest to full — so the state thrown
    away is the state that says the least.
    """
    limit = limiter(burst=1, max_clients=2)
    limit.check("victim")  # spends the victim's only token
    assert not limit.check("victim").allowed
    limit.check("other")
    limit.check("evictor")  # pushes the victim out of the table
    assert limit.check("victim").allowed


# --- The committed numbers -------------------------------------------------


def test_the_committed_allowance_is_what_the_service_enforces():
    limits = load_thresholds().budget
    limit = limits.limiter()
    assert limit.requests_per_minute == limits.max_requests_per_minute
    assert limit.burst == limits.max_client_burst


def test_the_committed_allowance_is_a_minority_of_one_process_minute():
    """The arithmetic that makes 60 a number rather than a preference."""
    limits = load_thresholds().budget
    share = (
        limits.max_requests_per_minute * limits.request_deadline_ms
    ) / PROCESS_MINUTE_MS
    assert share <= MAX_CLIENT_SHARE


def test_the_committed_allowance_covers_the_whole_question_set():
    """A limit an honest caller reaches is a limit that will be raised.

    The corpus is fixed and the drafts are committed, so the entire service is
    28 answers that do not change. The allowance is more than twice that a
    minute, which is why nothing here needs tuning against real traffic.
    """
    from rag_contract.evalset import load_questions

    assert load_thresholds().budget.max_requests_per_minute >= 2 * len(load_questions())


def test_a_burst_deeper_than_the_minutes_allowance_is_refused():
    from rag_contract.budget import BudgetError

    from .test_budget import LIMITS

    with pytest.raises(BudgetError, match="spent all at once"):
        BudgetLimits.from_mapping(
            {**LIMITS, "max_requests_per_minute": 10, "max_client_burst": 11}
        )


# --- Who a client is ------------------------------------------------------
#
# The identity, not the allowance. Every test here is one line of
# `resolve_client`, and each of them is a way the limiter stops being a limiter:
# read the header from the wrong end and any caller can mint an identity per
# request; ignore it behind a proxy and every visitor becomes one client.


def test_with_no_trusted_hop_the_client_is_the_socket_peer():
    assert resolve_client("10.0.0.1", None, 0) == "10.0.0.1"


def test_with_no_trusted_hop_the_header_changes_nothing():
    """A header nobody is meant to believe may not decide who is asking."""
    assert resolve_client("10.0.0.1", "1.1.1.1, 2.2.2.2", 0) == "10.0.0.1"


def test_one_trusted_hop_reads_the_entry_that_proxy_appended():
    """The rightmost entry is the peer the nearest proxy actually saw."""
    assert resolve_client("10.0.0.1", "203.0.113.9", 1) == "203.0.113.9"


def test_entries_left_of_the_trusted_hop_are_the_callers_and_are_ignored():
    """The forgery this counts from the right to defeat.

    A caller sending its own `X-Forwarded-For` only ever prepends: the proxy
    appends what it saw afterwards. So everything left of the trusted hop is
    the caller's own text, and reading it would hand a fresh identity to
    anyone who asked for one.
    """
    assert resolve_client("10.0.0.1", "evil, 203.0.113.9", 1) == "203.0.113.9"
    assert resolve_client("10.0.0.1", "a, b, c, 203.0.113.9", 1) == "203.0.113.9"


def test_two_trusted_hops_step_past_the_inner_proxy():
    """A CDN in front of the platform edge: the visitor is two in from the right."""
    assert resolve_client("10.0.0.1", "evil, 203.0.113.9, 198.51.100.4", 2) == (
        "203.0.113.9"
    )


def test_a_header_too_short_for_the_committed_chain_falls_back_to_the_peer():
    """Declaring more hops than exist refuses too much rather than too little.

    There is no header a caller can send that makes the list long enough to be
    believed, because a caller can only add entries on the left — so the
    failure mode of a wrong count is the old shared-bucket behaviour, not an
    unlimited one.
    """
    assert resolve_client("10.0.0.1", "203.0.113.9", 2) == "10.0.0.1"


def test_a_missing_header_behind_a_trusted_hop_falls_back_to_the_peer():
    assert resolve_client("10.0.0.1", None, 1) == "10.0.0.1"
    assert resolve_client("10.0.0.1", "", 1) == "10.0.0.1"


def test_whitespace_and_empty_entries_do_not_shift_the_count():
    """`X-Forwarded-For` is comma-space separated and proxies are inconsistent.

    An entry the parser miscounts is an entry the count points past, which is
    the forgery this is meant to stop.
    """
    assert resolve_client("10.0.0.1", " evil ,  203.0.113.9 ", 1) == "203.0.113.9"
    assert resolve_client("10.0.0.1", "evil, , 203.0.113.9", 1) == "203.0.113.9"


def test_a_caller_the_transport_cannot_name_shares_one_bucket():
    """The conservative direction: the alternative is no limit at all."""
    assert resolve_client(None, None, 0) == UNKNOWN_CLIENT
    assert resolve_client(None, None, 1) == UNKNOWN_CLIENT
