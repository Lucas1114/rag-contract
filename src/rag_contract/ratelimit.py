"""The third ceiling of guarantee 5: what one client is allowed to ask for.

What a rate limit here can honestly protect
-------------------------------------------

Not cost. A served request spends nothing — committed vectors, committed
drafts, no key — so the money argument that justifies most public rate limits
does not exist here and should not be borrowed.

Not per-request work either. That is what the deadline in `budget.py` is for,
and the question set is fixed: the most expensive request a client can send is
q14, and it is expensive because of the claims the model returned, not because
of anything the caller chose.

What is left took a measurement, and the measurement found something sharper
than "protect availability". One process, one client per thread, `/answer/q14`:

    clients   req/s   observed p99   service reported (median)   abandoned
        1     131        8.6 ms            7.0 ms                    0
        4     134       44.7 ms            8.5 ms                    0
        8     134       82.2 ms           14.9 ms                    0
       16     134      179.4 ms           27.2 ms                    0
       32     134      297.2 ms           39.2 ms                    0
       64     139      481.2 ms           57.2 ms                 10.2%

Three things are in that table.

**Throughput is flat.** 131 req/s at one client and 139 at sixty-four. The
answer path is synchronous CPU work under one interpreter lock, so concurrency
buys nothing: the capacity of a process is about 130 requests a second and
adding callers only adds queue.

**The deadline cannot see the queue.** At 32 clients a caller waits 297 ms and
the service reports having spent 39. `Spend` starts when `Service.answer`
begins, which is after the request has been accepted, parsed and handed to a
worker thread — so guarantee 5's ceiling is a ceiling on service time, and the
time a client actually waits is not bounded by anything in this repository.

**And past a point the deadline fires on the wrong requests.** At 64 clients
one request in ten is abandoned with a 503, and their spend reports look like
this one:

    "stages": {"retrieval": 124.957, "drafting": 0.001, "grounding": 28.462}

125 ms in retrieval, for a stage that measures 0.026 ms of work. Nothing was
retrieved slowly; the thread holding that request was descheduled and the wall
clock kept running. The request did nothing unusual and was abandoned anyway,
because someone else was loud.

That is the argument, and it is not the one this module started out expecting
to make. Without a limit on how fast one client may ask, *the mechanism that
holds guarantee 5 becomes a way to deny service*: a single caller can push the
process into contention until other callers' requests are abandoned for
spending a budget they never spent on work. The rate limit is what keeps the
deadline pointed at the thing it was built to catch — an unbounded number of
claims — rather than at whoever happened to be sharing the process.

Why the limit costs an honest client nothing
--------------------------------------------

The corpus is fixed, the question set is fixed and the drafts are committed, so
the answer to q14 today is the answer to q14 tomorrow, byte for byte. There is
no honest workload that needs to ask faster than the committed allowance: 60
requests a minute is the entire 28-question set, twice over, every minute,
forever. A caller that wants the whole set is done in half a minute and has
nothing left to ask.

That is worth stating because it is what makes the number defensible rather
than arbitrary. A rate limit that has to be tuned against real usage is a limit
whose bar moves; this one is bounded above by arithmetic (below) and below by a
workload that cannot grow.

Refusing, not waiting
---------------------

`embedding.py` already contains a `_RateLimiter` and it is deliberately not
reused. It solves the opposite problem from the opposite side: it schedules
*our* requests against a vendor's published allowance, and when a request does
not fit it sleeps until it does. Blocking is right there — there is one caller,
it is a batch job, and waiting is cheaper than failing. Every one of those
properties is inverted here. There are many callers, the caller is not ours,
and a limiter that made a client wait would be holding a worker thread open for
exactly the client it has decided is asking for too much, which is the
contention above with extra steps. This one refuses, immediately, and says when
to come back.

Token bucket, because two numbers are needed
--------------------------------------------

A rate alone does not describe the failure measured above. 60 requests a minute
permits 60 at once and then silence, and 60 at once is the contention that
produced the abandoned requests. So the allowance has a rate *and* a depth:
`max_requests_per_minute` is how fast the bucket refills and `max_client_burst`
is how much of it a client may spend at once. At the committed 10, a single
client cannot put more than ten requests into a 130-per-second process
simultaneously, which the table above places well inside the region where
nothing is abandoned.

Who a client is, and why it is counted from the right
------------------------------------------------------

The socket peer, unless the deployment has declared how many proxies stand in
front of it. `trusted_proxy_hops` in `eval/thresholds.yaml` is that count, and
`resolve_client` below reads `X-Forwarded-For` from the *right* by exactly that
many entries.

The direction is the whole of the security argument. `X-Forwarded-For` is a
list each proxy appends to, and what it appends is the peer *it* saw. So the
rightmost entry was written by the hop nearest this process, the one to its
left by the hop before that, and so on: the last N entries are the testimony of
the N machines the deployment has decided to trust, and everything further left
was written by whoever was calling. Reading the leftmost entry — the usual
mistake, and the one that reads like "the original client" — trusts a value the
caller types, which lets any client mint a fresh identity per request and walk
past this limiter by setting a header.

Counting from the right also fails safe when the count is wrong. Declare more
hops than the chain actually has and the header is too short to satisfy them,
so the request falls back to the socket peer: every caller collapses into one
bucket, which is the old global-limit behaviour and refuses too much rather
than too little. There is no arrangement of headers a caller can send that
makes the list long enough to be believed, because a caller can only add
entries on the left.

The count was zero until this service was deployed behind one, and zero remains
the right answer for a process exposed directly. What it cost while it was zero
is worth stating, because it is what made the number necessary: behind a proxy
every request arrives from the proxy, so all callers shared one bucket and the
per-client limit was a global one — the second visitor to a public deployment
would have been refused.

What this does not cover
------------------------

Many clients rather than one. A per-client limit bounds what one caller can do
and cannot bound what a thousand of them do; that needs a global concurrency
limit or something upstream of the process, and neither is in this repository. Guarantee 5 is about ceilings this
service enforces on itself, and stating the edge of that is worth more than a
limiter that implies it covers more than it does.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

Clock = Callable[[], float]

# How many callers to remember at once. The table is the limiter's own memory
# footprint and therefore its own availability surface, so it is bounded rather
# than left to grow with whoever shows up. See `RateLimiter.check` for what
# happens at the bound, which is the conservative direction: forgetting a
# bucket only ever forgives a client, and a limiter that started refusing
# callers it has no record of would have become the outage it exists to stop.
MAX_TRACKED_CLIENTS = 10_000

UNKNOWN_CLIENT = "unknown"

FORWARDED_FOR = "x-forwarded-for"


def resolve_client(
    peer: str | None, forwarded_for: str | None, trusted_proxy_hops: int
) -> str:
    """Who is asking, as far as this service is willing to believe.

    `trusted_proxy_hops` entries in from the right of `X-Forwarded-For`, or the
    socket peer when the count is zero or the header cannot support it. See the
    module docstring for why the direction is the security property and why the
    fallback is the conservative one.

    A caller the transport cannot identify and no trusted hop named shares one
    bucket with every other such caller, which is again the conservative
    direction: the alternative is an unidentified client having no limit at all.
    """
    if trusted_proxy_hops > 0 and forwarded_for:
        hops = [entry.strip() for entry in forwarded_for.split(",")]
        hops = [entry for entry in hops if entry]
        if len(hops) >= trusted_proxy_hops:
            return hops[-trusted_proxy_hops]
    return peer or UNKNOWN_CLIENT


@dataclass(frozen=True)
class Decision:
    """What the limiter says about one request, before anything serves it."""

    allowed: bool
    client: str
    retry_after_s: int
    remaining: float

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "retry_after_s": self.retry_after_s,
            "remaining": round(self.remaining, 3),
        }


@dataclass
class _Bucket:
    tokens: float
    last: float


class RateLimiter:
    """One token bucket per client, refilled by the clock rather than a timer.

    The clock is injected for the same reason `Budget`'s is: the committed
    allowance cannot be reached by any test that waits, and a test that proved
    this one by sleeping for a minute would be proving that `time.sleep` works.
    Driving the clock is also what lets the gate check the behaviour
    deterministically — see `gate.rate_limit_check`.

    Refilling on read rather than on a schedule means an idle client costs
    nothing between requests, and that a bucket nobody has touched for long
    enough is indistinguishable from one that never existed. That is what makes
    the eviction below safe.
    """

    def __init__(
        self,
        requests_per_minute: int,
        burst: int,
        *,
        clock: Clock = time.monotonic,
        max_clients: int = MAX_TRACKED_CLIENTS,
    ):
        self.requests_per_minute = requests_per_minute
        self.burst = burst
        self.clock = clock
        self.max_clients = max_clients
        self._per_second = requests_per_minute / 60.0
        self._buckets: OrderedDict[str, _Bucket] = OrderedDict()
        # Requests are served on a thread pool — the handlers are synchronous —
        # so two of them can reach this at once. The lock is held only for the
        # arithmetic below and never while anything is being served, which is
        # the same rule `IndexRegistry` follows.
        self._lock = threading.Lock()

    def check(self, client: str) -> Decision:
        """Spend one token for `client`, or refuse and say when to come back."""
        now = self.clock()
        with self._lock:
            bucket = self._buckets.get(client)
            if bucket is None:
                bucket = _Bucket(tokens=float(self.burst), last=now)
                self._buckets[client] = bucket
            else:
                elapsed = max(now - bucket.last, 0.0)
                bucket.tokens = min(
                    float(self.burst), bucket.tokens + elapsed * self._per_second
                )
                bucket.last = now
            self._buckets.move_to_end(client)

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                decision = Decision(
                    allowed=True,
                    client=client,
                    retry_after_s=0,
                    remaining=bucket.tokens,
                )
            else:
                # The refused request does not spend a token. A limiter that
                # charged for refusals would push a hammering client's recovery
                # further away every time it knocked, which turns a rate limit
                # into a ban nobody committed a number for.
                wait = (1.0 - bucket.tokens) / self._per_second
                decision = Decision(
                    allowed=False,
                    client=client,
                    # Retry-After is whole seconds, and rounding down would
                    # invite a retry that is still too early.
                    retry_after_s=max(1, math.ceil(wait)),
                    remaining=bucket.tokens,
                )

            # Evicting in access order drops the client that has gone longest
            # without asking for anything, which is the one whose bucket is
            # closest to full — so the state being thrown away is the state
            # that says the least. It can only ever forgive.
            while len(self._buckets) > self.max_clients:
                self._buckets.popitem(last=False)

        return decision

    @property
    def tracked(self) -> int:
        with self._lock:
            return len(self._buckets)
