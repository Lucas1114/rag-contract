"""Which index a request uses, and what happens when that changes.

Guarantee 4 has two halves. The index version is content-addressed, which
`index.py` already does. This is the other half: the service holds one current
index, that index can be replaced while the service is running, and no request
in flight is dropped or silently moved onto the new one.

The mechanism is a single reference
-----------------------------------

The registry holds a reference to an `Index`. A request acquires that reference
once, at entry, and uses the object it got for the rest of its work. A cutover
rebinds the registry's reference to a new object. A request that acquired
before the swap is already holding the old `Index` and finishes against it;
the old object stays alive exactly as long as some request still holds it, and
is collected when the last one returns.

That is the whole implementation, and it is worth saying why so little is
needed. There is no drain, no quiesce, no request counter, no reader-writer
lock and no grace period. Rebinding a name cannot produce a half-swapped
object, so a reader sees either the whole old index or the whole new one and
never a mixture of the two. The lock below guards the registry's own
bookkeeping — the reference and the generation log staying consistent with each
other — and is never held while a request is being served.

Acquiring once is the part that matters
---------------------------------------

A request that re-read the registry between steps could take its query vector
from one index and its chunks from another. Those are different arrays over
different text, addressed by different versions, and the answer built from the
mixture would be attributable to neither. `Index` is frozen and holds the
question vectors, the chunk vectors and the chunks together for that reason:
acquiring it once is acquiring all of them at once, and the version the
response reports is the version of the object the answer was actually built
from rather than whatever is current by the time the response is written.

An empty registry is a real state
---------------------------------

`acquire()` returns `None` when no index is loaded — before the first load,
or after `unload`. That is the index being unavailable, and it is what makes
`no_context` reachable. P04 defined that state and could not exercise it:
brute-force cosine over 868 chunks always returns ten, so no *question* can
reach it. It was never a property of the question, it is a property of the
service, and this is where it lives.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .index import INDEX_DIR, Index, load_index


@dataclass(frozen=True)
class Generation:
    """One index this registry has served, and why it started serving it."""

    version: str | None  # None when the generation is an unload
    at: str
    reason: str

    def to_dict(self) -> dict:
        return {"version": self.version, "at": self.at, "reason": self.reason}


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class IndexRegistry:
    """The service's current index, and the swap that replaces it."""

    def __init__(self, index: Index | None = None, *, reason: str = "initial load"):
        self._lock = threading.Lock()
        self._index = index
        self._history: list[Generation] = []
        if index is not None:
            self._history.append(Generation(index.version, _now(), reason))

    @classmethod
    def from_directory(cls, directory: Path = INDEX_DIR) -> IndexRegistry:
        return cls(load_index(directory))

    def acquire(self) -> Index | None:
        """The index this request will use, from entry to completion.

        Deliberately a plain read of the reference, with no lock: taking one
        here would serialise every request behind a swap that happens a handful
        of times in a deployment's life, to protect against an interleaving
        that cannot occur. The caller keeps the returned object; a later swap
        cannot reach into it.
        """
        return self._index

    @property
    def version(self) -> str | None:
        """The version currently being handed out. `None` when unloaded."""
        index = self._index
        return None if index is None else index.version

    @property
    def loaded(self) -> bool:
        return self._index is not None

    @property
    def history(self) -> tuple[Generation, ...]:
        with self._lock:
            return tuple(self._history)

    def install(self, index: Index, *, reason: str = "cutover") -> Generation:
        """Swap in a new index. Requests already in flight keep the old one.

        Returns the generation record rather than nothing, so a caller can log
        exactly what it cut over to without re-reading the registry and racing
        the next swap.
        """
        generation = Generation(index.version, _now(), reason)
        with self._lock:
            self._index = index
            self._history.append(generation)
        return generation

    def unload(self, *, reason: str = "index withdrawn") -> Generation:
        """Make the index unavailable. Requests in flight still finish.

        The state `no_context` describes. A request that has already acquired
        an index is unaffected — it holds the object, not the registry — and
        one arriving afterwards gets `None` and is answered with a 503 rather
        than with a refusal, because nothing was consulted and so nothing was
        decided.
        """
        generation = Generation(None, _now(), reason)
        with self._lock:
            self._index = None
            self._history.append(generation)
        return generation

    def reload(
        self, directory: Path = INDEX_DIR, *, reason: str = "reload"
    ) -> Generation:
        """Load the index from disk and cut over to it.

        The load happens before the swap, so a rebuild that produced a
        half-written or inconsistent index raises out of `load_index` and the
        registry keeps serving what it already had. A failed cutover leaves the
        service on the last index known to be good.
        """
        index = load_index(directory)
        return self.install(index, reason=reason)

    def to_dict(self) -> dict:
        return {
            "index_version": self.version,
            "loaded": self.loaded,
            "generations": [g.to_dict() for g in self.history],
        }
