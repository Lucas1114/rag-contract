"""What CI is allowed to do, asserted rather than promised.

Two constraints hold the whole project up. CI must not call a real LLM, and it
must not need an API key — otherwise the regression gate is unrunnable by a
reviewer, unreproducible, and quietly billable. Both are properties of the
import graph and the filesystem, so both can be checked here instead of being
believed.
"""

import json
import os
import subprocess
import sys

from rag_contract.cli import RESULTS_PATH
from rag_contract.evalset import load_questions
from rag_contract.evaluate import evaluate
from rag_contract.gate import freshness_check, load_thresholds, run_gate
from rag_contract.index import load_index
from rag_contract.lifecycle import index_status

NETWORK_MODULES = ("httpx", "requests", "urllib3", "openai", "anthropic")

# Everything the gate pulls in, plus the request path that serves the same
# index. `drafter` and `embedding` are both here and both must stay clean:
# their network clients are imported inside the functions that call them —
# `LiveDrafter.draft` and `embed_texts` — rather than at module scope, which is
# what keeps them out of this graph.
#
# `embedding` was outside this list until the index lifecycle needed the model
# name to say whether the committed index still describes the corpus. Moving
# its `import httpx` into the function was the alternative to duplicating the
# model name somewhere safer, and it is the stronger outcome: the one module
# that talks to an embedding API is now provably importable without one.
GATE_IMPORT = (
    "from rag_contract import "
    "answer_eval, answering, cli, drafter, embedding, evaluate, evalset, gate, "
    "grounding, index, lifecycle, registry, retrieval, service"
)


def test_the_gate_import_graph_reaches_no_network_client():
    probe = (
        f"{GATE_IMPORT}\n"
        "import sys\n"
        f"print(','.join(m for m in {NETWORK_MODULES!r} if m in sys.modules))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert loaded.stdout.strip() == "", (
        f"the gate imports {loaded.stdout.strip()}; CI would be one call away "
        "from a real API"
    )


def test_the_gate_runs_with_every_api_key_scrubbed():
    environment = {
        k: v for k, v in os.environ.items() if not k.endswith(("_API_KEY", "_TOKEN"))
    }
    completed = subprocess.run(
        [sys.executable, "-m", "rag_contract.cli", "gate", "--check-report", "-q"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_the_committed_report_still_matches_the_committed_index():
    # The README quotes this file. A stale report means the README quotes a
    # number no commit ever measured.
    report = evaluate(load_index(), load_questions())
    assert json.loads(RESULTS_PATH.read_text()) == report


def test_the_committed_index_describes_the_committed_corpus():
    # Guarantee 4 in CI. The vectors are a build artefact produced by a
    # command that needs a key, so a corpus change cannot trigger a rebuild
    # here — only a red build naming the command that does.
    assert freshness_check(index_status(load_index())).passed


def test_the_committed_index_clears_the_committed_thresholds():
    checks = run_gate(json.loads(RESULTS_PATH.read_text()), load_thresholds())
    failed = [c.name for c in checks if not c.passed]
    assert failed == []
