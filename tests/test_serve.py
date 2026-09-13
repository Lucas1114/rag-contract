"""The entry point a container and a developer both use.

There is nothing to measure here — `serve` composes nothing and decides
nothing, which is the point of it. What is worth pinning is the one piece of
configuration this process takes from its environment, and that it is the only
one: everything that governs behaviour is committed to `eval/thresholds.yaml`
so that a deployment cannot move it, and a second environment variable creeping
in here is how that stops being true.
"""

import pytest

from rag_contract.cli import DEFAULT_PORT, build_parser, cmd_serve, main


def test_it_binds_the_documented_port_by_default(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    assert build_parser().parse_args(["serve"]).port == DEFAULT_PORT


def test_a_platform_assigned_port_is_honoured(monkeypatch):
    """The one thing the environment decides, because it is not behaviour."""
    monkeypatch.setenv("PORT", "8080")
    assert build_parser().parse_args(["serve"]).port == 8080


def test_an_explicit_port_beats_the_environment(monkeypatch):
    monkeypatch.setenv("PORT", "8080")
    assert build_parser().parse_args(["serve", "--port", "9001"]).port == 9001


@pytest.mark.parametrize("value", ["http", "0", "70000", "-1"])
def test_a_port_that_is_not_a_port_stops_the_process(monkeypatch, value):
    """Refused at startup rather than defaulted around.

    Falling back to 8000 when the platform said something else would produce a
    process that starts, passes its own health check and is unreachable.
    """
    monkeypatch.setenv("PORT", value)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["serve"])


def test_an_empty_port_is_an_unset_one(monkeypatch):
    monkeypatch.setenv("PORT", "")
    assert build_parser().parse_args(["serve"]).port == DEFAULT_PORT


def test_it_binds_loopback_unless_told_otherwise(monkeypatch):
    """A container passes 0.0.0.0; a developer should not have to think about it."""
    monkeypatch.delenv("PORT", raising=False)
    assert build_parser().parse_args(["serve"]).host == "127.0.0.1"
    assert build_parser().parse_args(["serve", "--host", "0.0.0.0"]).host == "0.0.0.0"


def test_serve_runs_the_committed_application(monkeypatch):
    """It hands uvicorn an app it did not configure.

    The service, the allowance and the hop count all come from
    `create_app`, so a process started here and a process started by a platform
    are the same process.
    """
    started = {}

    def fake_run(app, **kwargs):
        started["app"] = app
        started["kwargs"] = kwargs

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.delenv("PORT", raising=False)
    assert main(["serve", "--host", "0.0.0.0", "--port", "8081"]) == 0
    assert started["kwargs"]["host"] == "0.0.0.0"
    assert started["kwargs"]["port"] == 8081

    from rag_contract.gate import load_thresholds

    committed = load_thresholds().budget
    assert started["app"].state.trusted_proxy_hops == committed.trusted_proxy_hops
    assert started["app"].state.limiter.burst == committed.max_client_burst


def test_serve_needs_no_key(monkeypatch):
    """The served path spends nothing, so nothing here may ask for a key."""
    started = {}
    import uvicorn

    monkeypatch.setattr(
        uvicorn, "run", lambda app, **kw: started.setdefault("app", app)
    )
    for key in [k for k in list(__import__("os").environ) if k.endswith("_API_KEY")]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("PORT", raising=False)
    assert cmd_serve(build_parser().parse_args(["serve"])) == 0
    assert started["app"] is not None
