"""Session fixtures: one runner process for the whole suite, one report.

Run with:  scripts/conformance.sh          (builds if needed, one command)
      or:  python -m pytest tests/conformance
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness import Client, Report, RunnerServer, categorize, find_runner, rss_kind  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.environ.get("RUNNER_CONFORMANCE_OUT", os.path.join(HERE, "out"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "known_gap(phase, what): documents behaviour that is currently WRONG or "
        "missing. The assertion pins today's behaviour so the gap cannot widen "
        "silently; when the named phase lands, the test must be rewritten.")


@pytest.fixture(scope="session")
def report():
    os.makedirs(OUT, exist_ok=True)
    r = Report(OUT, os.path.join(HERE, "fixtures"),
               update_fixtures=os.environ.get("RUNNER_CONFORMANCE_UPDATE") == "1")
    yield r


@pytest.fixture(scope="session")
def server(report):
    model = os.environ.get("RUNNER_TEST_MODEL", os.path.join(ROOT, "test.gguf"))
    if not os.path.exists(model):
        pytest.exit(f"model {model} not found; build it with "
                    f"`python3 scripts/make-test-model.py test.gguf`", returncode=2)
    os.makedirs(OUT, exist_ok=True)
    exe = find_runner(ROOT)
    # --gpu off keeps the harness reproducible and runnable on any machine;
    # kernel-verify.py is the gate that proves GPU == CPU, not this suite.
    srv = RunnerServer(exe, model, ctx=1024, parallel=2,
                       extra_args=["--gpu", "off"],
                       log_path=os.path.join(OUT, "runner.log"))
    with srv:
        yield srv
    path = os.path.join(OUT, "report.json")
    doc = report.write(path, srv.peak_rss_kb, rss_kind())
    print(f"\nconformance report: {path}")
    print(f"  requests={doc['totals']['requests']} "
          f"tests={doc['totals']['tests']} "
          f"failed={doc['totals']['failed']} "
          f"known_gap={doc['totals']['known_gap_tests']} "
          f"peak_rss_kb={doc['resources']['peak_rss_kb']} "
          f"({doc['resources']['peak_rss_kind']})")
    if doc["totals"]["failed"]:
        print(f"  failures by category: {doc['totals']['failures_by_category']}")


@pytest.fixture(scope="session")
def client(server, report):
    """Session-scoped: the client is stateless (one socket per request), and
    module-scoped fixtures that record a stream once need to depend on it."""
    server.assert_alive()
    return Client(server, report)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Tag every failure with its taxonomy category, in the report and on the
    terminal line, so a red run says protocol/schema/transport at a glance."""
    outcome = yield
    rep = outcome.get_result()
    if rep.when != "call":
        return
    rp = item.config._conformance_report
    if rp is None:
        return
    gap = item.get_closest_marker("known_gap") is not None
    if rep.passed:
        rp.record_outcome(item.nodeid, "passed", known_gap=gap)
    elif rep.skipped:
        rp.record_outcome(item.nodeid, "skipped", known_gap=gap)
    else:
        exc = call.excinfo.value if call.excinfo else None
        cat = categorize(exc)
        rp.record_outcome(item.nodeid, "failed", category=cat,
                          message=str(exc).splitlines()[0] if exc else None,
                          known_gap=gap)
        rep.sections.append(("conformance", f"failure category: {cat}"))


@pytest.fixture(scope="session", autouse=True)
def _bind_report(request, report):
    request.config._conformance_report = report
    yield
    request.config._conformance_report = None


def pytest_addoption(parser):
    parser.addini("dummy_conformance", "unused", default="")


def pytest_sessionstart(session):
    session.config._conformance_report = None
