"""Process-level lifecycle contracts for the serving runtime."""

import os
import signal
import subprocess
import sys
import time
import json
import urllib.error
import urllib.request

import pytest

from harness import RunnerServer, find_runner, free_port


def test_server_ignore_eos_reaches_each_slot_engine(tmp_path):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model = tmp_path / "eos-only.gguf"
    subprocess.run([sys.executable, os.path.join(root, "scripts", "make-test-model.py"),
                    "--suppress-all-but-eos", str(model)], check=True, cwd=root)
    with RunnerServer(find_runner(root), str(model), ctx=64, parallel=1,
                      extra_args=["--gpu", "off", "--ignore-eos"]) as srv:
        payload = json.dumps({
            "messages": [{"role": "user", "content": "keep going"}],
            "max_tokens": 20, "temperature": 0,
            "cache_prompt": False,
        }).encode()
        def complete():
            request = urllib.request.Request(
                srv.base_url + "/v1/chat/completions", data=payload,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.load(response)

        def assert_ignored():
            body = complete()
            assert body["usage"]["completion_tokens"] == 20
            assert body["choices"][0]["finish_reason"] == "length"

        assert_ignored()
        # A one-slot server can unload/reload its sole model. engine_init()
        # runs again on reload and must not silently erase the process flag.
        with urllib.request.urlopen(srv.base_url + "/unload", data=b"",
                                    timeout=10) as response:
            assert json.load(response) == {"status": "ok"}
        assert_ignored()


def test_agent_profile_is_admitted_and_exposed(tmp_path):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model = tmp_path / "profile.gguf"
    subprocess.run([sys.executable, os.path.join(root, "scripts", "make-test-model.py"),
                    "--agent-profile", str(model)], check=True, cwd=root)
    cli = subprocess.run([find_runner(root), "-m", str(model), "-p", "probe",
                          "-n", "1", "--gpu", "off"], text=True,
                         capture_output=True, timeout=10)
    assert cli.returncode == 0
    assert "agent-profile: protocol=1 tokenizer=1" in cli.stderr
    srv = RunnerServer(find_runner(root), str(model), ctx=64, parallel=1,
                       extra_args=["--gpu", "off"])
    srv.start()
    try:
        with urllib.request.urlopen(srv.base_url + "/v1/capabilities", timeout=5) as r:
            profile = json.load(r)["agent_profile"]
        assert profile == {
            "protocol_version": 1,
            "tokenizer_version": 1,
            "schema_id": "gridcore.agent.action.v1",
            "schema_digest": "a" * 64,
            "required_features": ["dense", "json_schema"],
        }
    finally:
        srv.stop()


def test_unknown_required_agent_feature_fails_before_serving(tmp_path):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model = tmp_path / "future-profile.gguf"
    subprocess.run([sys.executable, os.path.join(root, "scripts", "make-test-model.py"),
                    "--agent-feature", "future_attention_v99", str(model)],
                   check=True, cwd=root)
    proc = subprocess.run([find_runner(root), "-m", str(model), "-p", "probe",
                           "-n", "1", "--gpu", "off"], text=True,
                          capture_output=True, timeout=10)
    assert proc.returncode != 0
    assert "unsupported required agent feature 'future_attention_v99'" in proc.stderr
    assert "loaded " not in proc.stderr


def test_thinking_prelude_is_bounded_with_distinct_finish_reason(tmp_path):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model = tmp_path / "ornith.gguf"
    fixture_env = os.environ.copy()
    fixture_env["ORNITH_TEST_SEED"] = "1"  # deterministically emits prelude whitespace
    subprocess.run([sys.executable, os.path.join(root, "scripts", "make-test-ornith.py"),
                    str(model)], check=True, cwd=root, env=fixture_env)
    # ctx=256, not the 64 the neighbouring tests use, and the difference is
    # not padding. This test ASKS for the ornith template, and until runner
    # 88837e8 `--chat-template` was parsed and then silently discarded under
    # --serve -- so it actually ran against the fixture's auto-detected
    # fallback, whose render of this conversation is 34 tokens. The ornith
    # template renders the same conversation in 73. The flag works now, the
    # prompt the test asked for no longer fits in 64, and the request 400s
    # with context_length_exceeded before reaching the behaviour under test.
    # The number was calibrated against a bug; this is what it should have
    # been all along.
    srv = RunnerServer(find_runner(root), str(model), ctx=256, parallel=1,
                       extra_args=["--gpu", "off", "--chat-template", "ornith"])
    srv.start()
    try:
        cap = 8
        payload = json.dumps({
            "messages": [{"role": "user", "content": "emit json"}],
            "max_tokens": cap,
            "temperature": 0,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "answer", "schema": {"type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"]}}},
        }).encode()
        req = urllib.request.Request(srv.base_url + "/v1/chat/completions",
                                     data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as response:
            body = json.load(response)
        choice = body["choices"][0]
        # The prelude stays bounded and keeps its own finish reason. What
        # changed on 2026-08-07: hitting the cap used to END the turn, so a
        # caller who asked for schema-constrained JSON received an empty
        # document. It now closes the prelude and spends the rest of the
        # budget on the payload that was actually requested.
        #
        # This test previously asserted `completion_tokens <= cap // 2` and
        # empty content -- i.e. it pinned "stop and return nothing". Both are
        # replaced by the stronger guarantee below. The behaviour it protected
        # was measured on a real model (e4b-q4km, gemma4, whose prelude tags
        # are <|channel>thought / <channel|>): two of four tool prompts opened
        # a thinking block, never closed it, and returned ONE byte from 100
        # generated tokens. With this change all four return valid calls.
        # The wire value is the OpenAI-standard "length" as of 2026-08-08 —
        # `reasoning_limit` is not in the OpenAI enum and broke typed clients.
        # The distinction it carried is not lost, it moved to an extension
        # field, so this asserts BOTH halves: standards-compliant on the wire,
        # and the prelude-specific reason still recoverable.
        assert choice["finish_reason"] == "length"
        assert body["runner_telemetry"]["finish_detail"] == "reasoning_limit"
        assert body["usage"]["completion_tokens"] <= cap
        payload = json.loads(choice["message"]["content"])   # must be JSON now
        assert "answer" in payload                           # and match the schema
    finally:
        srv.stop()


def test_thinking_prelude_cap_does_not_apply_without_a_constraint(tmp_path):
    """The other half of the prelude contract, ratified by the owner 2026-08-15.

    Runner is deliberately ASYMMETRIC, and both halves are load-bearing:

      * WITH a constraint (json_schema / json_object / tools) the prelude is
        bounded at max_new/2, and hitting that bound CLOSES the prelude and
        spends the rest of the budget on the payload, because the caller is
        owed a document. Pinned by the two tests above. That is the shape
        Anthropic's extended thinking uses -- thinking gets its own budget
        under max_tokens and the answer is still produced.
      * WITHOUT one there is NO prelude bound at all. The turn runs to
        max_tokens like any other, `finish_reason` is the plain "length", and
        no `finish_detail` appears because the prelude cap never fired. That
        resembles OpenAI's reasoning models, where reasoning and output share
        the ceiling and a reasoning-heavy turn can return little or nothing.

    This second half had nothing pinning it, and the gap was not harmless: the
    engine carried an unreachable `else` branch that ENDED the turn at the
    bound, and on 2026-08-15 it was read as the unconstrained policy and
    described to the owner as one. The branch is gone; this test is what stops
    the question being re-answered from a code path nobody can reach.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model = tmp_path / "ornith.gguf"
    fixture_env = os.environ.copy()
    fixture_env["ORNITH_TEST_SEED"] = "1"
    subprocess.run([sys.executable, os.path.join(root, "scripts", "make-test-ornith.py"),
                    str(model)], check=True, cwd=root, env=fixture_env)
    # ctx=256 for the reason spelled out in the two tests above: this forces
    # the ornith template, whose render of the conversation does not fit in 64.
    srv = RunnerServer(find_runner(root), str(model), ctx=256, parallel=1,
                       extra_args=["--gpu", "off", "--chat-template", "ornith"])
    srv.start()
    try:
        payload = json.dumps({
            "messages": [{"role": "user", "content": "emit json"}],
            "max_tokens": 8,
            "temperature": 0,
            # NO response_format and NO tools -- that is the whole point
        }).encode()
        req = urllib.request.Request(srv.base_url + "/v1/chat/completions",
                                     data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as response:
            body = json.load(response)
        choice = body["choices"][0]
        # the budget ran out, and that is all that happened
        assert choice["finish_reason"] == "length"
        # the prelude bound did NOT fire: no constraint, so it is not armed
        assert "finish_detail" not in body.get("runner_telemetry", {}), (
            "an unconstrained turn reported a prelude-specific finish reason; "
            "the cap is supposed to be armed only under a constraint")
        # and the whole budget was available to it, not half
        assert body["usage"]["completion_tokens"] == 8
    finally:
        srv.stop()


def test_thinking_prelude_reason_survives_streaming(tmp_path):
    """A streamed turn must not lose the reason a buffered one keeps.

    openai_finish() widens `reasoning_limit` to the standard `length` on the
    wire, and the buffered path preserves the distinction in
    runner_telemetry.finish_detail. Streamed chat carries no runner_telemetry
    at all, so until 2026-08-09 the same turn was fully explained when
    buffered and unexplained when streamed -- and a client that streams is
    exactly the one watching tokens arrive and wondering why they stopped.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model = tmp_path / "ornith.gguf"
    fixture_env = os.environ.copy()
    fixture_env["ORNITH_TEST_SEED"] = "1"
    subprocess.run([sys.executable, os.path.join(root, "scripts", "make-test-ornith.py"),
                    str(model)], check=True, cwd=root, env=fixture_env)
    # ctx=256 for the reason spelled out in
    # test_thinking_prelude_is_bounded_with_distinct_finish_reason: the ornith
    # template this test forces renders 73 prompt tokens, not the 34 of the
    # fallback it silently got while --chat-template was ignored under --serve.
    srv = RunnerServer(find_runner(root), str(model), ctx=256, parallel=1,
                       extra_args=["--gpu", "off", "--chat-template", "ornith"])
    srv.start()
    try:
        payload = json.dumps({
            "messages": [{"role": "user", "content": "emit json"}],
            "max_tokens": 8,
            "temperature": 0,
            "stream": True,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "answer", "schema": {"type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"]}}},
        }).encode()
        req = urllib.request.Request(srv.base_url + "/v1/chat/completions",
                                     data=payload,
                                     headers={"Content-Type": "application/json"})
        terminal = None
        with urllib.request.urlopen(req, timeout=10) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                blob = line[len("data: "):]
                if blob == "[DONE]":
                    break
                chunk = json.loads(blob)
                if chunk["choices"] and chunk["choices"][0].get("finish_reason"):
                    terminal = chunk
        assert terminal is not None, "stream ended without a finish_reason chunk"
        # standards-compliant on the wire ...
        assert terminal["choices"][0]["finish_reason"] == "length"
        # ... and the prelude-specific reason still recoverable, which is the
        # half that did not exist for streamed turns before
        assert terminal["runner_telemetry"]["finish_detail"] == "reasoning_limit"
    finally:
        srv.stop()


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal contract")
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_signal_performs_clean_shutdown(sig):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model = os.environ.get("RUNNER_TEST_MODEL", os.path.join(root, "test.gguf"))
    srv = RunnerServer(find_runner(root), model, ctx=1024, parallel=2,
                       extra_args=["--gpu", "off"])
    srv.start()
    try:
        srv.proc.send_signal(sig)
        assert srv.proc.wait(timeout=10) == 0
    finally:
        srv.stop()


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal contract")
def test_signal_during_startup_aborts_startup():
    """A SIGTERM that lands while models are still loading must abort startup.

    The one wrong outcome is a survivor: a signal swallowed mid-load used to
    leave the half-started server running forever, because the stop flag was
    only checked after accept() failed. The delays sweep the load window; a
    signal landing before the handlers install kills by default disposition,
    which is an equally acceptable exit."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model = os.environ.get("RUNNER_TEST_MODEL", os.path.join(root, "test.gguf"))
    exe = find_runner(root)
    # Output is captured rather than discarded so that a survivor is
    # DIAGNOSABLE. One was reported on 2026-08-02 at the 2 ms delay and
    # estimated at "~1 in 20 runs" from that single sighting; it left no
    # record of how far startup had got, which is most of why it could not be
    # chased afterwards. It has not been reproduced since — 960 spawns across
    # idle and loaded boxes and two builds, plus 20 whole-suite runs, all
    # clean — and a rate cannot be established from one occurrence anyway.
    # The next one will at least say where it was.
    for delay in [0.0003, 0.0005, 0.0008, 0.001, 0.002, 0.005] * 2:
        proc = subprocess.Popen(
            [exe, "-m", model, "--serve", "--port", str(free_port()),
             "--parallel", "2", "-c", "1024", "--gpu", "off"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            time.sleep(delay)
            proc.send_signal(signal.SIGTERM)
            try:
                rc = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                out = proc.stdout.read().decode("utf-8", "replace")
                pytest.fail(
                    "server survived a SIGTERM sent %.1fms after spawn.\n"
                    "startup output was:\n%s"
                    % (delay * 1000, out or "  (nothing — it had not got as "
                       "far as printing anything)"))
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stdout.close()
        assert rc in (0, -signal.SIGTERM)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal contract")
def test_second_signal_exits_immediately():
    """A second SIGINT during the shutdown drain must exit now, with 130.

    The drain waits for in-flight work, and a stalled client pins it for the
    full request-read budget; the operator's second Ctrl-C used to find the
    listener already closed, do nothing, and leave only SIGKILL. The stalled
    connection below holds a slot in its read loop to keep the drain open
    while the second signal lands."""
    import socket
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model = os.environ.get("RUNNER_TEST_MODEL", os.path.join(root, "test.gguf"))
    srv = RunnerServer(find_runner(root), model, ctx=1024, parallel=2,
                       extra_args=["--gpu", "off"])
    srv.start()
    stalled = None
    try:
        stalled = socket.create_connection(("127.0.0.1", srv.port), timeout=15)
        stalled.sendall(b"POST /v1/chat/completions HTTP/1.1\r\n")  # never finishes
        time.sleep(0.5)                    # let a slot enter its read loop
        srv.proc.send_signal(signal.SIGINT)
        time.sleep(0.3)
        assert srv.proc.poll() is None, "drain should still be pinned by the stalled client"
        srv.proc.send_signal(signal.SIGINT)
        assert srv.proc.wait(timeout=3) == 130
    finally:
        if stalled is not None:
            stalled.close()
        srv.stop()


def _chat(base_url, **kw):
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}],
                       "max_tokens": 8, **kw}).encode()
    req = urllib.request.Request(base_url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def test_draft_model_survives_unload():
    """A --draft configured at startup must come back with the next reload.

    /unload frees the draft with the target; the lazy reload used to re-attach
    only a draft that still existed, so one /unload disabled speculative
    decoding for the life of the server."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model = os.environ.get("RUNNER_TEST_MODEL", os.path.join(root, "test.gguf"))
    with RunnerServer(find_runner(root), model, ctx=1024, parallel=1,
                      extra_args=["--gpu", "off", "--draft", model]) as srv:
        assert _chat(srv.base_url)["runner_telemetry"]["speculative"] is True
        with urllib.request.urlopen(srv.base_url + "/unload", data=b"", timeout=5) as r:
            assert json.load(r) == {"status": "ok"}
        assert _chat(srv.base_url)["runner_telemetry"]["speculative"] is True


def test_unload_answers_promptly_and_defers_under_active_generation():
    """POST /unload during a generation must answer now and free at the boundary.

    /unload used to queue behind the busy slot, so an operator reclaiming
    memory waited out whatever generation (or --wait-for-vram load) was in
    flight — exactly the moment /unload matters. It is now answered from the
    accept loop: a busy server records the wish, leaves the in-flight work
    untouched, and unloads when the last request finishes. The stalled reader
    below pins the generation mid-stream to hold that window open."""
    import socket
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model = os.environ.get("RUNNER_TEST_MODEL", os.path.join(root, "test.gguf"))
    with RunnerServer(find_runner(root), model, ctx=1024, parallel=1,
                      extra_args=["--gpu", "off"]) as srv:
        conn = socket.create_connection(("127.0.0.1", srv.port), timeout=30)
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}],
                           "max_tokens": 900, "stream": True}).encode()
        conn.sendall(b"POST /v1/chat/completions HTTP/1.1\r\n"
                     b"Host: localhost\r\n"
                     b"Content-Type: application/json\r\n" +
                     ("Content-Length: %d\r\n\r\n" % len(body)).encode() + body)
        assert conn.recv(1), "generation never started"   # stream is live
        started = time.time()
        with urllib.request.urlopen(srv.base_url + "/unload", data=b"", timeout=5) as r:
            payload = json.load(r)
        assert payload["status"] == "ok"
        assert time.time() - started < 2.0, "/unload must not queue behind the busy slot"
        # the in-flight generation is untouched: drain it to its normal end
        conn.settimeout(60)
        data = b""
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            data += chunk
        conn.close()
        assert b"finish_reason" in data or b"[DONE]" in data, \
            "the deferred unload must not cut the active stream short"
        # and the model is gone at the request boundary
        deadline = time.time() + 10
        resident = "unchecked"
        while time.time() < deadline:
            with urllib.request.urlopen(srv.base_url + "/v1/capabilities",
                                        timeout=5) as r:
                resident = json.load(r)["resident"]
            if resident is None:
                break
            time.sleep(0.2)
        assert resident is None, "pending unload must fire once the slot is idle"


def test_unload_clears_reported_model_context():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model = os.environ.get("RUNNER_TEST_MODEL", os.path.join(root, "test.gguf"))
    with RunnerServer(find_runner(root), model, ctx=1024, parallel=1,
                      extra_args=["--gpu", "off"]) as srv:
        with urllib.request.urlopen(srv.base_url + "/unload", data=b"", timeout=5) as r:
            assert json.load(r) == {"status": "ok"}
        with urllib.request.urlopen(srv.base_url + "/v1/capabilities", timeout=5) as r:
            caps = json.load(r)
        assert caps["resident"] is None
        assert caps["context"] == 0


def test_unload_rejects_get_so_a_web_page_cannot_free_the_model():
    """The drive-by vector, closed.

    /unload used to be a GET, which any page the user happened to be visiting
    could fire with `<img src="http://127.0.0.1:PORT/unload">` — no preflight,
    no CORS, and no DNS rebinding needed, because binding to loopback does not
    stop a browser. Verified against the old build: GET returned 200 and the
    server logged a real unload.

    A POST is not a CORS simple request unless its Content-Type says so, so
    requiring one restores the preflight. The refusal is a 405 naming the
    method rather than a 404, so an operator's existing script says what
    changed instead of looking like a missing route.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model = os.environ.get("RUNNER_TEST_MODEL", os.path.join(root, "test.gguf"))
    with RunnerServer(find_runner(root), model, ctx=256, parallel=1,
                      extra_args=["--gpu", "off"]) as srv:
        try:
            with urllib.request.urlopen(srv.base_url + "/unload", timeout=5):
                raise AssertionError("GET /unload was accepted")
        except urllib.error.HTTPError as e:
            assert e.code == 405, f"expected 405, got {e.code}"
            assert b"POST" in e.read(), "the refusal does not name the method"
        # and the endpoint still works the supported way
        with urllib.request.urlopen(srv.base_url + "/unload", data=b"",
                                    timeout=5) as r:
            assert json.load(r) == {"status": "ok"}
