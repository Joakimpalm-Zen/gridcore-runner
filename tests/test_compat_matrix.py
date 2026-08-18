import importlib.util
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "compatibility" / "models.json"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "compat_matrix", ROOT / "scripts" / "compat_matrix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_manifest_covers_every_claimed_architecture():
    data = json.loads(MANIFEST.read_text())
    assert data["schema_version"] == "xyntetik.runner.model-compat.v2"
    # No manifest-wide reference revision. Rows carry their own (asserted
    # below, and every harness reads it there); a single header claim went
    # stale the moment muse and granite pinned b10353 and trinity pinned
    # b10280 while the header still said b10076 for all 24 rows. A field
    # nothing reads and nothing can keep true is a lie with a schema.
    assert "reference_revision" not in data
    # cpu_cuda certifies at 128 generated tokens (owner decision 2026-08-08),
    # which is also cpu_cuda_check.py's own default. The ledger's params were
    # authored later and declared 64, silently re-weakening the gate every
    # time it ran; pin the decided number so a param cannot walk it back.
    for model in data["models"]:
        params = model.get("check_params", {}).get("cpu_cuda")
        if params:
            assert params.get("tokens") == 128, model["id"]
    models = data["models"]
    assert {m["architecture"] for m in models} == {
        "llama", "qwen2", "qwen3", "qwen35", "phi3", "gemma3", "gemma4",
        "qwen3moe", "gemma4-moe", "gpt-oss", "apertus", "afmoe",
        "muse-glimmer", "granite",
    }
    assert len({m["id"] for m in models}) == len(models)
    for model in models:
        assert len(model["sha256"]) == 64
        int(model["sha256"], 16)
        assert model["reference"]["implementation"] == "llama.cpp"
        assert model["reference"]["revision"]
        assert model["checks"]
        capture_path = model.get("tokenizer_reference_ids")
        if capture_path:
            capture = json.loads((ROOT / capture_path).read_text())
            assert capture["ref"] == model["tokenizer_reference"]
            assert capture["ref_revision"] == model["tokenizer_reference_revision"]
            assert len(capture["ids"]) == 721


def test_manifest_validation_rejects_duplicate_ids(tmp_path):
    module = load_module()
    manifest = {
        "schema_version": "xyntetik.runner.model-compat.v1",
        "models": [
            {"id": "same", "architecture": "llama", "file": "a.gguf",
             "sha256": "0" * 64, "checks": ["load"]},
            {"id": "same", "architecture": "qwen2", "file": "b.gguf",
             "sha256": "1" * 64, "checks": ["load"]},
        ],
    }
    path = tmp_path / "models.json"
    path.write_text(json.dumps(manifest))
    try:
        module.load_manifest(path)
    except ValueError as exc:
        assert "duplicate model id" in str(exc)
    else:
        raise AssertionError("duplicate IDs were accepted")


def test_consumer_gate_covers_requested_integration_layers():
    data = json.loads(
        (ROOT / "tests" / "compatibility" / "consumers.json").read_text())
    assert data["schema_version"] == "xyntetik.runner.consumer-compat.v1"
    assert {consumer["id"] for consumer in data["consumers"]} >= {
        "openai-python", "openai-node", "anthropic-python",
        "anthropic-node", "litellm", "langchain-openai",
    }
    for consumer in data["consumers"]:
        assert consumer["version"]
        assert consumer["surface"] in {
            "chat_completions", "responses", "messages", "gateway"
        }


def test_reference_comparator_normalizes_openai_completion_text():
    spec = importlib.util.spec_from_file_location(
        "reference_compare", ROOT / "scripts" / "reference_compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.completion_text({
        "choices": [{"text": " alpha"}], "usage": {"completion_tokens": 1}
    }) == " alpha"
    try:
        module.completion_text({"choices": []})
    except ValueError as exc:
        assert "completion text" in str(exc)
    else:
        raise AssertionError("malformed completion response was accepted")

    assert module.port_pair(9300) == (9300, 9301)
    with __import__("pytest").raises(ValueError):
        module.port_pair(9399)


# --- RNR-007: the gate refuses to mark unexecuted checks complete ----------

def _write_manifest(tmp_path, checks):
    model_file = tmp_path / "m.gguf"
    model_file.write_bytes(b"not a real model")
    import hashlib
    digest = hashlib.sha256(model_file.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "xyntetik.runner.model-compat.v1",
        "models": [{
            "id": "m", "architecture": "llama", "file": str(model_file),
            "sha256": digest, "checks": checks,
        }],
    }
    path = tmp_path / "models.json"
    path.write_text(json.dumps(manifest))
    return path


def test_declared_but_unexecuted_checks_are_recorded_not_omitted(tmp_path):
    module = load_module()
    manifest = _write_manifest(tmp_path, ["load", "tokenizer", "greedy_reference"])
    out = tmp_path / "report.json"
    # a files-only run executes none of the declared checks
    rc = module.main(["--manifest", str(manifest), "--verify-files",
                      "--out", str(out)])
    report = json.loads(out.read_text())
    model = report["models"][0]
    # every declared check is present and explicitly not_executed — never omitted
    for name in ("load", "tokenizer", "greedy_reference"):
        assert model["checks"][name]["status"] == "not_executed"
    assert model["complete"] is False
    assert report["complete"] is False
    # without --require-complete, a clean file hash is not itself a gate failure
    assert rc == 0


def test_require_complete_fails_an_incomplete_report(tmp_path):
    module = load_module()
    manifest = _write_manifest(tmp_path, ["load", "greedy_reference"])
    out = tmp_path / "report.json"
    rc = module.main(["--manifest", str(manifest), "--verify-files",
                      "--require-complete", "--out", str(out)])
    assert rc == 1, "an incomplete report must fail the gate under --require-complete"


def test_reports_are_append_only(tmp_path):
    module = load_module()
    manifest = _write_manifest(tmp_path, ["load"])
    out = tmp_path / "report.json"
    assert module.main(["--manifest", str(manifest), "--verify-files",
                        "--out", str(out)]) == 0
    # a second run to the same path must refuse rather than clobber evidence
    with __import__("pytest").raises(SystemExit):
        module.main(["--manifest", str(manifest), "--verify-files",
                     "--out", str(out)])
    # ...unless explicitly forced
    assert module.main(["--manifest", str(manifest), "--verify-files",
                        "--out", str(out), "--force"]) == 0


def _executable(path, body):
    if os.name == "nt":
        # CreateProcess cannot execute a POSIX shebang script.  Keep this an
        # integration test of the real subprocess boundary by giving Windows
        # its native no-op wrapper instead of mocking the runner calls away.
        path = path.with_suffix(".cmd")
        body = "@exit /b 0\r\n"
    else:
        body = "#!/bin/sh\n" + body
    path.write_text(body)
    path.chmod(0o755)
    return path


def test_resolves_pinned_basename_recursively_and_executes_supported_checks(tmp_path):
    module = load_module()
    model = tmp_path / "models" / "publisher" / "model.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"fixture model")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "xyntetik.runner.model-compat.v1",
        "models": [{
            "id": "m", "architecture": "llama", "file": "models/model.gguf",
            "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "tokenizer_reference": "local/tokenizer",
            "checks": ["load", "tokenizer", "greedy_reference"],
        }],
    }))
    runner = _executable(tmp_path / "runner", "exit 0\n")
    reference = _executable(tmp_path / "llama-server", "exit 0\n")
    tokenizer = tmp_path / "difftok.py"
    tokenizer.write_text("print('0/721 strings differ')\n")
    greedy = tmp_path / "reference-compare.py"
    greedy.write_text("print('{\"totals\":{\"passed\":5,\"failed\":0}}')\n")
    module.TOKENIZER_SCRIPT = tokenizer
    module.GREEDY_SCRIPT = greedy
    out = tmp_path / "report.json"
    assert module.main([
        "--manifest", str(manifest), "--models-root", str(tmp_path / "models"),
        "--runner", str(runner), "--reference", str(reference),
        "--verify-files", "--execute-checks", "--out", str(out),
    ]) == 0
    item = json.loads(out.read_text())["models"][0]
    assert item["resolved_file"] == str(model)
    assert {name: check["status"] for name, check in item["checks"].items()} == {
        "load": "pass", "tokenizer": "pass", "greedy_reference": "pass",
    }
    assert item["complete"] is True
    report = json.loads(out.read_text())
    assert report["summary"] == {
        "models": 1,
        "files": {"pass": 1},
        "checks": {
            "load": {"pass": 1},
            "tokenizer": {"pass": 1},
            "greedy_reference": {"pass": 1},
        },
    }


def test_unavailable_executable_check_records_specific_reason(tmp_path):
    module = load_module()
    manifest = _write_manifest(tmp_path, ["greedy_reference"])
    out = tmp_path / "report.json"
    assert module.main([
        "--manifest", str(manifest), "--models-root", str(tmp_path),
        "--verify-files", "--execute-checks", "--out", str(out),
    ]) == 0
    check = json.loads(out.read_text())["models"][0]["checks"]["greedy_reference"]
    assert check == {"status": "not_executed", "reason": "reference_binary_not_provided"}


def test_unavailable_tokenizer_reference_is_skip_not_failure(tmp_path):
    module = load_module()
    script = tmp_path / "difftok.py"
    script.write_text(
        "import sys\nprint('Access to model is restricted', file=sys.stderr)\nsys.exit(1)\n")
    module.TOKENIZER_SCRIPT = script
    result = module.run_tokenizer(
        tmp_path / "model.gguf", "gated/model",
        ROOT / "tests/fixtures/tokenizer-corpus.txt", 10)
    assert result["status"] == "not_executed"
    assert result["reason"] == "tokenizer_reference_unavailable"


def test_tokenizer_capture_is_replayed_without_network_reference(tmp_path):
    module = load_module()
    script = tmp_path / "difftok.py"
    script.write_text(
        "import sys\nprint('ARGS ' + ' '.join(sys.argv[1:]))\n")
    module.TOKENIZER_SCRIPT = script
    capture = tmp_path / "reference-ids.json"
    capture.write_text("{}")
    result = module.run_tokenizer(
        tmp_path / "model.gguf", "gated/model",
        ROOT / "tests/fixtures/tokenizer-corpus.txt", 10,
        reference_ids=capture)
    assert result["status"] == "pass"
    assert f"--ref-ids {capture}" in result["stdout_tail"]
    assert "--ref gated/model" not in result["stdout_tail"]


# ------------------------------------------- executable check contracts (v2)
#
# cpu_cuda, chat and tool were declared in the manifest for months and landed as
# not_executed with reason "check_class_not_executable": the ledger listed a
# contract it could not run. v2 gives each of them structured parameters, so the
# manifest states WHAT the check means for that model and the runner can execute
# it. RNR-007 honesty is preserved: a check whose parameters are missing or whose
# prerequisites are absent is still not_executed with a specific reason, never a
# silent pass and never omitted.

def _write_manifest_v2(tmp_path, checks, params=None, schema="xyntetik.runner.model-compat.v2"):
    model_file = tmp_path / "m.gguf"
    model_file.write_bytes(b"not a real model")
    digest = hashlib.sha256(model_file.read_bytes()).hexdigest()
    entry = {"id": "m", "architecture": "llama", "file": str(model_file),
             "sha256": digest, "checks": checks}
    if params is not None:
        entry["check_params"] = params
    manifest = {"schema_version": schema, "models": [entry]}
    path = tmp_path / "models.json"
    path.write_text(json.dumps(manifest))
    return path


def test_manifest_accepts_structured_check_parameters():
    data = json.loads(MANIFEST.read_text())
    assert data["schema_version"] == "xyntetik.runner.model-compat.v2"
    # every model declaring one of the parameterised classes carries params for
    # it: a declared contract with no definition is the thing v2 removes
    for m in data["models"]:
        for cls in ("cpu_cuda", "chat", "tool"):
            if cls in m.get("checks", []):
                assert cls in m.get("check_params", {}), (m["id"], cls)


def test_cpu_cuda_params_declare_pins_and_a_minimum_prompt_length():
    data = json.loads(MANIFEST.read_text())
    seen = 0
    for m in data["models"]:
        p = m.get("check_params", {}).get("cpu_cuda")
        if not p:
            continue
        seen += 1
        assert isinstance(p.get("pins"), dict) and p["pins"], m["id"]
        # the TC bug sat one token past the old 15-token maximum; a cpu_cuda
        # contract that does not reach batch>16 prefill cannot catch that class
        assert p.get("min_prompt_tokens", 0) >= 24, m["id"]
    assert seen > 0, "no model declares cpu_cuda parameters"


def test_missing_check_params_are_not_executed_with_a_specific_reason(tmp_path):
    module = load_module()
    manifest = _write_manifest_v2(tmp_path, ["cpu_cuda", "chat"], params={})
    out = tmp_path / "report.json"
    module.main(["--manifest", str(manifest), "--verify-files",
                 "--execute-checks", "--out", str(out)])
    report = json.loads(out.read_text())
    checks = report["models"][0]["checks"]
    for cls in ("cpu_cuda", "chat"):
        assert checks[cls]["status"] == "not_executed"
        assert checks[cls]["reason"] == f"{cls}_params_not_declared"


def test_v1_manifests_still_load(tmp_path):
    # the schema bump must not orphan an older manifest
    module = load_module()
    manifest = _write_manifest_v2(tmp_path, ["load"], params=None,
                                  schema="xyntetik.runner.model-compat.v1")
    out = tmp_path / "report.json"
    rc = module.main(["--manifest", str(manifest), "--verify-files",
                      "--out", str(out)])
    assert rc == 0
    assert json.loads(out.read_text())["models"][0]["file_status"] == "pass"


# The ledger is a claim about ONE binary. run_cpu_cuda and run_tool shell out
# to sibling scripts whose --runner defaults to the repo root's ./runner, so a
# runner argument they do not forward certifies whichever binary happens to be
# lying there instead of the one under test.

def _captured_run(recorded):
    class Completed:
        returncode = 0
        stdout = "9/9 identical\n"
        stderr = ""

    def run(cmd, timeout, **kwargs):
        recorded.append(cmd)
        return Completed()
    return run


def test_cpu_cuda_check_is_run_against_the_binary_under_test(monkeypatch, tmp_path):
    module = load_module()
    recorded = []
    monkeypatch.setattr(module, "run_group", _captured_run(recorded))

    module.run_cpu_cuda(tmp_path / "runner-candidate", tmp_path / "m.gguf",
                        {"tokens": 128}, 60)

    assert "--runner" in recorded[0]
    assert recorded[0][recorded[0].index("--runner") + 1] == \
        str(tmp_path / "runner-candidate")


def test_tool_matrix_is_run_against_the_binary_under_test(monkeypatch, tmp_path):
    module = load_module()
    recorded = []
    monkeypatch.setattr(module, "run_group", _captured_run(recorded))

    module.run_tool(tmp_path / "runner-candidate", tmp_path / "m.gguf",
                    {"scenario_matrix": "agent-torture"}, 60)

    assert "--runner" in recorded[0]
    assert recorded[0][recorded[0].index("--runner") + 1] == \
        str(tmp_path / "runner-candidate")


def test_a_timeout_kills_the_grandchildren_too(tmp_path):
    """The check scripts spawn `runner --serve` of their own. subprocess.run's
    timeout kills the direct child only, and the fix written for that read the
    group leader off TimeoutExpired.process -- an attribute the exception does
    not have, so it always killed nothing."""
    import os
    import signal
    import subprocess
    import time

    import pytest

    if os.name == "nt":
        pytest.skip("no process groups on Windows")

    module = load_module()
    marker = tmp_path / "grandchild.pid"
    script = tmp_path / "spawner.sh"
    script.write_text("#!/bin/sh\n"
                      "sleep 120 &\n"
                      f"echo $! > {marker}\n"
                      "sleep 120\n")
    script.chmod(0o755)

    with pytest.raises(subprocess.TimeoutExpired):
        module.run_group([str(script)], 1.0, grace=0.2, capture_output=True,
                         text=True)

    grandchild = int(marker.read_text())
    for _ in range(50):
        try:
            os.kill(grandchild, 0)
        except OSError:
            break
        time.sleep(0.1)
    else:
        os.kill(grandchild, signal.SIGKILL)
        pytest.fail(f"grandchild {grandchild} outlived the timeout")
