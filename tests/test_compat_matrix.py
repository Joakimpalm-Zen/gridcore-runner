import importlib.util
import json
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
    assert data["schema_version"] == "gridcore.runner.model-compat.v1"
    models = data["models"]
    assert {m["architecture"] for m in models} == {
        "llama", "qwen2", "qwen3", "qwen35", "phi3", "gemma3", "gemma4"
    }
    assert len({m["id"] for m in models}) == len(models)
    for model in models:
        assert len(model["sha256"]) == 64
        int(model["sha256"], 16)
        assert model["reference"]["implementation"] == "llama.cpp"
        assert model["reference"]["revision"]
        assert model["checks"]


def test_manifest_validation_rejects_duplicate_ids(tmp_path):
    module = load_module()
    manifest = {
        "schema_version": "gridcore.runner.model-compat.v1",
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
    assert data["schema_version"] == "gridcore.runner.consumer-compat.v1"
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


# --- RNR-007: the gate refuses to mark unexecuted checks complete ----------

def _write_manifest(tmp_path, checks):
    model_file = tmp_path / "m.gguf"
    model_file.write_bytes(b"not a real model")
    import hashlib
    digest = hashlib.sha256(model_file.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "gridcore.runner.model-compat.v1",
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
