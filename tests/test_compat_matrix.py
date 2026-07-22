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
