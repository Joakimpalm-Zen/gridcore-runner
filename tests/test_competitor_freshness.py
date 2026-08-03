import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "competitor_freshness", ROOT / "scripts" / "competitor-freshness.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def _report(root, directory, runtime, version):
    path = root / directory / "report.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "runtime": {"name": runtime, "version": version}}))
    return path


def test_uses_newest_published_row_and_ignores_runner(tmp_path):
    _report(tmp_path, "old", "vllm", "0.25.0")
    newest = _report(tmp_path, "new", "vllm", "0.26.0 (--tool-call-parser hermes)")
    _report(tmp_path, "own", "runner", "runner 0.1.5-alpha")
    rows = MOD.read_published(tmp_path)
    assert rows == {"vllm": {"version": "0.26.0", "reports": [str(newest)]}}


def test_reports_stale_versions_from_official_registry_payloads(tmp_path):
    _report(tmp_path, "llama", "llama.cpp", "b10076")
    _report(tmp_path, "ollama", "ollama", "0.32.1")
    _report(tmp_path, "vllm", "vllm", "0.26.0")
    payloads = {
        MOD.REGISTRIES["llama.cpp"]: {"tag_name": "b10077"},
        MOD.REGISTRIES["ollama"]: {"tag_name": "v0.32.1"},
        MOD.REGISTRIES["vllm"]: {"info": {"version": "0.27.0"}},
    }
    result = MOD.check_freshness(
        MOD.read_published(tmp_path), lambda url: payloads[url])
    assert [(row["runtime"], row["published"], row["upstream"])
            for row in result["stale"]] == [
                ("llama.cpp", "b10076", "b10077"),
                ("vllm", "0.26.0", "0.27.0")]
    assert [row["runtime"] for row in result["current"]] == ["ollama"]
    assert result["skipped"] == []


def test_unreachable_registry_is_a_skip_not_stale(tmp_path):
    _report(tmp_path, "ollama", "ollama", "0.32.1")

    def unavailable(_url):
        raise OSError("temporary DNS failure")

    result = MOD.check_freshness(MOD.read_published(tmp_path), unavailable)
    assert result["stale"] == []
    assert result["current"] == []
    assert result["skipped"][0]["runtime"] == "ollama"
    assert "temporary DNS failure" in result["skipped"][0]["reason"]
