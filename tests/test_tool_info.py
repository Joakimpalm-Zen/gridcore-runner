"""`runner --tool-info -m MODEL` prints, on one JSON line, the native tool-call
protocol the runtime speaks for that model, resolved from its chat template.

The family and the native/generic flag are derived from the template layer's own
truth (template.c), not a static arch->family table, so this drives a few
templates through the flag and pins what each reports. The value is per-model
(a chat-template property), which is exactly why it cannot ride on --caps.
"""
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")


@pytest.fixture
def model(tmp_path):
    p = tmp_path / "tool-info.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(p)],
                   check=True, cwd=ROOT)
    return p


def _tool_info(model, *extra):
    proc = subprocess.run(
        [RUNNER, "--tool-info", "-m", str(model), "--gpu", "off", *extra],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    # stdout is JSON-only: exactly one line, and it parses.
    line = proc.stdout.decode().strip()
    assert "\n" not in line, "stdout must be a single JSON line: " + repr(line)
    return json.loads(line)


def test_reports_family_and_native_flag(model):
    if not RUNNER.exists():
        pytest.skip("runner not built")
    info = _tool_info(model)
    # The keys are always present and correctly typed for the cert axis.
    assert isinstance(info["tool_family"], str) and info["tool_family"]
    assert isinstance(info["native_tool_protocol"], bool)
    # The stock fixture is a plain llama GGUF with no native tool template, so it
    # resolves to the generic marker/envelope fallback.
    assert info["tool_family"] == "generic"
    assert info["native_tool_protocol"] is False


@pytest.mark.parametrize("template,family", [
    ("muse", "atem"),
    ("harmony", "harmony"),
    ("gemma4", "gemma4"),
    ("ornith", "qwen3_xml"),
])
def test_native_families_report_native(model, template, family):
    if not RUNNER.exists():
        pytest.skip("runner not built")
    info = _tool_info(model, "--chat-template", template)
    assert info["tool_family"] == family, info
    assert info["native_tool_protocol"] is True, info


def test_generic_family_is_not_native(model):
    if not RUNNER.exists():
        pytest.skip("runner not built")
    info = _tool_info(model, "--chat-template", "chatml")
    assert info["tool_family"] == "generic"
    assert info["native_tool_protocol"] is False
