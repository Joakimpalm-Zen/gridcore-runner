"""certify-envelope.py assembles a schema-valid measured-envelope manifest and
resolves the verdict from the evidence, failing closed without a reference sha."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = os.path.join(ROOT, "scripts", "certify-envelope.py")
MODEL = os.environ.get("RUNNER_TEST_MODEL", os.path.join(ROOT, "test.gguf"))
RUNNER = os.path.join(ROOT, "runner")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fixture_report(model_sha, checks):
    return {
        "schema_version": "xyntetik.runner.model-compat-report.v1",
        "models": [{
            "id": "fixture", "architecture": "llama",
            "file": "models/fixture.gguf", "sha256": model_sha,
            "checks": {k: {"status": v} for k, v in checks.items()},
        }],
    }


@unittest.skipUnless(os.path.exists(RUNNER) and os.path.exists(MODEL),
                     "needs a built runner and test.gguf")
class CertifyEnvelopeTests(unittest.TestCase):
    def _run(self, *args):
        p = subprocess.run([sys.executable, SCRIPT, "--model", MODEL,
                            "--runner", RUNNER, *args],
                           text=True, capture_output=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        return json.loads(p.stdout)

    def test_manifest_shape_and_runtime_identity(self):
        m = self._run()
        for key in ("schema_version", "artifact", "runtime", "hardware",
                    "config", "quality", "verdict", "measured"):
            self.assertIn(key, m)
        self.assertEqual(m["artifact"]["sha256"], sha256_file(MODEL))
        # The default measurement is --gpu off, so even a GPU host records the
        # CPU kernel set and never attaches a GPU shader identity.
        self.assertIn("kernel_set", m["runtime"])
        self.assertEqual(m["runtime"]["kernel_set"]["backend"], "cpu")
        self.assertNotIn("shader_source_sha256", m["runtime"]["kernel_set"])

    def test_gpu_auto_uses_the_available_backend(self):
        caps = json.loads(subprocess.run(
            [RUNNER, "--caps"], cwd=ROOT, check=True,
            stdout=subprocess.PIPE, text=True).stdout)
        expected = (caps.get("gpu") or {}).get("backend", "cpu")
        m = self._run("--gpu", "auto")
        self.assertEqual(m["runtime"]["kernel_set"]["backend"], expected)

    def test_passing_evidence_plus_reference_is_certified(self):
        sha = sha256_file(MODEL)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(fixture_report(sha, {"load": "pass", "tokenizer": "pass"}), f)
            report = f.name
        try:
            m = self._run("--compat-report", report, "--reference-sha", "abc123")
            self.assertEqual(m["quality"]["gate"], "pass")
            self.assertEqual(m["verdict"], "certified")
        finally:
            os.unlink(report)

    def test_runs_but_a_gate_fails_is_experimental(self):
        # Sharpened semantics (2026-08-20): a model that LOADS + runs but fails a
        # stricter gate is USABLE -> experimental, not refused. outside-envelope
        # is reserved for won't-run (see below).
        sha = sha256_file(MODEL)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(fixture_report(sha, {"load": "pass", "tokenizer": "fail"}), f)
            report = f.name
        try:
            m = self._run("--compat-report", report, "--reference-sha", "abc123")
            self.assertEqual(m["verdict"], "experimental")
        finally:
            os.unlink(report)

    def test_load_failure_is_outside_envelope(self):
        # A model that won't LOAD should not run -> outside-envelope (refuse).
        sha = sha256_file(MODEL)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(fixture_report(sha, {"load": "fail"}), f)
            report = f.name
        try:
            m = self._run("--compat-report", report, "--reference-sha", "abc123")
            self.assertEqual(m["verdict"], "outside-envelope")
        finally:
            os.unlink(report)

    def test_pass_without_reference_fails_closed_to_experimental(self):
        # A quality number relative to no named reference is not a measurement.
        sha = sha256_file(MODEL)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(fixture_report(sha, {"load": "pass"}), f)
            report = f.name
        try:
            m = self._run("--compat-report", report)  # no --reference-sha
            self.assertEqual(m["quality"]["gate"], "pass")
            self.assertEqual(m["verdict"], "experimental")
        finally:
            os.unlink(report)

    def test_no_evidence_is_experimental(self):
        m = self._run("--reference-sha", "abc123")  # no compat report
        self.assertIsNone(m["quality"]["gate"])
        self.assertEqual(m["verdict"], "experimental")

    # --- tool-calling axis (reported-only) -----------------------------------

    def _write(self, obj):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(obj, f)
        f.close()
        return f.name

    def test_tool_calling_axis_assembled_and_reported_only(self):
        sha = sha256_file(MODEL)
        compat = {
            "schema_version": "xyntetik.runner.model-compat-report.v1",
            "models": [{
                "id": "fixture", "architecture": "llama",
                "file": "models/fixture.gguf", "sha256": sha,
                "checks": {
                    "load": {"status": "pass"},
                    "tokenizer": {"status": "pass"},
                    "tool": {"status": "pass",
                             "totals": {"requests": 120, "passed": 120,
                                        "failed": 0}},
                },
            }],
        }
        trunc = {
            "schema_version": "xyntetik.truncation-benchmark.v1",
            "runner_property": {"holds": True, "violations": []},
            "configuration": {"model": "granite-4.1-3b"},
            "runtime": {"version": "0.1.20-alpha"},
            "rungs": [
                {"max_tokens": 1, "tool_calls_present": True,
                 "arguments_parseable": True},
                {"max_tokens": 8, "tool_calls_present": True,
                 "arguments_parseable": True},
                {"max_tokens": 64, "tool_calls_present": True,
                 "arguments_parseable": True},   # control, excluded
            ],
        }
        fidelity = {
            "schema_version": "xyntetik.quant-fidelity.v1",
            "model": "granite-4.1-3b",
            "variants": [{"label": "Q4_0", "quant": "Q4_0", "tool_fidelity": {
                "schema_conformance": {"rate": 1.0},
                "tool_selection": {"rate": 1.0},
                "argument_agreement": {"rate": 0.5}}}],
        }
        cp, tp, fp = self._write(compat), self._write(trunc), self._write(fidelity)
        try:
            m = self._run("--compat-report", cp, "--reference-sha", "abc123",
                          "--truncation-report", tp, "--quant-fidelity-report", fp)
        finally:
            for p in (cp, tp, fp):
                os.unlink(p)
        tc = m["tool_calling"]
        # two truncated rungs (1, 8) recover; the control (64) is excluded
        self.assertEqual(tc["truncation_recovery"]["holds"], True)
        self.assertEqual(tc["truncation_recovery"]["rungs_passed"], 2)
        self.assertEqual(tc["truncation_recovery"]["rungs_total"], 2)
        self.assertEqual(tc["truncation_recovery"]["measured"]["scope"], "engine")
        self.assertEqual(tc["schema_shape"]["held_to_quant"], "Q4_0")
        self.assertEqual(tc["schema_shape"]["schema_conformance_rate"], 1.0)
        self.assertEqual(tc["agent_torture"]["gate"], "pass")
        self.assertEqual(tc["agent_torture"]["passed"], 120)
        # rollup gate is pass on the three guarantee blocks (native does not gate)
        self.assertEqual(tc["gate"], "pass")
        # REPORTED-ONLY: the fidelity verdict is unchanged by the axis.
        self.assertEqual(m["quality"]["gate"], "pass")
        self.assertEqual(m["verdict"], "certified")

    def test_tool_calling_partial_when_guarantee_evidence_missing(self):
        trunc = {
            "schema_version": "xyntetik.truncation-benchmark.v1",
            "runner_property": {"holds": True},
            "rungs": [
                {"max_tokens": 1, "tool_calls_present": True,
                 "arguments_parseable": True},
                {"max_tokens": 64, "tool_calls_present": True,
                 "arguments_parseable": True},
            ],
        }
        tp = self._write(trunc)
        try:
            m = self._run("--reference-sha", "abc123", "--truncation-report", tp)
        finally:
            os.unlink(tp)
        tc = m["tool_calling"]
        self.assertEqual(tc["truncation_recovery"]["holds"], True)
        self.assertIsNone(tc["schema_shape"])
        self.assertIsNone(tc["agent_torture"])
        # a guarantee block is missing -> partial, never pass
        self.assertEqual(tc["gate"], "partial")

    def test_no_report_evidence_makes_no_guarantee_claim(self):
        # native protocol may still be reported (it is per-model truth), but with
        # no truncation/fidelity/tool evidence NO guarantee block may appear and
        # the fidelity verdict must be untouched.
        m = self._run("--reference-sha", "abc123")
        self.assertEqual(m["verdict"], "experimental")
        tc = m.get("tool_calling")
        if tc is not None:
            self.assertIsNone(tc.get("truncation_recovery"))
            self.assertIsNone(tc.get("schema_shape"))
            self.assertIsNone(tc.get("agent_torture"))
            self.assertNotEqual(tc.get("gate"), "pass")


if __name__ == "__main__":
    unittest.main()
