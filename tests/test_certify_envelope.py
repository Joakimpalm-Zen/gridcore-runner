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
        # runtime identity carries a kernel-set key (the shader sha on Metal,
        # or a flagged null on a CPU-only build) — never absent.
        self.assertIn("kernel_set", m["runtime"])
        self.assertIn("backend", m["runtime"]["kernel_set"])

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

    def test_failing_check_is_outside_envelope(self):
        sha = sha256_file(MODEL)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(fixture_report(sha, {"load": "pass", "tokenizer": "fail"}), f)
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


if __name__ == "__main__":
    unittest.main()
