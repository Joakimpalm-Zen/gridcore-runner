"""Per-run JSON report: fixtures, metrics and the failure taxonomy.

Two kinds of artifact come out of a run:

  fixtures/   normalized request/response shapes. The *committed* copies under
              tests/conformance/fixtures are the baseline; a run writes its own
              under out/fixtures and compares. Normalization strips everything
              host- or model-dependent (ids, timestamps, generated text,
              timings) so a fixture asserts protocol shape, never stub-model
              output, which varies with host FP codegen.

  report.json latency, prompt/generation speed and peak RSS per request, plus
              every test outcome tagged with its error category.
"""

import json
import os
import platform
import time

from _errors import CATEGORIES, ProtocolError

REDACTED = "<redacted>"
VOLATILE = {"id", "created", "system_fingerprint"}
# generated text and anything timing-shaped: present-and-typed is the contract,
# the value is not
VOLATILE_TEXT = {"content", "text", "reasoning_content", "embedding", "token"}
VOLATILE_NUM = {"generation_seconds", "generation_tok_s", "logprob",
                "prompt_tokens", "completion_tokens", "total_tokens",
                "prompt_cached_tokens", "prompt_eval_tokens", "index"}
# protocol constants: these string values ARE the contract and are kept verbatim.
# finish_reason is deliberately NOT here — it legitimately varies with what the
# stub model emitted on this host, so tests assert it directly instead.
STABLE_STR = {"object", "role", "type", "owned_by"}


def normalize(value, key=None):
    """Reduce a decoded response to a host-independent shape descriptor."""
    if isinstance(value, dict):
        return {k: normalize(v, k) for k, v in sorted(value.items())}
    if isinstance(value, list):
        # a numeric vector under a volatile key (an embedding) carries no shape
        # worth pinning, only a length. A *list of objects* under the same key
        # is the opposite: "content" is a string on the chat surface but the
        # block/part list on the Anthropic and Responses ones, and there that
        # list IS the contract, so it must keep being described structurally.
        if key in VOLATILE_TEXT and value and \
                all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    for v in value):
            return [f"<{len(value)} numbers>"]
        # collapse homogeneous lists to one representative shape + a count so a
        # fixture does not encode how many tokens the stub model happened to emit
        shapes = [normalize(v, key) for v in value]
        uniq = []
        for s in shapes:
            if s not in uniq:
                uniq.append(s)
        return uniq
    if key in VOLATILE or key in VOLATILE_TEXT:
        return REDACTED if value is not None else None
    if key in VOLATILE_NUM and isinstance(value, (int, float)):
        return "<number>"
    if key in STABLE_STR and isinstance(value, str):
        return value
    if isinstance(value, str):
        return REDACTED
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return "<number>"
    return REDACTED


class Report:
    def __init__(self, out_dir, fixture_dir, update_fixtures=False):
        self.out_dir = out_dir
        self.fixture_dir = fixture_dir
        self.update_fixtures = update_fixtures
        self.requests = []
        self.outcomes = []
        self.quality_notes = []
        self.started = time.time()
        os.makedirs(os.path.join(out_dir, "fixtures"), exist_ok=True)
        os.makedirs(fixture_dir, exist_ok=True)

    # -------------------------------------------------------------- fixtures
    def check_fixture(self, name, value):
        """Compare a normalized value against its committed baseline.

        With RUNNER_CONFORMANCE_UPDATE=1 the baseline is (re)written instead —
        the only supported way to change a fixture, so a drift is always an
        explicit commit."""
        shape = normalize(value)
        blob = json.dumps(shape, indent=2, sort_keys=True) + "\n"
        with open(os.path.join(self.out_dir, "fixtures", f"{name}.json"), "w",
                  encoding="utf-8") as f:
            f.write(blob)
        path = os.path.join(self.fixture_dir, f"{name}.json")
        if self.update_fixtures or not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(blob)
            return
        with open(path, encoding="utf-8") as f:
            want = f.read()
        if want != blob:
            raise ProtocolError(
                f"response shape drifted from fixture {name}.json "
                f"(re-record with RUNNER_CONFORMANCE_UPDATE=1 if intended)",
                expected=want, actual=blob)

    # --------------------------------------------------------------- metrics
    def record_request(self, rec):
        self.requests.append(rec)

    def record_outcome(self, nodeid, status, category=None, message=None,
                       known_gap=False):
        self.outcomes.append({"test": nodeid, "status": status,
                              "category": category, "message": message,
                              "known_gap": known_gap})

    def note_quality(self, test, message, **ctx):
        self.quality_notes.append({"test": test, "message": message, **ctx})

    # ---------------------------------------------------------------- output
    def write(self, path, peak_rss_kb, rss_kind):
        lat = sorted(r["latency_ms"] for r in self.requests if "latency_ms" in r)
        by_cat = {c: 0 for c in CATEGORIES}
        by_cat["unknown"] = 0
        for o in self.outcomes:
            if o["status"] == "failed" and o["category"] in by_cat:
                by_cat[o["category"]] += 1
        doc = {
            "schema": "gridcore-runner/conformance-report/1",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "duration_s": round(time.time() - self.started, 3),
            "platform": {"system": platform.system(),
                         "machine": platform.machine(),
                         "python": platform.python_version()},
            "resources": {"peak_rss_kb": peak_rss_kb, "peak_rss_kind": rss_kind},
            "totals": {
                "requests": len(self.requests),
                "tests": len(self.outcomes),
                "failed": sum(1 for o in self.outcomes if o["status"] == "failed"),
                "known_gap_tests": sum(1 for o in self.outcomes if o["known_gap"]),
                "failures_by_category": by_cat,
            },
            "latency_ms": {
                "min": lat[0] if lat else None,
                "median": lat[len(lat) // 2] if lat else None,
                "max": lat[-1] if lat else None,
            },
            "requests": self.requests,
            "outcomes": self.outcomes,
            "model_quality_notes": self.quality_notes,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")
        return doc
