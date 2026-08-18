#!/usr/bin/env python3
"""Quant-vs-tool-call-fidelity harness (suite plan P2).

Measures how tool-calling / structured-output fidelity degrades across a
quantization ladder of ONE model. For each GGUF variant it spawns Runner
(one server at a time, sequential — never two variants loaded together) and
runs the agent-torture request matrix at temperature 0, then scores three
axes against a pinned highest-precision REFERENCE variant:

  * schema conformance   — does the variant's own output validate against
                            the tool/response_format JSON schema?
  * tool selection        — does the variant call the same function the
                            reference variant called for the same request?
  * argument agreement    — do the parsed argument/content documents match
                            the reference's EXACTLY?

...plus a KLD/top-1/top-8 next-token-distribution comparison against the
reference, reusing scripts/kld-compare-raw.py's raw-completion protocol
(sound across quants of the same model; no chat template involved).

Constrained decoding may hold syntax at 100% while argument-content fidelity
decays with bits, or it may not — this harness is built to show either
result, not to presuppose one (see docs/quant-fidelity.md).

Self-validation (zero-point mode): before measuring any variant, the
REFERENCE model is run through the pipeline TWICE (two independent server
spawns) and the two runs are compared with the exact same machinery used for
every other variant. Two greedy runs of the same weights must agree on every
tool call and score 0 KLD; if they do not, the harness refuses to measure
anything else — a broken comparator or nondeterministic decode would make
every downstream number meaningless.

Usage:
  # zero-point only: reference and sole variant are the same file
  quant-fidelity.py \\
      --reference q4_0=models/granite-4.1-8b-Q4_0.gguf \\
      --variant   q4_0=models/granite-4.1-8b-Q4_0.gguf \\
      --cases 16 --kld-positions 20 --out tests/quant-fidelity/out/smoke

  # a real ladder: one reference (usually the highest-precision quant),
  # several variants (repeat --variant; the reference itself does not need
  # to be repeated as a variant, but including it is how zero-point folds
  # into the same report)
  quant-fidelity.py \\
      --reference q8_0=models/m-Q8_0.gguf \\
      --variant q8_0=models/m-Q8_0.gguf \\
      --variant q4_k_m=models/m-Q4_K_M.gguf \\
      --variant q2_k=models/m-Q2_K.gguf \\
      --out tests/quant-fidelity/out/granite-ladder

Shared-box etiquette: before every server spawn this script waits (15s
poll, printed) until `pgrep -f "runner -m"` shows no OTHER runner process,
so it never contends with a concurrent load on a memory-constrained host.
Disable with --skip-idle-wait.

Output: tests/quant-fidelity/out/<name>/report.json
(schema_version "xyntetik.quant-fidelity.v1") and report.md.

Exit codes: 0 report complete (zero-point passed, every variant measured
with nothing not_executed); 1 zero-point passed but the report is
incomplete (a variant failed to spawn, or has not_executed cases/positions
— the report.json is still written and is honest about what happened);
2 zero-point FAILED, no variant was measured; 3 the reference model itself
never came up, so not even zero-point could run.
"""

import argparse
import importlib.util
import json
import subprocess
import urllib.request
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))

from harness import (Client, ConformanceError, ProtocolError,  # noqa: E402
                     RunnerServer, TransportError, categorize, find_runner,
                     validate_against_schema)

SCHEMA_VERSION = "xyntetik.quant-fidelity.v1"
DEFAULT_CORPUS = ROOT / "tests" / "fixtures" / "mixed-corpus.txt"


def _load_sibling(module_name, filename):
    """Load another hyphenated script in scripts/ as an importable module,
    the same importlib idiom scripts/stress-models.py and every scripts
    test file already use for this (hyphens make `import` unusable)."""
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AGENT_TORTURE = _load_sibling("quant_fidelity_agent_torture", "agent-torture.py")
KLD_RAW = _load_sibling("quant_fidelity_kld_raw", "kld-compare-raw.py")


# --------------------------------------------------------------- categories
# Which fidelity axes apply to each agent-torture case category. "tool" and
# "tool_stream" categories carry a forced tool call; "final" is a
# response_format-constrained document; "prose" (stream_normalization) is
# plain text with no schema/tool contract at all, and is only run to keep
# the transport path exercised — it never enters the schema/tool/argument
# denominators.
CATEGORY_KIND = {
    "nested_arguments": "tool",
    "tool_selection": "tool",
    "forced_truncation": "tool",
    "large_enum_selection": "tool",
    "reasoning_then_tool": "tool",
    "structured_final": "final",
    "tool_stream_normalization": "tool_stream",
    "stream_normalization": "prose",
}


def build_cases(count):
    """The tool-scenario matrix, reused wholesale from agent-torture.py."""
    return AGENT_TORTURE.build_cases(count)


# ------------------------------------------------------------------ judging
def _schema_for_tool(request, tool_name):
    for tool in request.get("tools") or []:
        if tool["function"]["name"] == tool_name:
            return tool["function"]["parameters"]
    return None


def _judged(**fields):
    base = {"executed": True, "protocol_ok": False, "schema_valid": False,
           "tool_name": None, "arguments": None, "content": None,
           "error": None}
    base.update(fields)
    return base


def _fail(result, category, message):
    result["error"] = {"category": category, "message": message}
    return result


def _validate_tool_call(result, request, name, arguments):
    """Shared tail of judge_tool_response/judge_tool_stream: resolve the
    schema by the tool the model actually NAMED (never by what the request
    forced) — a hallucinated call to a tool that was never offered must
    score schema_valid=False, not silently validate against some other
    tool's schema. tool_name/arguments are already on ``result`` by the
    time this runs and are never rolled back: schema validity and tool
    SELECTION are different axes, and a schema violation must not erase
    which tool was picked (that would corrupt the tool_selection metric,
    which has to stay readable independently of schema conformance)."""
    schema = _schema_for_tool(request, name)
    if schema is None:
        offered = [t["function"]["name"] for t in request.get("tools") or []]
        return _fail(result, "protocol",
                    f"model called a tool not offered in this request: "
                    f"{name!r} (offered: {offered})")
    try:
        validate_against_schema(arguments, schema)
    except ConformanceError as exc:
        return _fail(result, categorize(exc), str(exc))
    result["schema_valid"] = True
    return result


def judge_tool_response(request, response):
    """Score a buffered tool-call response. Never raises: every failure
    mode (bad status, wrong call count, invalid JSON, unoffered tool,
    schema violation) lands in ``error`` with a taxonomy category from the
    conformance harness, and the case still counts as *executed* — it got
    a real answer, just a bad one. tool_name/arguments are preserved as
    soon as they are known, even if a later step fails, so tool-selection
    and argument-agreement scoring survive an otherwise-invalid call."""
    result = _judged()
    try:
        response.expect_status(200)
        message = response.choice.get("message") or {}
    except ConformanceError as exc:
        return _fail(result, categorize(exc), str(exc))
    # Recorded before the call-count check, because a turn with NO tool call is
    # a legitimate answer that compare_case scores on this prose alone: leaving
    # it None makes "both declined and said the same thing" compare None with
    # None and agree unconditionally.
    result["content"] = message.get("content")
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        return _fail(result, "protocol",
                    f"expected exactly one tool call, got {calls!r}")
    function = calls[0].get("function") or {}
    name = function.get("name")
    result["tool_name"] = name
    arguments_raw = function.get("arguments")
    if not isinstance(arguments_raw, str):
        return _fail(result, "protocol", "tool arguments are not a JSON string")
    try:
        arguments = json.loads(arguments_raw)
    except ValueError as exc:
        return _fail(result, "schema", f"tool arguments are invalid JSON: {exc}")
    result["arguments"] = arguments
    result["protocol_ok"] = True
    return _validate_tool_call(result, request, name, arguments)


def judge_tool_stream(request, stream):
    """Same contract as judge_tool_response for a streamed tool call,
    normalized through agent-torture's transport-invariant SSE parser."""
    result = _judged()
    try:
        stream.expect_sse()
        normalized = AGENT_TORTURE.normalize_sse(stream.raw, [stream.raw])
    except ConformanceError as exc:
        return _fail(result, categorize(exc), str(exc))
    result["content"] = normalized["text"]   # see judge_tool_response
    if not normalized["saw_done"]:
        return _fail(result, "protocol", "stream omitted [DONE]")
    calls = normalized["tool_calls"]
    if len(calls) != 1:
        return _fail(result, "protocol",
                    f"expected exactly one streamed tool call, got {calls!r}")
    call = calls[0]
    name = call["name"]
    result["tool_name"] = name
    try:
        arguments = json.loads(call["arguments"])
    except ValueError as exc:
        return _fail(result, "schema",
                    f"streamed tool arguments are invalid JSON: {exc}")
    result["arguments"] = arguments
    result["protocol_ok"] = True
    return _validate_tool_call(result, request, name, arguments)


def judge_final(request, response):
    """Score a response_format-constrained final answer. Like the tool
    judges above, ``content`` is preserved across a schema violation."""
    result = _judged()
    try:
        response.expect_status(200)
        message = response.choice.get("message") or {}
    except ConformanceError as exc:
        return _fail(result, categorize(exc), str(exc))
    if message.get("tool_calls"):
        return _fail(result, "protocol",
                    "a structured final answer emitted a tool call")
    content_raw = message.get("content")
    if not isinstance(content_raw, str) or not content_raw.strip():
        return _fail(result, "protocol", "structured final produced no content")
    try:
        document = json.loads(content_raw)
    except ValueError as exc:
        return _fail(result, "schema", f"structured final is not JSON: {exc}")
    result["content"] = document
    result["protocol_ok"] = True
    schema = request["response_format"]["json_schema"]["schema"]
    try:
        validate_against_schema(document, schema)
    except ConformanceError as exc:
        return _fail(result, categorize(exc), str(exc))
    result["schema_valid"] = True
    return result


def judge_prose(request, stream):
    """Plain streamed prose: no schema, no tool — only the transport path
    is exercised. schema_valid stays None (not applicable, not failed)."""
    try:
        stream.expect_sse()
        normalized = AGENT_TORTURE.normalize_sse(stream.raw, [stream.raw])
        if not normalized["saw_done"]:
            raise ProtocolError("stream omitted [DONE]")
        result = _judged(content=normalized["text"], protocol_ok=True,
                         schema_valid=None)
        return result
    except ConformanceError as exc:
        result = _judged(schema_valid=None)
        result["error"] = {"category": categorize(exc), "message": str(exc)}
        return result


JUDGE_BY_KIND = {
    "tool": judge_tool_response,
    "tool_stream": judge_tool_stream,
    "final": judge_final,
    "prose": judge_prose,
}


# ------------------------------------------------------------- case runner
def run_case(client, case):
    """Issue one case's request against ``client`` and judge the result.
    Never raises: a transport-level failure (spawn died, socket refused,
    read timeout) is caught here and the case comes back not executed,
    with the reason preserved — see RNR-007 in scripts/compat_matrix.py."""
    category = case["category"]
    kind = CATEGORY_KIND[category]
    try:
        if kind in ("prose", "tool_stream"):
            response = client.chat_stream(case["request"], name=case["id"])
        else:
            response = client.chat(case["request"], name=case["id"])
    except Exception as exc:  # noqa: BLE001 - genuinely any transport failure
        # ConformanceError.__str__ already embeds "[category] "; a bare
        # exception (socket/OS error) gets the tag added here so every
        # not_executed_reason is uniformly taggable.
        reason = str(exc) if isinstance(exc, ConformanceError) else f"[transport] {exc}"
        return {"id": case["id"], "category": category, "executed": False,
               "not_executed_reason": reason}
    judged = JUDGE_BY_KIND[kind](case["request"], response)
    judged["id"] = case["id"]
    judged["category"] = category
    return judged


def run_matrix(client, cases):
    """Run every case; return {case_id: judged_dict}, order-preserving."""
    return {case["id"]: run_case(client, case) for case in cases}


# --------------------------------------------------------- reference compare
def compare_case(ref_judged, var_judged):
    """Score one variant case against the reference's judged case for the
    SAME case id. Returns a dict; never raises."""
    var_judged = var_judged or {}
    kind = CATEGORY_KIND.get(var_judged.get("category"))
    out = {"id": var_judged.get("id"), "category": var_judged.get("category"),
          "executed": bool(var_judged.get("executed")),
          "schema_valid": None, "tool_selection_correct": None,
          "argument_exact_match": None, "comparable": False,
          "issue_reason": None}
    if not out["executed"]:
        out["issue_reason"] = var_judged.get("not_executed_reason", "unknown failure")
        return out
    out["schema_valid"] = var_judged.get("schema_valid")
    ref_judged = ref_judged or {}
    if not ref_judged.get("executed"):
        out["issue_reason"] = "reference case did not execute; no basis for comparison"
        return out
    out["comparable"] = True
    if kind in ("tool", "tool_stream"):
        ref_name = ref_judged.get("tool_name")
        var_name = var_judged.get("tool_name")
        if ref_name is None:
            # The REFERENCE answered without calling a tool. There is no
            # selection to be correct about, so scoring one penalises the
            # variant for the reference's behaviour. Agreement here means the
            # variant also declined and said the same thing.
            #
            # Getting this wrong broke a whole model: Qwen3-8B's zero point
            # failed twice with "10 case(s) disagreed between two greedy runs
            # of the same reference weights" while the two runs were in fact
            # byte-identical, because both declined to call a tool in five
            # categories and "both declined" was recorded as disagreement. A
            # tool-tuned model calls a tool everywhere and never trips it; a
            # thinking model that sometimes answers directly trips it in every
            # such case, and the harness then refuses to measure it at all.
            out["tool_selection_correct"] = None
            out["argument_exact_match"] = (
                var_name is None and
                var_judged.get("content") == ref_judged.get("content"))
        else:
            correct = var_name == ref_name
            out["tool_selection_correct"] = correct
            out["argument_exact_match"] = (
                correct and var_judged.get("arguments") == ref_judged.get("arguments"))
    elif kind == "final":
        out["argument_exact_match"] = (
            var_judged.get("content") == ref_judged.get("content"))
    return out


def compare_matrix(ref_by_id, var_by_id, case_ids):
    return [compare_case(ref_by_id.get(cid), var_by_id.get(cid))
           for cid in case_ids]


def _rate(items, key):
    scored = [c for c in items if c.get(key) is not None]
    hit = sum(1 for c in scored if c[key])
    return {"count": hit, "scored": len(scored),
           "rate": round(hit / len(scored), 6) if scored else None}


def aggregate_tool_fidelity(comparisons):
    return {
        "requests": len(comparisons),
        "schema_conformance": _rate(comparisons, "schema_valid"),
        "tool_selection": _rate(comparisons, "tool_selection_correct"),
        "argument_agreement": _rate(comparisons, "argument_exact_match"),
    }


def not_executed_entries(comparisons):
    """Every scenario that could not run OR could not be scored, honestly
    recorded — never silently dropped (RNR-007)."""
    return [{"id": c["id"], "category": c["category"], "reason": c["issue_reason"]}
           for c in comparisons if c["issue_reason"] is not None]


# --------------------------------------------------------------------- KLD
def collect_distributions(endpoint, model_name, words, max_positions, stride,
                          top_n=20):
    """Next-token top-N logprob distributions at a stride of prefix
    positions over ``words``, via kld-compare-raw.py's raw-completion
    query. Returns (positions: [{"index", "dist"}], failures: [{"index",
    "reason"}]) — failures are preserved, not just counted, so a KLD lane
    honors the same not-silently-dropped rule as the tool matrix."""
    positions, failures = [], []
    prefix = ""
    scored = 0
    for i, word in enumerate(words):
        prefix = (prefix + " " + word).strip() if prefix else word
        if i % stride != 0:
            continue
        if scored >= max_positions:
            break
        if len(failures) >= max_positions:
            # a dead lane fails FAST and bounded: without this, a wrong
            # model name or downed server walked the entire corpus (5052
            # attempts for a --kld-positions 8 run) before reporting
            break
        try:
            dist = KLD_RAW.query(endpoint, model_name, prefix, top_n=top_n)
        except Exception as exc:  # noqa: BLE001
            failures.append({"index": i, "reason": str(exc)})
            continue
        positions.append({"index": i, "dist": dist})
        scored += 1
    return positions, failures


def compare_distributions(ref_positions, var_positions):
    """Pairs reference/variant distributions by list order (both were
    walked over the identical corpus+stride, so position i in each list is
    the same prefix) and aggregates exactly like kld-compare-raw.py's own
    summary — same fields, so results are directly comparable."""
    n = min(len(ref_positions), len(var_positions))
    klds, top1s, overlaps = [], [], []
    for i in range(n):
        # (variant, reference), the order kld-compare-raw.score_pair documents
        # and every published number was measured in. KLD is asymmetric and the
        # missing-token floor is applied to whichever side comes first, so the
        # reversed call answers a different question at a materially smaller
        # value -- and cannot be checked against the adopted bound.
        kld, agree, overlap8 = KLD_RAW.kld_and_agreement(
            var_positions[i]["dist"], ref_positions[i]["dist"])
        klds.append(kld)
        top1s.append(agree)
        overlaps.append(overlap8)
    return {
        "positions_scored": n,
        "mean_kld": sum(klds) / len(klds) if klds else None,
        "top1_agreement_pct": 100.0 * sum(top1s) / len(top1s) if top1s else None,
        "mean_top8_overlap": sum(overlaps) / len(overlaps) if overlaps else None,
    }


# --------------------------------------------------------------- zero-point
def zero_point_verdict(gold_comparisons, gold_kld_failures, var_kld_failures,
                       kld_summary, kld_epsilon=1e-6):
    """Whether the reference-vs-itself pass is EXACT. gold_comparisons is
    compare_matrix(gold, check, ids) — the second real run scored against
    the first with the identical machinery used for every other variant.
    Refuses on anything less than perfect agreement and ~0 KLD."""
    mismatches = [c for c in gold_comparisons
                 if c["executed"] and c["comparable"] and
                 (c["tool_selection_correct"] is False or
                  c["argument_exact_match"] is False)]
    not_executed = [c for c in gold_comparisons if not c["executed"] or not c["comparable"]]
    reasons = []
    if mismatches:
        reasons.append(f"{len(mismatches)} case(s) disagreed between two greedy "
                       "runs of the same reference weights")
    if not_executed:
        why = "; ".join(sorted({str(c.get("issue_reason")) for c in not_executed}))
        reasons.append(f"{len(not_executed)} case(s) did not execute or could "
                       f"not be compared on one of the two runs ({why})")
    if gold_kld_failures or var_kld_failures:
        first = (gold_kld_failures + var_kld_failures)[0]["reason"]
        reasons.append(f"{len(gold_kld_failures) + len(var_kld_failures)} KLD "
                       f"position(s) failed on one of the two runs; "
                       f"first: {first}")
    mean_kld = kld_summary.get("mean_kld")
    if mean_kld is None or mean_kld > kld_epsilon:
        reasons.append(f"mean_kld={mean_kld!r} exceeds epsilon={kld_epsilon}")
    top1 = kld_summary.get("top1_agreement_pct")
    if top1 is None or top1 != 100.0:
        reasons.append(f"top1_agreement_pct={top1!r} != 100.0")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "mismatches": [{"id": c["id"], "category": c["category"]}
                       for c in mismatches],
        "kld_summary": kld_summary,
    }


# -------------------------------------------------------------- idle wait
def other_runner_pids(pgrep_output, exclude_pid=None):
    """Pure: parse `pgrep -f "runner -m"` stdout into a pid list, excluding
    our own spawn (if any). Factored out so the wait loop is testable
    without shelling out."""
    pids = [int(p) for p in pgrep_output.split() if p.strip()]
    if exclude_pid is not None:
        pids = [p for p in pids if p != exclude_pid]
    return pids


def wait_for_idle_runner(poll_seconds=15, out=sys.stderr, sleep=time.sleep,
                         run=subprocess.run):
    """Block until no OTHER `runner -m` process is alive on this shared
    box, printing status each poll. On platforms without pgrep (e.g.
    Windows) this is a no-op — there is nothing safe to check."""
    while True:
        try:
            proc = run(["pgrep", "-f", "runner -m"], capture_output=True,
                      text=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            return
        if proc.returncode not in (0, 1):
            print(f"quant-fidelity: pgrep failed ({proc.stderr.strip()}); "
                 "proceeding without the idle wait", file=out)
            return
        pids = other_runner_pids(proc.stdout)
        if not pids:
            return
        print(f"quant-fidelity: waiting for other runner process(es) to exit: "
             f"{pids}", file=out)
        sleep(poll_seconds)


# -------------------------------------------------------------- server glue
def served_model_name(base_url):
    """The name the server actually registered — /v1/models is ground
    truth. Passing anything else (a ladder label, say) 404s every request
    while the transport still looks healthy: the exact trap the 2026-08-11
    kld-compare-raw note documented, rebuilt here on day one by passing the
    variant LABEL as the model name. Never guess the name; ask the server."""
    req = urllib.request.Request(f"{base_url}/v1/models")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["data"][0]["id"]


def spawn(exe, model, ctx, out_dir, tag):
    return RunnerServer(str(exe), str(model), ctx=ctx, parallel=1,
                        extra_args=["--gpu", "off"],
                        log_path=str(out_dir / f"runner.{tag}.log"))


def measure_pass(exe, model_path, model_label, cases, ctx, corpus_words,
                 kld_positions, kld_stride, out_dir, tag, idle_wait=True,
                 request_timeout=120.0):
    """One full pass over ``model_path``: spawn, run the tool matrix, run
    the KLD corpus, stop. Returns (judged_by_id, kld_positions, kld_failures,
    elapsed_ms) or raises TransportError if the server never came up (the
    caller decides whether that is fatal)."""
    if idle_wait:
        wait_for_idle_runner()
    started = time.monotonic()
    server = spawn(exe, model_path, ctx, out_dir, tag)
    with server as srv:
        client = Client(srv, _NullReport(), timeout=request_timeout)
        judged_by_id = run_matrix(client, cases)
        positions, failures = collect_distributions(
            srv.base_url, served_model_name(srv.base_url), corpus_words,
            kld_positions, kld_stride)
    elapsed_ms = round((time.monotonic() - started) * 1000, 2)
    return judged_by_id, positions, failures, elapsed_ms


class _NullReport:
    def record_request(self, record):
        pass


# ----------------------------------------------------------------- report
def build_report(reference, zero_point, variant_rows, configuration):
    complete = zero_point["passed"] and all(
        row["status"] == "measured" and not row["tool_fidelity_not_executed"]
        and not row["kld_not_executed"]
        for row in variant_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reference": reference,
        "configuration": configuration,
        "zero_point": zero_point,
        "variants": variant_rows,
        "complete": complete,
    }


def render_markdown(report):
    lines = [f"# Quant fidelity — {report['reference']['label']} reference\n"]
    zp = report["zero_point"]
    lines.append(f"Zero-point (reference vs itself): "
                f"**{'PASSED' if zp['passed'] else 'FAILED'}**"
                + ("" if zp["passed"] else f" — {'; '.join(zp['reasons'])}"))
    lines.append("")
    header = ["variant", "reference?", "schema conformance", "tool selection",
              "argument agreement", "mean KLD", "top-1 agreement",
              "not executed"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    def pct(rate_obj):
        if not rate_obj or rate_obj["rate"] is None:
            return "n/a"
        return f"{rate_obj['rate'] * 100:.1f}% ({rate_obj['count']}/{rate_obj['scored']})"

    for row in report["variants"]:
        if row["status"] != "measured":
            lines.append("| " + " | ".join([
                row["label"], "yes" if row["is_reference"] else "",
                "not_executed", "not_executed", "not_executed", "n/a", "n/a",
                row.get("status_reason", "")]) + " |")
            continue
        tf = row["tool_fidelity"]
        kld = row["kld_summary"]
        mean_kld = f"{kld['mean_kld']:.4g}" if kld.get("mean_kld") is not None else "n/a"
        top1 = f"{kld['top1_agreement_pct']:.1f}%" if kld.get("top1_agreement_pct") is not None else "n/a"
        not_exec = len(row["tool_fidelity_not_executed"]) + len(row["kld_not_executed"])
        lines.append("| " + " | ".join([
            row["label"], "yes" if row["is_reference"] else "",
            pct(tf["schema_conformance"]), pct(tf["tool_selection"]),
            pct(tf["argument_agreement"]), mean_kld, top1, str(not_exec)]) + " |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------- CLI
def _parse_labeled_path(value):
    label, sep, path = value.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(
            f"expected LABEL=PATH, got {value!r}")
    return {"label": label, "path": Path(path)}


def _version(exe):
    proc = subprocess.run([str(exe), "--version"], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          check=False)
    return proc.stdout.strip() or f"exit {proc.returncode}"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--reference", type=_parse_labeled_path, required=True,
                        help="LABEL=PATH of the highest-precision pin")
    parser.add_argument("--variant", type=_parse_labeled_path, action="append",
                        default=[], required=True,
                        help="LABEL=PATH; repeatable. Include the reference "
                             "itself to fold zero-point into the same report "
                             "at no extra spawn cost.")
    parser.add_argument("--cases", type=int, default=16,
                        help="tool-scenario request budget per pass (default: 16)")
    parser.add_argument("--kld-corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--kld-positions", type=int, default=20)
    parser.add_argument("--kld-stride", type=int, default=1)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--request-timeout", type=float, default=120.0,
                        help="per-request client timeout in seconds. A 5 GB "
                             "model generating tool responses on a loaded "
                             "8 GB CPU box needs minutes, not the default; "
                             "a timeout surfaces as a transport failure in "
                             "the report, never a hang")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing --out report (default: refuse)")
    parser.add_argument("--skip-idle-wait", action="store_true",
                        help="do not wait for other runner processes before "
                             "spawning (default: wait, 15s poll)")
    parser.add_argument("--zero-point-kld-epsilon", type=float, default=1e-6)
    args = parser.parse_args(argv)

    report_path = args.out / "report.json"
    if report_path.exists() and not args.force:
        parser.error(f"refusing to overwrite existing evidence {report_path} "
                    "(pass --force)")
    if not args.reference["path"].is_file():
        parser.error(f"reference model not found: {args.reference['path']}")
    if not args.kld_corpus.is_file():
        parser.error(f"KLD corpus not found: {args.kld_corpus}")

    exe = args.runner or Path(find_runner(str(ROOT)))
    args.out.mkdir(parents=True, exist_ok=True)
    idle_wait = not args.skip_idle_wait
    cases = build_cases(args.cases)
    case_ids = [c["id"] for c in cases]
    corpus_words = args.kld_corpus.read_text(encoding="utf-8").split()

    try:
        print(f"quant-fidelity: zero-point pass 1/2 (reference={args.reference['label']})")
        gold_by_id, gold_kld, gold_kld_fail, gold_ms = measure_pass(exe, args.reference["path"], args.reference["label"], cases, args.ctx,
            corpus_words, args.kld_positions, args.kld_stride, args.out, "gold",
            idle_wait=idle_wait, request_timeout=args.request_timeout)
        print(f"quant-fidelity: zero-point pass 2/2 (reference={args.reference['label']})")
        check_by_id, check_kld, check_kld_fail, check_ms = measure_pass(exe, args.reference["path"], args.reference["label"], cases, args.ctx,
            corpus_words, args.kld_positions, args.kld_stride, args.out, "check",
            idle_wait=idle_wait, request_timeout=args.request_timeout)
    except TransportError as exc:
        # the reference itself never came up: there is no basis for a
        # zero-point pass at all, let alone a ladder. Fail loudly and
        # distinctly from "zero-point ran but disagreed" (exit 2).
        print(f"quant-fidelity: reference model failed to spawn, cannot "
             f"establish zero-point: {exc}", file=sys.stderr)
        return 3

    zp_comparisons = compare_matrix(gold_by_id, check_by_id, case_ids)
    zp_kld_summary = compare_distributions(gold_kld, check_kld)
    zero_point = zero_point_verdict(zp_comparisons, gold_kld_fail, check_kld_fail,
                                    zp_kld_summary, args.zero_point_kld_epsilon)

    configuration = {
        "runner": str(exe), "runner_version": _version(exe),
        "cases": args.cases, "kld_corpus": str(args.kld_corpus),
        "kld_positions": args.kld_positions, "kld_stride": args.kld_stride,
        "ctx": args.ctx,
    }
    reference_meta = {"label": args.reference["label"],
                      "path": str(args.reference["path"])}

    if not zero_point["passed"]:
        report = build_report(reference_meta, zero_point, [], configuration)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (args.out / "report.md").write_text(render_markdown(report))
        print("quant-fidelity: ZERO-POINT FAILED, refusing to measure any "
             f"variant: {'; '.join(zero_point['reasons'])}", file=sys.stderr)
        print(f"report: {report_path}")
        return 2
    print("quant-fidelity: zero-point PASSED "
         f"(mean_kld={zp_kld_summary['mean_kld']}, "
         f"top1={zp_kld_summary['top1_agreement_pct']}%)")

    reference_resolved = args.reference["path"].resolve()
    variant_rows = []
    for variant in args.variant:
        label, path = variant["label"], variant["path"]
        if not path.is_file():
            variant_rows.append({
                "label": label, "path": str(path),
                "is_reference": path.resolve() == reference_resolved,
                "status": "not_executed",
                "status_reason": f"model file not found: {path}",
                "tool_fidelity": None, "tool_fidelity_not_executed": [],
                "kld_summary": None, "kld_not_executed": [],
                "reused_zero_point_run": False, "elapsed_ms": None,
            })
            continue
        is_reference = path.resolve() == reference_resolved
        if is_reference:
            # the reference model was just measured twice for zero-point;
            # reuse the second pass rather than loading an 8B+ model a
            # third time on a memory-constrained box
            var_by_id, var_kld, var_kld_fail = check_by_id, check_kld, check_kld_fail
            elapsed_ms = check_ms
            reused = True
        else:
            print(f"quant-fidelity: measuring variant {label!r}")
            try:
                var_by_id, var_kld, var_kld_fail, elapsed_ms = measure_pass(exe, path, label, cases, args.ctx, corpus_words,
                    args.kld_positions, args.kld_stride, args.out,
                    label.replace("/", "_"), idle_wait=idle_wait,
                    request_timeout=args.request_timeout)
            except TransportError as exc:
                variant_rows.append({
                    "label": label, "path": str(path), "is_reference": False,
                    "status": "not_executed", "status_reason": str(exc),
                    "tool_fidelity": None, "tool_fidelity_not_executed": [],
                    "kld_summary": None, "kld_not_executed": [],
                    "reused_zero_point_run": False, "elapsed_ms": None,
                })
                continue
            reused = False
        comparisons = compare_matrix(gold_by_id, var_by_id, case_ids)
        variant_rows.append({
            "label": label, "path": str(path), "is_reference": is_reference,
            "status": "measured",
            "tool_fidelity": aggregate_tool_fidelity(comparisons),
            "tool_fidelity_not_executed": not_executed_entries(comparisons),
            "kld_summary": compare_distributions(gold_kld, var_kld),
            "kld_not_executed": [{"index": f["index"], "reason": f["reason"]}
                                 for f in var_kld_fail],
            "reused_zero_point_run": reused, "elapsed_ms": elapsed_ms,
        })

    report = build_report(reference_meta, zero_point, variant_rows, configuration)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md = render_markdown(report)
    (args.out / "report.md").write_text(md)
    print(md)
    print(f"report: {report_path}")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
