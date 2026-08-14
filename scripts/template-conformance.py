#!/usr/bin/env python3
"""Byte-exact conformance gate: runner's native C chat renderer vs each
family's UPSTREAM jinja chat template.

Runner renders chat templates natively in C instead of executing the model's
jinja template. That is deliberate (no jinja interpreter in the inference
path, and constrained decoding needs the framing as structure rather than as
text), but it means every family can silently DRIFT from its reference and
nothing notices -- goldens written from runner's own output only prove
self-consistency. tests/test_template.c is exactly that kind of golden. This
script supplies the missing oracle.

For each family it:
  1. loads the upstream `chat_template` (HuggingFace tokenizer_config.json /
     chat_template.jinja, or tokenizer.chat_template out of a local GGUF),
  2. renders a fixed conversation matrix through it with transformers-
     compatible jinja semantics (trim_blocks + lstrip_blocks, raise_exception,
     strftime_now, bos_token/eos_token, add_generation_prompt),
  3. renders the SAME matrix through runner's C renderer
     (render_messages_with_tools, via scripts/template-conformance-render.c),
  4. diffs byte-for-byte, and subtracts the deviations runner takes ON PURPOSE
     -- scripts/template-conformance-allowlist.json, every entry carrying a
     written reason and a source citation the harness re-verifies,
  5. subtracts the differences already KNOWN to be bugs --
     scripts/template-conformance-baseline.json, a backlog with per-entry
     status, not permission.

    make template-conformance            run the gate
    make template-conformance-refresh    re-fetch the upstream oracles first
    make template-conformance-baseline   re-record the known-difference backlog

Exit status:
    0   every selected family checked; no new drift; allowlist healthy
    1   NEW drift, drift that CHANGED SHAPE, a fixed-but-still-listed backlog
        entry, an upstream oracle that moved, or a rotted allowlist entry
    2   NOT CHECKED -- jinja2 missing, no compiler, or an oracle that could
        not be obtained. This is never reported as conformance. A gate that
        quietly checks nothing is worse than one that fails.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
RENDER_C = os.path.join(SCRIPTS, "template-conformance-render.c")
ALLOWLIST = os.path.join(SCRIPTS, "template-conformance-allowlist.json")
BASELINE = os.path.join(SCRIPTS, "template-conformance-baseline.json")
RUNNER_SRC = ["src/template.c", "src/json.c", "src/tokenizer.c",
              "src/gguf.c", "src/compat.c", "src/quants.c"]

EXIT_OK, EXIT_DRIFT, EXIT_NOT_CHECKED = 0, 1, 2


# --------------------------------------------------------------- families
#
# Where each family's reference template comes from. `hf` reads the model
# repo's tokenizer_config.json (falling back to chat_template.jinja); `gguf`
# reads tokenizer.chat_template straight out of a local GGUF, which is what
# src/template.c cites for the families whose HF repo is gated.

class Family:
    def __init__(self, runner, source, note="", tool_family=False,
                 thinking_var=None, skip=None):
        self.runner = runner            # --chat-template name
        self.source = source            # ("hf", repo) | ("gguf", path)
        self.note = note
        self.tool_family = tool_family  # matrix includes the tool rows
        self.thinking_var = thinking_var
        self.skip = skip                # reason this family has no oracle


FAMILIES = {
    "chatml": Family(
        "chatml", ("hf", "Qwen/Qwen2.5-7B-Instruct"),
        note="generic ChatML; Qwen2.5 is the concrete checkpoint the repo "
             "tests against (models/Qwen2.5-7B-Instruct-Q4_K_M.gguf)",
        tool_family=True),
    "chatml-think": Family(
        "chatml-think", ("hf", "Qwen/Qwen3-4B"),
        note="src/template.h cites 'Qwen/Qwen3-* tokenizer_config.json'",
        tool_family=True, thinking_var="enable_thinking"),
    "llama2": Family(
        "llama2", ("hf", "unsloth/llama-2-7b-chat"),
        note="meta-llama/Llama-2-7b-chat-hf is gated (HTTP 401); this mirror "
             "carries the identical Llama-2 template"),
    "llama3": Family(
        "llama3", ("hf", "unsloth/Llama-3.2-3B-Instruct"),
        note="meta-llama/Llama-3.2-3B-Instruct is gated (HTTP 401); this "
             "mirror carries the identical Llama-3.2 template"),
    "zephyr": Family("zephyr", ("hf", "HuggingFaceH4/zephyr-7b-beta")),
    "gemma": Family(
        "gemma", ("gguf", "models/google_gemma-3-4b-it-Q4_K_M.gguf"),
        note="google/gemma-3-4b-it is gated on HF (401); the GGUF the repo "
             "already tests against carries the same tokenizer.chat_template"),
    "gemma4": Family(
        "gemma4", ("gguf", "models/e2b-q40.gguf"),
        note="src/template.h cites 'gemma-4 tokenizer.chat_template (read "
             "from the GGUF)'; byte-identical to google/gemma-4-E2B-it's "
             "chat_template.jinja on HF",
        thinking_var="enable_thinking"),
    "mistral": Family("mistral", ("hf", "mistralai/Mistral-7B-Instruct-v0.3")),
    "phi3": Family("phi3", ("hf", "microsoft/Phi-3.5-mini-instruct")),
    "apertus": Family(
        "apertus", ("hf", "swiss-ai/Apertus-8B-Instruct-2509"),
        note="tests/test_template.c cites this repo's chat_template.jinja"),
    "ornith": Family(
        "ornith", ("hf", "ornith-ai/Ornith-1.0-9B"),
        note="docs/ornith-reference.md names the model; the -GGUF repo is the "
             "same checkpoint", tool_family=True,
        thinking_var="enable_thinking"),
    "muse": Family(
        "muse", ("hf", "meta-models/Muse-Glimmer-30B"),
        note="src/template.c cites 'the model's OWN tokenizer.chat_template, "
             "read from the official meta-models GGUF'",
        tool_family=True),
    "granite": Family(
        "granite", ("gguf", "models/granite-4.1-8b-Q4_0.gguf"),
        note="src/template.c cites 'the model's OWN tokenizer.chat_template'; "
             "the ibm-granite HF repo is gated (401)"),
    "harmony": Family(
        "harmony", ("gguf", "models/gpt-oss-20b-MXFP4.gguf"),
        note="src/template.c cites 'the model's OWN tokenizer.chat_template "
             "in the official GGUF'",
        tool_family=True),
    "raw": Family("raw", None,
                  skip="TMPL_RAW is runner's own no-framing mode: it "
                       "concatenates message contents and has no upstream "
                       "template to conform to."),
}

# The GGUF-sourced families need a multi-GB local model to FETCH their oracle
# (after that the cached .jinja is enough). Anywhere without models/ -- CI,
# a fresh clone -- only this subset is checkable, and asking for the rest
# would make the gate red for a reason that has nothing to do with drift.
NETWORK_ONLY = tuple(n for n, f in FAMILIES.items()
                     if f.source and f.source[0] == "hf")


# ------------------------------------------------------------- the matrix

SYS = "You are a helpful assistant."
U1 = "What color is the sky?"
A1 = "Blue, because of Rayleigh scattering."
U2 = "And at sunset?"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string",
                                    "description": "City name."}},
            "required": ["city"],
        },
    },
}]
CALL = {"id": "call_1", "type": "function",
        "function": {"name": "get_weather",
                     "arguments": '{"city": "Oslo"}'}}


def matrix(family):
    """The same conversations for every family, so results are comparable."""
    cases = [
        ("user-only+gen", [{"role": "user", "content": U1}], True, "default"),
        ("user-only-nogen", [{"role": "user", "content": U1}], False,
         "default"),
        ("system+user+gen", [{"role": "system", "content": SYS},
                             {"role": "user", "content": U1}], True,
         "default"),
        ("system+multiturn+gen", [{"role": "system", "content": SYS},
                                  {"role": "user", "content": U1},
                                  {"role": "assistant", "content": A1},
                                  {"role": "user", "content": U2}], True,
         "default"),
        ("multiturn+gen", [{"role": "user", "content": U1},
                           {"role": "assistant", "content": A1},
                           {"role": "user", "content": U2}], True, "default"),
        ("multiturn-nogen", [{"role": "user", "content": U1},
                             {"role": "assistant", "content": A1},
                             {"role": "user", "content": U2}], False,
         "default"),
        ("trailing-assistant+gen", [{"role": "user", "content": U1},
                                    {"role": "assistant", "content": A1}],
         True, "default"),
        ("trailing-assistant-nogen", [{"role": "user", "content": U1},
                                      {"role": "assistant", "content": A1}],
         False, "default"),
    ]
    if family.thinking_var:
        for mode in ("on", "off"):
            cases.append((f"think-{mode}",
                          [{"role": "system", "content": SYS},
                           {"role": "user", "content": U1}], True, mode))
    if family.tool_family:
        cases.append(("tool-call+result", [
            {"role": "system", "content": SYS},
            {"role": "user", "content": "Weather in Oslo?"},
            {"role": "assistant", "content": "", "tool_calls": [CALL]},
            {"role": "tool", "name": "get_weather", "tool_call_id": "call_1",
             "content": '{"temp_c": -3}'},
        ], True, "default"))
    return cases


# ------------------------------------------------- runner-side flattening
#
# render_messages_with_tools takes flat {role, content, name, channel} turns,
# not OpenAI messages. src/server.c:message_text() does that flattening
# before rendering, so the harness has to do the same or it would compare
# runner against an input the server never hands it. This mirrors
# src/server.c; it is the one part of the comparison that is a
# RE-IMPLEMENTATION rather than runner's own code, and it only matters for
# the tool row. If server.c's flattening changes, this goes stale silently --
# the known weak spot of this harness, called out here on purpose.

def flatten_for_runner(family, msgs):
    out = []
    for m in msgs:
        role, content = m["role"], m.get("content") or ""
        calls = m.get("tool_calls") or []
        if family.runner == "ornith":
            if role == "assistant":
                content = ("<think>\n" + (m.get("reasoning_content") or "")
                           + "\n</think>\n\n" + content)
                for c in calls:  # tool_history_render_for, ornith branch
                    args = json.loads(c["function"]["arguments"])
                    content += "<tool_call>\n<function=%s>\n" % (
                        c["function"]["name"])
                    for k, v in args.items():
                        content += "<parameter=%s>\n%s\n</parameter>\n" % (
                            k, v if isinstance(v, str) else json.dumps(v))
                    content += "</function>\n</tool_call>"
            elif role == "tool":
                content = "<tool_response>\n" + content + "\n</tool_response>"
                role = "user"
            out.append({"role": role, "content": content})
            continue
        if family.runner == "harmony":
            if role == "assistant":
                if m.get("reasoning_content"):
                    out.append({"role": "assistant",
                                "content": m["reasoning_content"],
                                "channel": "analysis"})
                if content:
                    out.append({"role": "assistant", "content": content,
                                "channel": "commentary" if calls else None})
                for c in calls:
                    out.append({"role": "assistant",
                                "content": c["function"]["arguments"],
                                "name": c["function"]["name"]})
                continue
            out.append({"role": role, "content": content,
                        "name": m.get("name")})
            continue
        if family.runner == "muse":
            name = m.get("name")
            if role == "assistant" and calls:
                name = calls[0]["function"]["name"]
                body = ""
                for n, c in enumerate(calls):
                    args = json.loads(c["function"]["arguments"])
                    if n:
                        body += ("<|eom|><|start|>assistant to=%s<|message|>"
                                 % c["function"]["name"])
                    body += ('<atem:function_calls>\n<atem:invoke name="%s">\n'
                             % c["function"]["name"])
                    for k, v in args.items():
                        body += ('<atem:parameter name="%s">%s'
                                 "</atem:parameter>\n") % (
                            k, v if isinstance(v, str) else json.dumps(v))
                    body += "</atem:invoke>\n</atem:function_calls>"
                content += body
            out.append({"role": role, "content": content, "name": name})
            continue
        # generic families: runner's own tool-call marker, from
        # tool_history_render_for()'s default branch
        for c in calls:
            content += "<|tool_call>call:%s%s<tool_call|>" % (
                c["function"]["name"], c["function"]["arguments"])
        out.append({"role": role, "content": content})
    return out


# --------------------------------------------------------- oracle loading

def hf_raw(repo, path, tries=3):
    url = "https://huggingface.co/%s/raw/main/%s" % (repo, path)
    req = urllib.request.Request(url, headers={"User-Agent": "runner-conf"})
    last = None
    for _ in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8"), None
        except Exception as e:                                  # noqa: BLE001
            last = e
            # 401/404 are answers, not flakes: a gated or renamed repo will
            # not become reachable on a retry.
            if any(c in str(e) for c in ("401", "403", "404")):
                break
    return None, "%s: %s" % (url, last)


def gguf_kv(path):
    def u32(f):
        return struct.unpack("<I", f.read(4))[0]

    def u64(f):
        return struct.unpack("<Q", f.read(8))[0]

    def rstr(f):
        return f.read(u64(f)).decode("utf-8", "replace")

    def val(f, t):
        if t == 8:
            return rstr(f)
        if t == 9:
            at, n = u32(f), u64(f)
            return [val(f, at) for _ in range(n)]
        fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
               7: "<?", 10: "<Q", 11: "<q", 12: "<d"}[t]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

    with open(path, "rb") as f:
        magic, _ver, _nt, n_kv = struct.unpack("<IIQQ", f.read(24))
        if magic != 0x46554747:
            raise SystemExit("%s is not a GGUF file" % path)
        kv = {}
        for _ in range(n_kv):
            k = rstr(f)
            kv[k] = val(f, u32(f))
    return kv


def fetch_oracle(name, fam, oracle_dir):
    """Write <oracle_dir>/<name>.jinja and <name>.json. Returns (ok, err)."""
    kind, ref = fam.source
    tmpl, meta = None, {}
    if kind == "hf":
        raw, err = hf_raw(ref, "tokenizer_config.json")
        if raw:
            cfg = json.loads(raw)
            t = cfg.get("chat_template")
            if isinstance(t, list):
                t = ([x for x in t if x.get("name") == "default"]
                     or t)[0]["template"]
            tmpl = t
            for k in ("bos_token", "eos_token"):
                v = cfg.get(k)
                meta[k] = v.get("content") if isinstance(v, dict) else v
        if tmpl is None:
            tmpl, err2 = hf_raw(ref, "chat_template.jinja")
            err = err or err2
        if tmpl is None:
            return False, "cannot fetch %s (%s)" % (ref, err)
    else:
        # models/ is gitignored, so the GGUF may live elsewhere:
        # RUNNER_MODELS_DIR overrides, otherwise ROOT/models.
        base = os.environ.get("RUNNER_MODELS_DIR")
        path = (os.path.join(base, os.path.basename(ref)) if base
                else os.path.join(ROOT, ref))
        if not os.path.exists(path):
            return False, ("%s not present. Point RUNNER_MODELS_DIR at the "
                           "model shelf, or run --network-only to check just "
                           "the families whose oracle is fetched over HTTP."
                           % path)
        kv = gguf_kv(path)
        tmpl = kv.get("tokenizer.chat_template")
        toks = kv.get("tokenizer.ggml.tokens") or []
        for k, idk in (("bos_token", "tokenizer.ggml.bos_token_id"),
                       ("eos_token", "tokenizer.ggml.eos_token_id")):
            i = kv.get(idk)
            if isinstance(i, int) and 0 <= i < len(toks):
                meta[k] = toks[i]
    if not tmpl:
        return False, "no chat_template at %s:%s" % (kind, ref)
    os.makedirs(oracle_dir, exist_ok=True)
    with open(os.path.join(oracle_dir, name + ".jinja"), "w") as f:
        f.write(tmpl)
    meta["source"] = "%s:%s" % (kind, ref)
    meta["sha256"] = sha(tmpl)
    meta["fetched_at"] = datetime.datetime.now(
        datetime.timezone.utc).replace(microsecond=0).isoformat()
    with open(os.path.join(oracle_dir, name + ".json"), "w") as f:
        json.dump(meta, f, indent=1)
    return True, None


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------- jinja (reference)

def jinja_env():
    import jinja2.ext
    from jinja2.exceptions import TemplateError
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    # transformers' _compile_jinja_template, reproduced: trim_blocks and
    # lstrip_blocks are ON there, and getting them wrong manufactures fake
    # whitespace drift in every template that indents its {% %} tags.
    def raise_exception(message):
        raise TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None,
               sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    def strftime_now(fmt):
        return datetime.datetime.now().strftime(fmt)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    return env


def render_reference(env, tmpl_src, meta, msgs, add_gen, thinking,
                     thinking_var, tools):
    kwargs = dict(messages=msgs, add_generation_prompt=add_gen,
                  bos_token=meta.get("bos_token") or "",
                  eos_token=meta.get("eos_token") or "")
    if tools:
        kwargs["tools"] = tools
    if thinking_var and thinking != "default":
        kwargs[thinking_var] = (thinking == "on")
    return env.from_string(tmpl_src).render(**kwargs)


# ------------------------------------------------------------ runner side

def build_renderer():
    """Compile the runner-side driver. Returns (path, err)."""
    out = os.path.join(tempfile.mkdtemp(prefix="tmplconf"), "render")
    cc = os.environ.get("CC", "cc")
    cmd = [cc, "-O1", "-std=c11", "-I", os.path.join(ROOT, "src"), RENDER_C]
    cmd += [os.path.join(ROOT, s) for s in RUNNER_SRC]
    cmd += ["-o", out, "-lm"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        return None, "cannot build the runner-side renderer:\n%s" % p.stderr
    return out, None


def render_runner(binary, cases):
    p = subprocess.run([binary], input=json.dumps({"cases": cases}),
                       capture_output=True, text=True)
    if p.returncode:
        raise SystemExit("runner renderer failed: %s" % p.stderr)
    return {r["id"]: r["prompt"] for r in json.loads(p.stdout)["results"]}


# ----------------------------------------------------------- the allowlist
#
# Deviations runner takes ON PURPOSE. Each `patch` rewrites the ORACLE text
# into what runner would legitimately produce, so what survives into the diff
# is unexplained. Loaded from JSON so the policy is one reviewable file rather
# than lambdas buried in a script, and so a citation can be re-verified.

class Deviation:
    def __init__(self, raw):
        self.id = raw["id"]
        self.family = raw["family"]
        self.why = raw.get("why", "")
        self.cites = [c for c in (raw.get("cite"),
                                  raw.get("corroborating_cite")) if c]
        self.patch_spec = raw["patch"]
        self.used = 0

    def key(self):
        return self.id.split("/", 1)[-1]


def compile_patch(dev, bos):
    """Turn a declarative patch spec into oracle_text -> oracle_text."""
    spec = dev.patch_spec
    op = spec.get("op")
    if op == "strip-bos":
        b = bos or "\0"
        return lambda s: s[len(b):] if s.startswith(b) else s
    if op == "drop-first-match":
        rx = re.compile(spec["pattern"],
                        re.S if spec.get("dotall") else 0)
        return lambda s: rx.sub("", s, count=1)
    if op == "builtin":
        fn = BUILTIN_PATCHES.get(spec["name"])
        if not fn:
            raise SystemExit("allowlist %s: unknown builtin %r"
                             % (dev.id, spec["name"]))
        return fn
    raise SystemExit("allowlist %s: unknown patch op %r" % (dev.id, op))


# harmony's generation prompt is the one deviation that is conditional on the
# rendered text rather than a fixed rewrite: runner primes the analysis
# channel only when there are NO tools (with tools it leaves the header bare
# so the constrained recipient stays reachable). "namespace functions" is how
# the harmony reference spells a tool declaration block.
def _harmony_prime_analysis(s):
    if s.endswith("<|start|>assistant") and "namespace functions" not in s:
        return s + "<|channel|>analysis<|message|>"
    return s


BUILTIN_PATCHES = {"harmony-prime-analysis": _harmony_prime_analysis}


def load_allowlist(path):
    with open(path) as f:
        doc = json.load(f)
    devs = {}
    seen = set()
    for raw in doc["deviations"]:
        d = Deviation(raw)
        if d.id in seen:
            raise SystemExit("allowlist: duplicate id %s" % d.id)
        seen.add(d.id)
        devs.setdefault(d.family, []).append(d)
    return devs


def check_citations(devs, families):
    """Re-verify every citation. Returns a list of rot messages."""
    rot = []
    cache = {}
    for fam in families:
        for d in devs.get(fam, []):
            if not d.why.strip():
                rot.append("%s: no `why` -- an unexplained deviation is a bug,"
                           " not an allowlist entry" % d.id)
            if not d.cites:
                rot.append("%s: no `cite` -- an entry that cannot cite a "
                           "reason is an unfixed bug, move it to the baseline"
                           % d.id)
            for c in d.cites:
                rot += check_one_citation(d.id, c, cache)
    return rot


def check_one_citation(did, c, cache):
    path = os.path.join(ROOT, c["file"])
    if path not in cache:
        try:
            with open(path, encoding="utf-8") as f:
                cache[path] = f.read().splitlines()
        except OSError:
            cache[path] = None
    lines = cache[path]
    if lines is None:
        return ["%s: cited file %s does not exist" % (did, c["file"])]
    m = re.match(r"^(\d+)(?:-(\d+))?$", str(c.get("lines", "")))
    if not m:
        return ["%s: cite.lines %r is not N or N-M"
                % (did, c.get("lines"))]
    lo = int(m.group(1))
    hi = int(m.group(2) or lo)
    anchor = c.get("anchor")
    if not anchor:
        return ["%s: cite has no `anchor`; a bare line number cannot be "
                "re-verified and rots on the next edit" % did]
    if lo < 1 or hi > len(lines):
        return ["%s: %s:%s is past the end of the file (%d lines)"
                % (did, c["file"], c["lines"], len(lines))]
    if anchor in "\n".join(lines[lo - 1:hi]):
        return []
    # STALE vs DEAD: say which, and where it went.
    hits = [i + 1 for i, ln in enumerate(lines) if anchor in ln]
    if hits:
        return ["STALE CITATION %s: anchor %r is no longer in %s:%s -- it is "
                "now at line %s. Update cite.lines."
                % (did, anchor, c["file"], c["lines"],
                   ",".join(map(str, hits)))]
    return ["DEAD CITATION %s: anchor %r is gone from %s entirely. The reason "
            "this deviation was allowed no longer exists in the source; "
            "re-justify it or treat the difference as a bug."
            % (did, anchor, c["file"])]


# ------------------------------------------------------------- the baseline
#
# Differences that are KNOWN BUGS. Not permission -- a backlog. Each entry
# carries a status and an excerpt so the file reads as a bug list, and a
# digest so a difference that CHANGES SHAPE is not mistaken for the one that
# was signed off.

def diff_digest(want, mine):
    return sha(_denow(want) + "\x00" + _denow(mine))


def _denow(s):
    """Neutralise today's date so a backlog entry does not expire at midnight.

    Only TODAY's date in the formats the reference templates actually use is
    substituted -- a general \\d{4}-\\d{2}-\\d{2} scrub would also hide a
    hard-coded date, which is exactly the kind of drift this gate is for.
    """
    now = datetime.datetime.now()
    for fmt in ("%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %Y"):
        s = s.replace(now.strftime(fmt), "<TODAY>")
    return s


def first_diff_at(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def excerpt(s, at, width=90):
    lo = max(0, at - 20)
    return repr(s[lo:lo + width])


def load_baseline(path):
    if not os.path.exists(path):
        return {"version": 1, "oracles": {}, "known_differences": []}
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------ main

def run(args):
    names = args.family or list(FAMILIES)
    for n in names:
        if n not in FAMILIES:
            raise SystemExit("unknown family %r (have: %s)"
                             % (n, ", ".join(FAMILIES)))

    devs = load_allowlist(args.allowlist)
    base = load_baseline(args.baseline)
    known = {e["id"]: e for e in base.get("known_differences", [])}
    base_oracles = base.get("oracles", {})

    # ---- oracle side must actually be available, or nothing is "checked"
    not_checked = []
    try:
        env = jinja_env()
    except ImportError:
        print("NOT CHECKED: jinja2 is not installed, so no reference render "
              "could be produced.")
        print("             install it:  %s -m pip install jinja2"
              % os.path.basename(sys.executable))
        print("             (this is NOT a pass: nothing was compared)")
        return EXIT_NOT_CHECKED

    binary, err = build_renderer()
    if err:
        print("NOT CHECKED: %s" % err)
        return EXIT_NOT_CHECKED

    results = []        # (family, case, verdict, detail)
    oracle_meta = {}
    structural = []     # families with no upstream template BY DESIGN
    stale = []          # --refresh failed but a cached oracle was usable
    new_drift = changed = fixed = 0
    checked = 0

    for n in names:
        fam = FAMILIES[n]
        if fam.skip:
            # Not an availability problem: there is nothing upstream to
            # conform to. Reported, but it does not make the run incomplete.
            structural.append((n, fam.skip))
            continue
        jpath = os.path.join(args.oracle_dir, n + ".jinja")
        if args.refresh or not os.path.exists(jpath):
            ok, ferr = fetch_oracle(n, fam, args.oracle_dir)
            if not ok and not os.path.exists(jpath):
                not_checked.append((n, ferr))
                continue
            if not ok:
                # A refresh that failed with a usable cache still checks
                # something; say loudly that the oracle may be stale rather
                # than throwing away a comparison we can make.
                stale.append((n, ferr))
        tmpl_src = open(jpath).read()
        meta = json.load(open(os.path.join(args.oracle_dir, n + ".json")))
        oracle_meta[n] = {"source": meta.get("source", "?"),
                          "sha256": meta.get("sha256") or sha(tmpl_src),
                          "fetched_at": meta.get("fetched_at", "?")}

        fam_devs = devs.get(n, [])
        patches = [(d, compile_patch(d, meta.get("bos_token")))
                   for d in fam_devs]

        cases = matrix(fam)
        jobs = []
        for cid, msgs, add_gen, thinking in cases:
            tools = TOOLS if "tool" in cid else None
            jobs.append({
                "id": cid, "template": fam.runner,
                "add_generation_prompt": add_gen, "thinking": thinking,
                "messages": flatten_for_runner(fam, msgs),
                "tools": tools,
                # src/server.c builds a tools system turn for every family
                # EXCEPT muse and harmony, whose renderers take the
                # declarations directly. Mirroring that matters: adding the
                # turn for harmony would manufacture a diff runner never
                # produces.
                "server_tools_turn": bool(tools) and fam.runner not in
                                     ("muse", "harmony"),
            })
        got = render_runner(binary, jobs)

        for cid, msgs, add_gen, thinking in cases:
            cid_full = "%s/%s" % (n, cid)
            tools = TOOLS if "tool" in cid else None
            omsgs = msgs
            if tools:
                # Reference templates assume tool_call.arguments is a MAPPING:
                # muse raises outright on a string, harmony/qwen/granite run
                # it through |tojson and would double-encode one. OpenAI wire
                # format is a JSON string, which is what runner is handed.
                # Parsing it for the oracle side compares the two renderers on
                # the same call rather than on an input the template rejects.
                omsgs = json.loads(json.dumps(msgs))
                for m in omsgs:
                    for c in m.get("tool_calls") or []:
                        c["function"]["arguments"] = json.loads(
                            c["function"]["arguments"])
            try:
                want = render_reference(env, tmpl_src, meta, omsgs, add_gen,
                                        thinking, fam.thinking_var, tools)
            except Exception as e:                          # noqa: BLE001
                # The reference refusing to render is not a pass and not a
                # difference: it means this case was never compared.
                not_checked.append((cid_full,
                                    "reference template refused: %s" % e))
                results.append((n, cid, "NOT-CHECKED", "reference refused"))
                continue
            checked += 1

            applied = []
            if not args.raw:
                for d, patch in patches:
                    patched = patch(want)
                    if patched != want:
                        applied.append(d.key())
                        d.used += 1
                        want = patched
            mine = got[cid]
            entry = known.get(cid_full)

            if mine == want:
                if entry and not args.write_baseline:
                    fixed += 1
                    results.append((n, cid, "FIXED",
                                    "was %s in the baseline; the backlog "
                                    "shrank -- re-record it" % entry.get(
                                        "status", "open")))
                else:
                    tag = ("ok" + (" (-%s)" % ",".join(applied)
                                   if applied else ""))
                    results.append((n, cid, "ok", tag))
                continue

            at = first_diff_at(want, mine)
            dig = diff_digest(want, mine)
            detail = {"applied": applied, "at": at, "digest": dig,
                      "reference_excerpt": excerpt(want, at),
                      "runner_excerpt": excerpt(mine, at)}
            if entry is None:
                new_drift += 1
                results.append((n, cid, "NEW-DRIFT", detail))
            elif entry.get("digest") != dig:
                changed += 1
                results.append((n, cid, "CHANGED", detail))
            else:
                detail["status"] = entry.get("status", "open")
                results.append((n, cid, "known", detail))
            if args.show:
                detail["show"] = (want, mine)

    # ---- allowlist rot
    #
    # Citations are checked for every SELECTED family, oracle or not: they are
    # a static property of the tree, so CI catches a citation that rotted
    # under a family it cannot fetch. UNUSED needs an actual run, so it is
    # only asserted for families that produced results.
    rot = check_citations(devs, names)
    for n in names:
        for d in devs.get(n, []):
            if n in oracle_meta and not d.used and not args.raw:
                rot.append("UNUSED %s: the patch never fired. Runner no "
                           "longer takes this deviation (or the reference "
                           "changed) -- delete the entry." % d.id)

    for n, ferr in stale:
        rot.append("STALE ORACLE %s: --refresh could not reach the upstream "
                   "template, so the comparison used the cached copy "
                   "(fetched %s). %s"
                   % (n, oracle_meta.get(n, {}).get("fetched_at", "?"), ferr))

    # ---- upstream oracle movement
    moved = []
    for n, om in oracle_meta.items():
        was = base_oracles.get(n)
        if was and was.get("sha256") != om["sha256"]:
            moved.append("%s: upstream template changed (%s -> %s, %s). The "
                         "comparison basis moved; re-review, then re-record."
                         % (n, was.get("sha256"), om["sha256"], om["source"]))

    if args.write_baseline:
        if not_checked:
            print("REFUSING to re-record the baseline: %d families/cases were "
                  "not compared." % len(not_checked))
            for what, why in not_checked:
                print("  %-16s %s" % (what, why))
            print("A baseline written from a partial run would silently drop "
                  "the backlog of everything it could not check.")
            return EXIT_NOT_CHECKED
        return write_baseline(args, results, oracle_meta, base, known,
                              set(oracle_meta))

    return report(args, results, not_checked, structural, rot, moved,
                  oracle_meta, checked, new_drift, changed, fixed)


def report(args, results, not_checked, structural, rot, moved, oracle_meta,
           checked, new_drift, changed, fixed):
    per_family = {}
    for n, cid, verdict, detail in results:
        per_family.setdefault(n, []).append((cid, verdict, detail))
        if verdict == "ok":
            line = detail
        elif verdict in ("NEW-DRIFT", "CHANGED", "known"):
            line = "%s%s at byte %d%s" % (
                verdict,
                " [%s]" % detail["status"] if verdict == "known" else "",
                detail["at"],
                " (-%s applied)" % ",".join(detail["applied"])
                if detail["applied"] else "")
        else:
            line = "%s  %s" % (verdict, detail)
        print("  %-13s %-26s %s" % (n, cid, line))
        if isinstance(detail, dict) and detail.get("show"):
            print("      reference: %s" % repr(detail["show"][0]))
            print("      runner   : %s" % repr(detail["show"][1]))
        elif isinstance(detail, dict) and verdict in ("NEW-DRIFT", "CHANGED"):
            # a `known` entry's excerpt is already in the baseline file; only
            # the differences that need a decision get printed here.
            print("      reference: %s" % detail["reference_excerpt"])
            print("      runner   : %s" % detail["runner_excerpt"])

    print()
    for n in per_family:
        bad = [v for _c, v, _d in per_family[n]
               if v in ("NEW-DRIFT", "CHANGED", "FIXED")]
        kn = [v for _c, v, _d in per_family[n] if v == "known"]
        state = ("CLEAN" if not bad and not kn else
                 "%d known" % len(kn) if not bad else
                 "%d NEEDS ATTENTION, %d known" % (len(bad), len(kn)))
        print("%-15s %-28s [%s %s]" % (n, state, oracle_meta[n]["source"],
                                       oracle_meta[n]["sha256"]))
    for n, why in structural:
        print("%-15s %-28s %s" % (n, "no upstream oracle", why))

    known_total = sum(1 for _n, _c, v, _d in results if v == "known")
    print("\n%d cases compared, %d known-bug differences (backlog), "
          "%d new, %d changed, %d fixed-but-still-listed"
          % (checked, known_total, new_drift, changed, fixed))

    bad = False
    if not_checked:
        bad = True
        print("\nNOT CHECKED (%d) -- these produced no comparison at all:"
              % len(not_checked))
        for what, why in not_checked:
            print("  %-16s %s" % (what, why))
    if rot:
        bad = True
        print("\nALLOWLIST ROT (%d) -- scripts/template-conformance-"
              "allowlist.json:" % len(rot))
        for r in rot:
            print("  " + r)
    if moved:
        bad = True
        print("\nORACLE MOVED (%d):" % len(moved))
        for m in moved:
            print("  " + m)

    if new_drift or changed or fixed:
        print("\nFAIL: %d new, %d changed, %d fixed-but-still-listed."
              % (new_drift, changed, fixed))
        print("      New/changed differences are drift: fix the renderer, or "
              "-- if the deviation is deliberate and citable -- add it to "
              "the allowlist.")
        print("      Once you have decided, re-record the backlog:")
        print("        make template-conformance-baseline")
        return EXIT_DRIFT
    if rot or moved:
        print("\nFAIL: the allowlist or the oracle pins no longer match the "
              "tree. Nothing drifted, but the gate's own premises did.")
        return EXIT_DRIFT
    if not_checked:
        print("\nNOT CHECKED: %d families/cases were not compared. This is "
              "not a pass." % len(not_checked))
        return EXIT_NOT_CHECKED
    print("\nOK: every compared case is conformant or an accounted-for "
          "deviation; %d known bugs remain in the backlog." % known_total)
    return EXIT_OK


def write_baseline(args, results, oracle_meta, old, known, ran):
    # Entries for families this run did not cover are carried over verbatim.
    # Regenerating from `--family harmony` must not delete everybody else's
    # backlog.
    entries = [e for e in old.get("known_differences", [])
               if e["id"].split("/", 1)[0] not in ran]
    for n, cid, verdict, detail in results:
        if verdict not in ("NEW-DRIFT", "CHANGED", "known"):
            continue
        cid_full = "%s/%s" % (n, cid)
        prev = known.get(cid_full, {})
        entries.append({
            "id": cid_full,
            "status": prev.get("status", "open"),
            "note": prev.get("note", ""),
            "digest": detail["digest"],
            "allowlist_applied": detail["applied"],
            "first_diff_at": detail["at"],
            "reference_excerpt": detail["reference_excerpt"],
            "runner_excerpt": detail["runner_excerpt"],
        })
    entries.sort(key=lambda e: e["id"])
    oracles = dict(old.get("oracles", {}))
    # fetched_at is per-run noise; the sha is the pin worth committing.
    oracles.update({n: {"source": m["source"], "sha256": m["sha256"]}
                    for n, m in oracle_meta.items()})
    doc = {
        "version": 1,
        "_README": [
            "KNOWN DIFFERENCES between runner's native C renderer and each",
            "family's upstream jinja template. These are BUGS AWAITING FIXES,",
            "not intentional deviations -- intentional deviations live in",
            "scripts/template-conformance-allowlist.json with a citation, and",
            "never appear here.",
            "",
            "This file exists so the gate is useful TODAY: it fails on NEW",
            "drift while the pre-existing backlog stays visible and counted.",
            "Every entry is a bug ticket. The file should only ever shrink.",
            "",
            "Regenerate after fixing (or after deliberately changing) a case:",
            "    make template-conformance-baseline",
            "`status` and `note` are preserved across regeneration; the",
            "digest and excerpts are recomputed.",
            "",
            "`digest` fingerprints the (reference, runner) pair, with today's",
            "date neutralised. If it moves, the difference changed SHAPE and",
            "the gate says CHANGED rather than quietly accepting a second bug",
            "hiding behind the first.",
            "",
            "`oracles` pins the sha256 of each upstream template this backlog",
            "was recorded against. If upstream moves, the gate says so.",
        ],
        "status_values": {
            "open": "a bug nobody is working on right now",
            "in-progress": "someone is actively fixing this; see `note`",
        },
        "generated_by": "scripts/template-conformance.py --write-baseline",
        "generated_at": datetime.datetime.now(
            datetime.timezone.utc).replace(microsecond=0).isoformat(),
        "oracles": dict(sorted(oracles.items())),
        "known_differences": entries,
    }
    with open(args.baseline, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    print("wrote %s: %d known differences across %d families"
          % (os.path.relpath(args.baseline, ROOT), len(entries),
             len({e["id"].split("/")[0] for e in entries})))
    return EXIT_OK


def main():
    ap = argparse.ArgumentParser(
        description="byte-exact conformance of runner's C chat renderer "
                    "against each family's upstream jinja template")
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch the upstream oracles before comparing "
                         "(needs network, and models/ for the GGUF-sourced "
                         "families)")
    ap.add_argument("--oracle-dir",
                    default=os.path.join(ROOT, ".build", "template-oracles"),
                    help="oracle cache (gitignored; `make clean` wipes it)")
    ap.add_argument("--allowlist", default=ALLOWLIST)
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--family", action="append",
                    help="limit to these families (repeatable)")
    ap.add_argument("--network-only", action="store_true",
                    help="select only the families whose oracle is fetched "
                         "over the network, i.e. the ones checkable without "
                         "a local models/ tree")
    ap.add_argument("--show", action="store_true",
                    help="print both renders in full for differing cases")
    ap.add_argument("--raw", action="store_true",
                    help="do not subtract the allowlisted deviations")
    ap.add_argument("--write-baseline", action="store_true",
                    help="re-record the known-difference backlog")
    args = ap.parse_args()
    if args.network_only:
        args.family = (args.family or []) + list(NETWORK_ONLY)
    if args.write_baseline and args.raw:
        raise SystemExit("--write-baseline with --raw would record the "
                         "intentional deviations as bugs")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
