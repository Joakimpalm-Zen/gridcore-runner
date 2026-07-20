#!/usr/bin/env python3
"""Quality gate for a lossy KV cache: does q8 make the model WORSE?

The token-divergence gate (scripts/kernel-verify.py) answers a different and
much weaker question — how *often* q8 changes the output. A cache that changed
every token but kept every answer right would fail that gate; a cache that
changed one token and turned every answer wrong would pass it. Neither is what
we care about shipping.

So this script scores task success, f16 vs q8, on the two workloads the phase
actually claims ("long-context structured agents"):

  retrieval  needle-in-haystack. A fact is buried at a known depth in a long
             document full of same-shaped distractor facts, and the model must
             read it back. This is the workload that stresses a lossy cache
             hardest: the answer exists only in the KV cache, nowhere in the
             recent tokens, so any information the quantizer destroyed is
             information the model cannot recover.

  tools      structured tool selection. Pick the right function from a menu of
             plausible ones and fill its arguments, both with a short prompt
             and with a long irrelevant history in front of it. Scored on
             tool-name accuracy and on exact argument match.

Both suites run at temperature 0 on byte-identical prompts, so the KV cache
format is the only variable.

The comparison is PAIRED — every case is run under both cache types — so the
result is reported as McNemar discordant counts (f16-right/q8-wrong vs
f16-wrong/q8-right) with an exact two-sided p-value, not as two independent
accuracies whose confidence intervals happen to overlap. At these trial
counts, unpaired intervals would be far too wide to conclude anything.

    python scripts/kv-quality.py --models models/Qwen3-4B-Q4_K_M.gguf
    python scripts/kv-quality.py --all --out results.json --gate

Exit 0 unless --gate is passed and q8 is materially worse (see --max-drop).
"""

import argparse
import json
import math
import os
import random
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODELS = [
    "models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
    "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    "models/Qwen3-4B-Q4_K_M.gguf",
    "models/Phi-3.5-mini-instruct-Q4_K_M.gguf",
    "models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
]

# ---------------------------------------------------------------- haystack
#
# Filler is generated from a fixed seed so every model and every cache type
# sees byte-identical documents. It deliberately contains many sentences of
# the *same shape* as the needle ("The <kind> code for the <city> <site> is
# <5 digits>."), because a haystack whose only number is the answer tests
# nothing but the model's ability to find a digit.

CITIES = ["Zurich", "Lisbon", "Osaka", "Nairobi", "Bergen", "Quito", "Perth",
          "Tallinn", "Rabat", "Cusco", "Malmo", "Dakar", "Hobart", "Vilnius"]
SITES = ["vault", "depot", "annex", "hangar", "archive", "workshop", "silo"]
KINDS = ["maintenance", "shipping", "inventory", "calibration", "dispatch",
         "inspection", "storage", "routing"]

TOPICS = [
    "The quarterly logistics review covered freight consolidation across the "
    "northern corridor, where volumes rose modestly against a soft comparison.",
    "Field technicians reported that the replacement bearings arrived without "
    "the torque specification sheet, delaying the scheduled overhaul.",
    "A revised handling procedure now requires two signatures before any "
    "pallet leaves the staging floor, which has slowed throughput slightly.",
    "Weather routing software was updated to account for seasonal current "
    "shifts, and the crews were briefed on the new waypoints.",
    "The procurement team consolidated three supplier contracts into one "
    "framework agreement, trimming administrative overhead.",
    "Warehouse lighting was converted to low-draw fixtures over the summer "
    "shutdown, and the energy audit will be repeated in the spring.",
    "Training records were migrated to the new system, though several older "
    "certifications did not carry over cleanly and had to be re-entered.",
    "An intermittent fault in the conveyor controller was traced to a loose "
    "ground strap rather than the drive board, as first suspected.",
    "The safety committee recommended repainting the floor markings and "
    "adding mirrors at two blind corners near the loading bays.",
    "Inventory reconciliation found a small discrepancy in fastener counts, "
    "attributed to unlogged consumption during the retrofit.",
]


def make_filler(rng, n_sentences):
    out = []
    for _ in range(n_sentences):
        if rng.random() < 0.30:
            out.append("The %s code for the %s %s is %05d." % (
                rng.choice(KINDS), rng.choice(CITIES), rng.choice(SITES),
                rng.randrange(10000, 99999)))
        else:
            out.append(rng.choice(TOPICS))
    return out


def build_haystack(n_sentences, seed=20250720):
    """Deterministic filler sentence list, shared by every run."""
    return make_filler(random.Random(seed), n_sentences)


NEEDLES = [
    ("Zurich", "vault", 74921),
    ("Nairobi", "hangar", 30586),
    ("Tallinn", "archive", 61247),
]


def needle_sentence(city, site, code):
    return "The secret access code for the %s %s is %d." % (city, site, code)


def retrieval_prompt(sentences, city, site, code, depth):
    """Insert the needle at `depth` (0.0 = top, 1.0 = bottom) of the filler."""
    body = list(sentences)
    at = int(round(depth * len(body)))
    body.insert(at, needle_sentence(city, site, code))
    doc = " ".join(body)
    return ("Read the following document carefully.\n\n" + doc +
            "\n\nQuestion: What is the secret access code for the %s %s? "
            "Reply with only the digits, nothing else." % (city, site))


DIGITS = re.compile(r"\d+")


def retrieval_ok(reply, code):
    """First number in the reply must be the needle. Strict on purpose: a
    model that lists every code it saw has not retrieved anything."""
    found = DIGITS.findall(reply or "")
    return bool(found) and found[0] == str(code)


# ------------------------------------------------------------------- tools

def tool(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props,
                       "required": required}}}


S = {"type": "string"}
I = {"type": "integer"}
B = {"type": "boolean"}


def enum(*vals):
    return {"type": "string", "enum": list(vals)}


# A menu of plausible, overlapping tools. The distractors matter more than the
# targets: "search_email" vs "search_files" vs "search_calendar" is the choice
# an agent actually gets wrong, and it is the choice a degraded cache should
# get wrong more often.
TOOLBOX = [
    tool("search_email", "Search the user's email messages",
         {"query": S, "limit": I}, ["query"]),
    tool("search_files", "Search files stored in the document system",
         {"query": S, "folder": S}, ["query"]),
    tool("search_calendar", "Search calendar events by keyword",
         {"query": S, "days_ahead": I}, ["query"]),
    tool("create_event", "Create a calendar event",
         {"title": S, "date": S, "duration_minutes": I}, ["title", "date"]),
    tool("send_email", "Send an email message",
         {"to": S, "subject": S, "body": S}, ["to", "subject"]),
    tool("get_weather", "Get the current weather for a city",
         {"city": S, "units": enum("celsius", "fahrenheit")}, ["city"]),
    tool("get_stock_price", "Look up the latest share price for a ticker",
         {"ticker": S}, ["ticker"]),
    tool("convert_currency", "Convert an amount between two currencies",
         {"amount": I, "from_currency": S, "to_currency": S},
         ["amount", "from_currency", "to_currency"]),
    tool("translate_text", "Translate text into another language",
         {"text": S, "target_language": S}, ["text", "target_language"]),
    tool("set_reminder", "Set a reminder at a given time",
         {"text": S, "time": S}, ["text", "time"]),
    tool("create_ticket", "Open a support ticket",
         {"title": S, "priority": enum("low", "medium", "high", "urgent")},
         ["title", "priority"]),
    tool("close_ticket", "Close an existing support ticket",
         {"ticket_id": S, "resolution": S}, ["ticket_id"]),
    tool("run_query", "Run a read-only SQL query against the warehouse",
         {"sql": S, "database": S}, ["sql"]),
    tool("restart_service", "Restart a named background service",
         {"service": S, "force": B}, ["service"]),
    tool("list_invoices", "List invoices for a customer",
         {"customer": S, "unpaid_only": B}, ["customer"]),
]

# (user request, expected tool, {arg: expected value or None = presence only})
# Argument scoring is exact on values that have exactly one defensible form
# (a ticker, a city, a numeric amount, an enum member); free-text arguments
# like `query` and `body` are checked for presence only, because there is no
# single right phrasing and scoring them would measure style, not correctness.
TOOL_CASES = [
    ("What's the weather in Reykjavik right now?", "get_weather",
     {"city": "Reykjavik"}),
    ("How much is NVDA trading at?", "get_stock_price", {"ticker": "NVDA"}),
    ("Convert 250 euros to Japanese yen.", "convert_currency",
     {"amount": 250}),
    ("Find the message where Priya sent me the lease terms.", "search_email",
     {"query": None}),
    ("Look through my documents for the 2023 tax worksheet.", "search_files",
     {"query": None}),
    ("Do I have anything on the calendar about the Baker audit?",
     "search_calendar", {"query": None}),
    ("Book a 30 minute sync called Roadmap Review on 2026-03-04.",
     "create_event", {"title": "Roadmap Review", "date": "2026-03-04"}),
    ("Email jonas@example.com with the subject Invoice 4471 and let him know "
     "it's been paid.", "send_email",
     {"to": "jonas@example.com", "subject": "Invoice 4471"}),
    ("Translate 'the shipment has cleared customs' into Portuguese.",
     "translate_text", {"target_language": None}),
    ("Remind me to submit the timesheet at 4pm on Friday.", "set_reminder",
     {"text": None, "time": None}),
    ("The build server is down, open an urgent ticket for it.",
     "create_ticket", {"priority": "urgent"}),
    ("Ticket SUP-8812 is resolved, you can close it out.", "close_ticket",
     {"ticket_id": "SUP-8812"}),
    ("Pull the total orders per region from the warehouse.", "run_query",
     {"sql": None}),
    ("Bounce the nginx service please.", "restart_service",
     {"service": "nginx"}),
    ("Show me every unpaid invoice for Meridian Freight.", "list_invoices",
     {"customer": "Meridian Freight", "unpaid_only": True}),
    ("What is the share price of TSLA today?", "get_stock_price",
     {"ticker": "TSLA"}),
    ("I need 1200 US dollars in Swiss francs.", "convert_currency",
     {"amount": 1200}),
    ("Is it raining in Lagos?", "get_weather", {"city": "Lagos"}),
    ("Search my inbox for the flight confirmation from Delta.",
     "search_email", {"query": None}),
    ("Put a hold on my calendar for Dentist on 2026-01-19.", "create_event",
     {"title": "Dentist", "date": "2026-01-19"}),
    ("Restart the postgres service, and force it if you have to.",
     "restart_service", {"service": "postgres"}),
    ("Open a low priority ticket titled Update office wifi password.",
     "create_ticket", {"priority": "low"}),
    ("Find the contract PDF in the legal folder.", "search_files",
     {"query": None}),
    ("Nudge me to call the plumber tomorrow at 9am.", "set_reminder",
     {"text": None, "time": None}),
]

# Filler chat turns used to push the real request far back in the context.
PAD_TOPICS = [
    ("Can you summarize what we agreed about the vendor migration?",
     "You agreed to stage the migration over three weekends, keep the legacy "
     "endpoint reachable through the second stage, and revisit the cutover "
     "date once the load tests finish."),
    ("What were the concerns raised about the reporting latency?",
     "The main concern was that nightly aggregation finishes too close to the "
     "start of business, leaving no room to rerun a failed job before users "
     "notice stale numbers."),
    ("Remind me how the on-call rotation works now.",
     "It is a weekly rotation with a secondary responder, handover happens "
     "Tuesday morning, and anyone who takes a page after midnight gets the "
     "following afternoon off."),
    ("Was there a decision on the archive retention window?",
     "Retention was left at seven years for financial records and shortened "
     "to eighteen months for build artifacts, with an exception process for "
     "anything under active audit."),
    ("What is blocking the warehouse refactor?",
     "It is waiting on the schema freeze, which itself is waiting on the "
     "product team to confirm whether the regional split is permanent."),
]


def tool_messages(pad_turns):
    msgs = []
    for i in range(pad_turns):
        u, a = PAD_TOPICS[i % len(PAD_TOPICS)]
        # vary the turn so a long pad is not a repeated block; the point is to
        # fill the cache with real, varied tokens
        msgs.append({"role": "user", "content": "%s (item %d)" % (u, i + 1)})
        msgs.append({"role": "assistant", "content": "%s (noted as item %d)"
                     % (a, i + 1)})
    return msgs


def norm(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().strip(".").lower()
    if isinstance(v, (int, float)):
        return float(v)
    return v


def score_tool_call(call, want_name, want_args):
    """Returns (name_ok, args_ok). args_ok requires name_ok."""
    if call is None:
        return False, False
    if call.get("name") != want_name:
        return False, False
    try:
        got = json.loads(call.get("arguments") or "{}")
    except (ValueError, TypeError):
        return True, False
    if not isinstance(got, dict):
        return True, False
    for k, want in want_args.items():
        if k not in got:
            return True, False
        if want is None:
            if got[k] in (None, "", [], {}):   # presence-only
                return True, False
            continue
        if norm(got[k]) != norm(want):
            return True, False
    return True, True


# -------------------------------------------------------------------- http

class Server:
    def __init__(self, binary, model, ctx, kv, port, gpu="auto", verbose=False):
        self.binary, self.model, self.ctx = binary, model, ctx
        self.kv, self.port, self.gpu = kv, port, gpu
        self.verbose = verbose
        self.proc = None
        self.log = None

    def __enter__(self):
        logdir = Path(os.environ.get("TMPDIR", "/tmp"))
        logpath = logdir / ("kvq-%s-%d.log" % (self.kv, self.port))
        self.log = open(logpath, "w")
        cmd = [str(Path(self.binary).resolve()), "--serve",
               "--port", str(self.port), "-m", str(self.model),
               "-c", str(self.ctx), "--kv", self.kv, "--gpu", self.gpu]
        if self.verbose:
            print("    $ " + " ".join(cmd), file=sys.stderr)
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=self.log,
                                     start_new_session=True)
        deadline = time.time() + 900
        while time.time() < deadline:
            if self.proc.poll() is not None:
                self.log.flush()
                raise RuntimeError("server exited (%d):\n%s" % (
                    self.proc.returncode, logpath.read_text()[-2000:]))
            try:
                with socket.create_connection(("127.0.0.1", self.port), 1):
                    return self
            except OSError:
                time.sleep(0.4)
        raise RuntimeError("server did not come up on port %d" % self.port)

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except OSError:
                self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except OSError:
                    self.proc.kill()
        if self.log:
            self.log.close()
        return False

    def post(self, path, body, timeout=3600):
        req = urllib.request.Request(
            "http://127.0.0.1:%d%s" % (self.port, path),
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())

    def count_tokens(self, messages):
        r = self.post("/v1/messages/count_tokens",
                      {"model": "x", "messages": messages})
        return r.get("input_tokens", 0)

    def chat(self, messages, tools=None, max_tokens=64):
        body = {"model": "x", "messages": messages, "temperature": 0,
                "max_tokens": max_tokens, "seed": 1234}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        r = self.post("/v1/chat/completions", body)
        msg = r["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        return {
            "content": msg.get("content") or "",
            "call": calls[0]["function"] if calls else None,
            "prompt_tokens": r.get("usage", {}).get("prompt_tokens", 0),
        }


# ------------------------------------------------------------------ suites

_SIZE_CACHE = {}


def size_haystack(server, length):
    """Sentences needed so the retrieval prompt lands near `length` tokens.

    Measured with the model's own tokenizer via /v1/messages/count_tokens
    rather than a chars-per-token guess, so the reported context lengths are
    real. Cached per (model, length): it costs a handful of round trips.
    """
    key = (str(server.model), length)
    if key in _SIZE_CACHE:
        return _SIZE_CACHE[key]

    def toks(n):
        p = retrieval_prompt(build_haystack(n), *NEEDLES[0], 0.5)
        return server.count_tokens([{"role": "user", "content": p}])

    lo, hi = 8, 16
    while toks(hi) < length and hi < 400000:
        lo, hi = hi, hi * 2
    while lo < hi - 4:
        mid = (lo + hi) // 2
        if toks(mid) < length:
            lo = mid
        else:
            hi = mid
    _SIZE_CACHE[key] = lo
    return lo


def run_retrieval(server, lengths, depths, progress):
    rows = []
    for length in lengths:
        n_sent = size_haystack(server, length)
        sentences = build_haystack(n_sent)
        for (city, site, code) in NEEDLES:
            for depth in depths:
                p = retrieval_prompt(sentences, city, site, code, depth)
                out = server.chat([{"role": "user", "content": p}],
                                  max_tokens=24)
                rows.append({
                    "suite": "retrieval",
                    "id": "len%d/d%03d/%s" % (length, int(depth * 100), city),
                    "target_len": length,
                    "prompt_tokens": out["prompt_tokens"],
                    "depth": depth,
                    "ok": retrieval_ok(out["content"], code),
                    "reply": (out["content"] or "")[:80],
                })
                progress(rows[-1])
    return rows


def run_tools(server, pads, progress):
    rows = []
    for pad in pads:
        pad_msgs = tool_messages(pad)
        for i, (req, want_name, want_args) in enumerate(TOOL_CASES):
            msgs = pad_msgs + [{"role": "user", "content": req}]
            out = server.chat(msgs, tools=TOOLBOX, max_tokens=128)
            name_ok, args_ok = score_tool_call(out["call"], want_name,
                                               want_args)
            rows.append({
                "suite": "tools",
                "id": "pad%d/case%02d" % (pad, i),
                "pad_turns": pad,
                "prompt_tokens": out["prompt_tokens"],
                "ok": args_ok,
                "name_ok": name_ok,
                "want": want_name,
                "got": (out["call"] or {}).get("name"),
                "reply": ((out["call"] or {}).get("arguments") or "")[:80],
            })
            progress(rows[-1])
    return rows


# -------------------------------------------------------------- statistics

def mcnemar_p(b, c):
    """Exact two-sided McNemar (binomial sign test on discordant pairs).

    b = f16 right / q8 wrong, c = f16 wrong / q8 right. Concordant pairs carry
    no information about a difference and are correctly excluded.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def paired_ci(n, b, c):
    """95% CI (percentage points) on the paired accuracy difference q8 - f16.

    A null result is only worth reporting alongside the effect it could have
    detected. Only discordant pairs carry information, so the width is set by
    b + c, not by n: 240 trials with 7 discordant pairs is a tight bound,
    240 trials with 120 would not be.
    """
    if not n:
        return 0.0, 0.0
    diff = 100.0 * (c - b) / n
    half = 100.0 * 1.96 * math.sqrt(float(b + c)) / n
    return diff - half, diff + half


def pair(rows_a, rows_b, key="ok"):
    """Pair two runs by case id -> (n, a_ok, b_ok, a_only, b_only)."""
    ia = {r["id"]: r for r in rows_a}
    ib = {r["id"]: r for r in rows_b}
    ids = [i for i in ia if i in ib]
    a_ok = sum(1 for i in ids if ia[i][key])
    b_ok = sum(1 for i in ids if ib[i][key])
    disc_b = sum(1 for i in ids if ia[i][key] and not ib[i][key])
    disc_c = sum(1 for i in ids if not ia[i][key] and ib[i][key])
    return len(ids), a_ok, b_ok, disc_b, disc_c


def train_ctx(binary, model):
    """Read the model's training context so the sweep never silently turns on
    YaRN rope scaling — that would change quality for reasons unrelated to the
    KV cache and make the comparison meaningless."""
    p = subprocess.run([str(Path(binary).resolve()), "-m", str(model), "-v",
                        "-p", "x", "-n", "1", "--temp", "0", "--gpu", "off"],
                       capture_output=True, text=True, timeout=900)
    m = re.search(r"context\s+\d+\s+\(train (\d+)\)", p.stdout + p.stderr)
    return int(m.group(1)) if m else 4096


def free_port(start):
    for p in range(start, start + 500):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("no free port")


def subset(rows, suite, **filt):
    out = [r for r in rows if r["suite"] == suite]
    for k, v in filt.items():
        out = [r for r in out if r.get(k) == v]
    return out


def pair_pooled(results, suite, key="ok"):
    n = ao = bo = db = dc = 0
    for r in results.values():
        a, b = r["runs"].get("f16", []), r["runs"].get("q8", [])
        if suite:
            a, b = subset(a, suite), subset(b, suite)
        if not a or not b:
            continue
        pn, pa_, pb_, pdb, pdc = pair(a, b, key)
        n, ao, bo, db, dc = n + pn, ao + pa_, bo + pb_, db + pdb, dc + pdc
    return n, ao, bo, db, dc


def report(results):
    print("\n" + "=" * 78)
    print("QUALITY: f16 vs q8 KV cache (paired, temperature 0)")
    print("=" * 78)
    for name, r in results.items():
        if "f16" not in r["runs"] or "q8" not in r["runs"]:
            continue
        a, b = r["runs"]["f16"], r["runs"]["q8"]
        print("\n%s" % name)
        print("  %-34s %9s %9s %9s %7s" % ("slice", "f16", "q8", "delta", "p"))
        groups = []
        for length in sorted({x["target_len"] for x in a
                              if x["suite"] == "retrieval"}):
            ga = subset(a, "retrieval", target_len=length)
            toks = [x["prompt_tokens"] for x in ga]
            groups.append(("retrieval @%d tok" % (max(toks) if toks else length),
                           ga, subset(b, "retrieval", target_len=length)))
        for pad in sorted({x["pad_turns"] for x in a if x["suite"] == "tools"}):
            ga = subset(a, "tools", pad_turns=pad)
            toks = [x["prompt_tokens"] for x in ga]
            groups.append(("tools pad=%d (~%d tok)" % (pad, max(toks) if toks
                                                       else 0),
                           ga, subset(b, "tools", pad_turns=pad)))
        for label, ga, gb in groups:
            n, ao, bo, db, dc = pair(ga, gb)
            if not n:
                continue
            print("  %-34s %4d/%-4d %4d/%-4d %+7.1fpp %7.3f" % (
                label, ao, n, bo, n, 100.0 * (bo - ao) / n,
                mcnemar_p(db, dc)))
        n, ao, bo, db, dc = pair(a, b)
        if n:
            print("  %-34s %4d/%-4d %4d/%-4d %+7.1fpp %7.3f  discordant %d/%d"
                  % ("ALL", ao, n, bo, n, 100.0 * (bo - ao) / n,
                     mcnemar_p(db, dc), db, dc))

    print("\n" + "-" * 78)

    def pooled_line(label, n, ao, bo, db, dc):
        lo, hi = paired_ci(n, db, dc)
        print("POOLED %-10s f16 %4d/%-4d  q8 %4d/%-4d  %+6.1fpp  "
              "95%% CI [%+.1f, %+.1f]pp  discordant %d/%d  p=%.3f" % (
                  label, ao, n, bo, n, 100.0 * (bo - ao) / n, lo, hi,
                  db, dc, mcnemar_p(db, dc)))

    for suite in ("retrieval", "tools", None):
        n, ao, bo, db, dc = pair_pooled(results, suite)
        if n:
            pooled_line(suite or "ALL", n, ao, bo, db, dc)
    # tool-name accuracy separately: picking the wrong function is a different
    # and more serious failure than fumbling an argument
    n, ao, bo, db, dc = pair_pooled(results, "tools", key="name_ok")
    if n:
        pooled_line("tool-name", n, ao, bo, db, dc)


def gate(results, args):
    n, ao, bo, db, dc = pair_pooled(results, None)
    if not n:
        print("\nGATE: no paired results", file=sys.stderr)
        return 1
    drop = 100.0 * (ao - bo) / n
    p = mcnemar_p(db, dc)
    lo, hi = paired_ci(n, db, dc)
    print("\nGATE: q8 accuracy delta %+.1fpp, 95%% CI [%+.1f, %+.1f]pp "
          "(allowed -%.1f), McNemar p=%.3f"
          % (-drop, lo, hi, args.max_drop, p))
    if drop > args.max_drop and p < args.alpha:
        print("GATE FAIL: q8 KV is materially and significantly worse than f16")
        return 1
    print("GATE PASS")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=str(ROOT / "runner"))
    ap.add_argument("--models", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--suites", default="retrieval,tools")
    ap.add_argument("--lengths", default="4096,16384,32768",
                    help="target retrieval context lengths in tokens; "
                         "lengths above a model's training context are "
                         "skipped for that model")
    ap.add_argument("--depths", default="0,25,50,75,100",
                    help="needle depth percentages")
    ap.add_argument("--pads", default="0,24",
                    help="tool-suite padding turns (0 = short prompt)")
    ap.add_argument("--kv", default="f16,q8")
    ap.add_argument("--gpu", default="auto")
    ap.add_argument("--port", type=int, default=18400)
    ap.add_argument("--out", default="")
    ap.add_argument("--gate", action="store_true",
                    help="exit 1 if q8 is materially worse than f16")
    ap.add_argument("--max-drop", type=float, default=5.0,
                    help="allowed pooled accuracy drop in percentage points")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--report", default="",
                    help="comma-separated result JSON files: merge and "
                         "re-report them instead of running anything. These "
                         "sweeps take hours, so re-analysis must not require "
                         "re-measurement.")
    args = ap.parse_args()

    if args.report:
        merged = {}
        for f in args.report.split(","):
            for name, r in json.loads(Path(f).read_text()).items():
                slot = merged.setdefault(
                    name, {"train_ctx": r["train_ctx"], "ctx": r["ctx"],
                           "runs": {}})
                for kv, rows in r["runs"].items():
                    slot["runs"].setdefault(kv, []).extend(rows)
        report(merged)
        return gate(merged, args) if args.gate else 0

    models = ([m for m in args.models.split(",") if m] or
              (DEFAULT_MODELS if args.all else DEFAULT_MODELS[:1]))
    suites = args.suites.split(",")
    kvs = args.kv.split(",")
    lengths = [int(x) for x in args.lengths.split(",") if x]
    depths = [int(x) / 100.0 for x in args.depths.split(",") if x != ""]
    pads = [int(x) for x in args.pads.split(",") if x != ""]

    results = {}
    for model in models:
        mp = Path(model)
        if not mp.is_absolute():
            mp = ROOT / model
        if not mp.exists():
            print("skip (missing): %s" % model, file=sys.stderr)
            continue
        name = mp.name
        tctx = train_ctx(args.binary, mp)
        # Only the retrieval suite drives the served context. Sizing the cache
        # for lengths no suite will use would cost VRAM (and, once the cache
        # no longer fits, GPU layers) for nothing.
        mlens = ([l for l in lengths if l + 512 <= tctx]
                 if "retrieval" in suites else [])
        if "retrieval" in suites and not mlens:
            print("  %s: train ctx %d, no requested retrieval length fits"
                  % (name, tctx), file=sys.stderr)
        need = (max(mlens) + 1024) if mlens else 0
        ctx = min(max(need, 8192), tctx)
        results[name] = {"train_ctx": tctx, "ctx": ctx, "runs": {}}
        print("\n=== %s (train ctx %d, serving -c %d) ===" % (name, tctx, ctx))

        for kv in kvs:
            port = free_port(args.port)
            t0 = time.time()
            rows = []

            def progress(r, kv=kv):
                if args.verbose:
                    print("    [%s] %-28s %s %s" % (
                        kv, r["id"], "ok  " if r["ok"] else "MISS",
                        (r.get("reply") or "")[:48].replace("\n", " ")),
                        flush=True)

            with Server(args.binary, mp, ctx, kv, port, args.gpu,
                        args.verbose) as srv:
                if "retrieval" in suites and mlens:
                    rows += run_retrieval(srv, mlens, depths, progress)
                if "tools" in suites:
                    rows += run_tools(srv, pads, progress)
            results[name]["runs"][kv] = rows
            n_ok = sum(1 for r in rows if r["ok"])
            print("  %-4s %3d/%3d ok  (%.0fs)" % (kv, n_ok, len(rows),
                                                  time.time() - t0),
                  flush=True)
        if args.out:   # checkpoint after every model; these runs are long
            Path(args.out).write_text(json.dumps(results, indent=1))

    report(results)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))
        print("\nraw results -> %s" % args.out)
    return gate(results, args) if args.gate else 0


if __name__ == "__main__":
    sys.exit(main())
