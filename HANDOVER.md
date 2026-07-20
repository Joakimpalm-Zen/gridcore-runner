# Handover — runner performance/hardening session, 2026-07-20

Everything this session produced is **already merged into `main`** (tip at
handover time: `c1ad553`, now an ancestor of `e81bd24`). The branch
`session/2026-07-20-runner-perf` is only a marker and contains no unique
commits — that is why it looks empty. This file is the actual handover.

Line numbers below were **verified against `origin/main` at `e81bd24`**.
The 8 commits after `c1ad553` touched `src/tokenizer.c`, templates and
tests only — `src/json.c` is untouched, so these still resolve.

---

## 1. UNFINISHED AND IMPORTANT — three confirmed crash-on-OOM bugs

All three are in `src/json.c`, all reachable from **untrusted HTTP input**
(request bodies) or untrusted files, so each is a remote crash when an
allocation fails. This engine deliberately runs near memory limits
(`--reserve`, multi-GB weights, hybrid GPU/CPU splits), so allocation
failure is a normal operating condition here, not an exotic one.

I found and verified these by reading the code; **I did not fix them.**

### 1.1 `src/json.c:75` + `:77` — string parse (crash + leak)

```c
char *out = malloc(cap);                                    // :75 unchecked
while (...) {
    if (m + 8 > cap) { cap *= 2; out = realloc(out, cap); } // :77 unchecked
    ...
    if (ch == '"') { c->p++; out[m] = 0; return out; }      // dereferences
```

Two defects: the `malloc` result is never checked, and
`out = realloc(out, cap)` is the classic idiom that **loses the original
pointer** when realloc fails (leak) and then yields NULL (crash on the
next write). Driven by long strings in an untrusted body.

**Fix:** check the `malloc`; on realloc failure `free(out)` and return
`NULL`. This is clean because `parse_string` already returns `NULL` for
malformed input and **every caller already handles that** — failure has a
path home. Use a temp: `char *tmp = realloc(out, cap); if (!tmp) { free(out); return NULL; } out = tmp;`

### 1.2 `src/json.c:187` — array/object element append (crash)

```c
v->items = realloc(v->items, sizeof(jv *) * (v->n + 1));  // :187 unchecked
if (v->type == J_OBJ)
    v->keys = realloc(v->keys, sizeof(char *) * (v->n + 1));  // :189 unchecked
v->items[v->n] = item;                                    // dereferences
```

Driven by element count from untrusted input.

**Fix:** same temp-pointer pattern; on failure `free(key)` (as the
existing error path does) and `goto fail` — **the `fail` label already
exists in this function**, and it frees the partially built value.

### 1.3 `src/json.c:310` — `sb_put`, the HTTP **response** builder (crash) — NEEDS A DECISION

```c
void sb_put(sbuf *b, const char *s, size_t n) {
    if (b->n + n + 1 > b->cap) {
        b->cap = (b->n + n + 1) * 2 + 256;
        b->s = realloc(b->s, b->cap);   // :310 unchecked
    }
    memcpy(b->s + b->n, s, n);          // dereferences
```

This one is on the path that assembles **every HTTP response body**.

It cannot be fixed as locally as the others: `sb_put` returns `void` and
`sbuf` is `typedef struct sbuf { char *s; size_t n, cap; } sbuf;`
(`src/json.h:33`), so there is nowhere to report failure.

**My recommendation (deliberately left for you to confirm):** add an `ok`
flag to `sbuf`, make the whole `sb_*` family a no-op once it is false, and
have the `server.c` callers check it and answer **500**.

**Do not silently truncate the response.** A truncated body that still
looks like a success is worse than a crash — the client parses garbage and
cannot tell anything went wrong. If you prefer a different contract (e.g.
`sb_put` returning `bool`), that is fine, but the failure must reach the
client as an error.

### 1.4 Wider survey — needs triage, NOT verified

A heuristic scan flagged ~70 further `malloc`/`calloc`/`realloc` sites
without a *nearby* NULL check across `model.c`, `server.c`, `tokenizer.c`,
`schema.c`, `cuda.c`. **Many are probably false positives** (checked a few
lines further away, or guarded by an existing `CK(...)` / `w.ok` pattern).
Only the three above are confirmed. Worth a proper audit; do not mass-patch
from a grep.

A codex read-only audit of this was attempted and **died at the OS level**
(`0xC0000142` — its sandbox helper cannot start processes while another
codex instance is running). It correctly refused to invent findings. Re-run
it **serially**.

---

## 2. NOT STARTED — fuzz/property tests (design ready)

Every parser here is hand-written and eats untrusted input, which makes
this the highest-value remaining hardening work.

Targets, in priority order, with entry points:

| target | entry point | note |
|---|---|---|
| `json_parse` | `src/json.c:228` | bounded by `n`, does **not** require NUL termination (existing tests rely on this) |
| `schema_compile` | `src/runner.h:344` | feed it JSON parsed from fuzz input; free with `schema_free` |
| `sval_feed` | `src/schema.c:827` | streaming schema validator; compile a schema, then feed arbitrary chunks |
| `jsonv_feed` | `src/jsonmode.c:245` | streaming JSON validator used by `--json` |
| `gguf_open` | `src/gguf.c:84` | takes a path, so the harness must write the buffer to a temp file; must **reject**, never crash |
| `tok_encode` | `src/tokenizer.c:417` | needs a loaded tokenizer — skip unless it stays clean. NOTE: `tokenizer.c` changed heavily after my session; re-read it first |

Plan: libFuzzer harnesses in `tests/fuzz/` (one per target, each freeing
everything so leak/UAF reports mean something); small **committed** seed
corpora in `tests/fuzz/corpus/<target>/` (for gguf seeds use
`scripts/make-test-model.py`, which supports `--zero-first-dim`,
`--wrap-first-offset`, `--suppress-all-but-eos`); a `make fuzz` target
building with clang `-fsanitize=fuzzer,address,undefined` and **short
bounded** runs (`-max_total_time=20 -rss_limit_mb=2048`) so it is usable in
CI; it must no-op with a clear message when clang is absent (the Windows
dev box is msys2/gcc) and must not break `make`, `make test` or
`make smoke`. Add a bounded ubuntu+clang CI job; do not modify existing CI
steps.

**Rule:** if a harness finds a real parser crash, **stop and report it with
the reproducing input** — do not quietly patch the parser. The finding is
worth more than the harness.

**Useful check:** the three `json.c` bugs in §1 should reproduce
immediately. If they do not, the harness is not reaching the code.

---

## 3. Gates you should use (they are why the kernel work was safe)

- `python scripts/kernel-verify.py --baseline <known-good.exe> --candidate <new.exe> --model <gguf>`
  → demands **token-identical** greedy output on 5 prompts. A faster or
  tidier binary that changes tokens is a **regression**, not a win. Copy
  the current `runner.exe` aside before building a candidate.
- `python scripts/kernel-bench.py --runner <exe> --model <gguf>`
  → prefill/decode tok/s as JSON.
- `RUNNER_CUDA_PROFILE=1` → per-phase GPU timings (matvec/attention/norms/
  memcpy/launch), off by default.
- `RUNNER_DEBUG_ACT=1` → per-layer activation stats incl. inf/nan counts.
  This is what proved a suspected gemma4 numerical bug did **not** exist.
- `RUNNER_FORCE_REQUANT=1`, `RUNNER_REQUANT_ONLY=<substr>` → let the
  quantizer do same-size conversions / only matching tensors, for
  bisecting a model group by group.

Reference numbers on an RTX 3070 (2k-token prompt), so you can spot a
regression: **Qwen3-4B-Q8 114 prefill / 59 decode**, **Qwen2.5-7B-Q4_K
112 / 36**, **Llama-3.1-8B-Q5_K_M 98 / 31**.

---

## 4. Loose ends

- `scripts/kernel-bench.py` uses a synthetic `item0000 item0001 ...`
  prompt; some instruct models emit EOS immediately on it and the script
  then reports decode as 0 tok/s (seen with Ministral-8B). Both binaries
  do it equally so A/B comparisons stay valid, but it reads like a
  regression. Give it a more natural prompt.
- `make test` skips the Python client tests unless `PYTHON` points at an
  interpreter with pytest (msys2's python lacks it):
  `make test PYTHON=/c/Users/zen/AppData/Local/Programs/Python/Python312/python.exe`.
  The skip is deliberate and non-fatal.
- Windows quirk already worked around in the Makefile: a `python3`
  **app-execution-alias stub** exists on PATH but fails when run, so the
  interpreter is probed by *running* it, not by `command -v`. Keep that.
- `docs/specs/2026-07-11-cuda-backend-design.md` is marked **historical**:
  it lists partial/layer-split offload as a non-goal (it shipped) and
  describes a kernel design that has been replaced. README is
  authoritative where they disagree.

---

## 5. Operating the assistant models (cost me real time — save yourself)

- Run **one `codex` at a time**. A second concurrent instance (even
  read-only) dies with `0xC0000142`.
- Always redirect stdin for background runs: `codex exec ... < /dev/null`.
  Without it, `codex exec` blocks waiting on stdin — this idled for 32
  minutes with 0% CPU before I noticed.
- Never let a subagent wait on a Monitor event or a background batch to
  wake it; it will simply stop. Give it foreground commands.
- Treat any agent's "verified / all green" as a **claim**. One reported
  `make smoke` passing when it failed here; another's confident root-cause
  hypothesis was refuted in three minutes by reading the code. Re-run the
  gates yourself.
