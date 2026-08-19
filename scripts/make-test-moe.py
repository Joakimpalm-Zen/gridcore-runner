#!/usr/bin/env python3
"""Generate a matched dense + MoE GGUF trio for the MoE equivalence test.

The three files share every weight except the FFN, and each MoE variant is
constructed to be MATHEMATICALLY IDENTICAL to the dense model's FFN, so the
runner's already-trusted dense path is the oracle — no separate reference
engine is needed. All weights are F32, so there is no quantization error and
the greedy output must be byte-identical.

  <out>.dense.gguf   plain llama, dense SwiGLU FFN with down = D
  <out>.moe1.gguf    llama+experts, expert_count=2 expert_used=1, zero router
                     (uniform logits -> top-1 picks expert 0 by index), expert 0
                     FFN == dense, expert 1 = zeros. Weight of the one expert is
                     1.0, so output == dense exactly. Validates: metadata parse,
                     router matmul+softmax+top-1 selection, 3D expert slice at
                     offset 0, SwiGLU, and that expert 1 does not leak in.
  <out>.moe2.gguf    expert_count=2 expert_used=2, zero router -> weights
                     [0.5, 0.5], BOTH experts == dense. 0.5*y + 0.5*y == y
                     exactly, so output == dense. Validates: top-2 selection,
                     renormalization summing to 1, the weighted multi-expert sum,
                     and reading BOTH expert slices (offsets 0 and 1).
"""
import struct
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "test-moe"
E, HEADS, KV, FF, LAYERS = 32, 4, 2, 64, 2
VOCAB = ["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)]
TTYPE = [2, 3, 3] + [6] * 256
U32, F32, STR, ARR, I32, BOOL = 4, 6, 8, 9, 5, 7
T_F32, T_MXFP4 = 0, 39


def s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def ku(k, v): return s(k) + struct.pack("<II", U32, v)
def kf(k, v): return s(k) + struct.pack("<If", F32, v)
def ks(k, v): return s(k) + struct.pack("<I", STR) + s(v)
def kb(k, v): return s(k) + struct.pack("<IB", BOOL, bool(v))
def kas(k, xs): return s(k) + struct.pack("<IIQ", ARR, STR, len(xs)) + b"".join(s(x) for x in xs)
def kaf(k, xs): return s(k) + struct.pack("<IIQ", ARR, F32, len(xs)) + struct.pack(f"<{len(xs)}f", *xs)
def kai(k, xs): return s(k) + struct.pack("<IIQ", ARR, I32, len(xs)) + struct.pack(f"<{len(xs)}i", *xs)
def kau(k, xs): return s(k) + struct.pack("<IIQ", ARR, U32, len(xs)) + struct.pack(f"<{len(xs)}I", *xs)
def kab(k, xs): return s(k) + struct.pack("<IIQ", ARR, BOOL, len(xs)) + bytes(1 if x else 0 for x in xs)


_seed = 0x50E
def rnd():
    global _seed
    _seed = (_seed * 1103515245 + 12345) & 0x7fffffff
    return (_seed / 0x7fffffff - .5) * .08


def flist(n): return [rnd() for _ in range(n)]
def pack(xs): return struct.pack(f"<{len(xs)}f", *xs)
def ones(n): return pack([1.0] * n)
def zeros(n): return pack([0.0] * n)


def mxfp4_payload(row_len, rows, seed):
    """Deterministic MXFP4 tensor payload with row_len as ne[0]."""
    assert row_len % 32 == 0
    out = bytearray()
    for r in range(rows):
        for b in range(row_len // 32):
            out.append(124 + ((seed + r + b) & 1))  # scales 2^-3 / 2^-2
            for j in range(16):
                lo = (seed + r * 3 + b * 5 + j) & 15
                hi = (seed + r * 7 + b * 11 + j + 5) & 15
                out.append((hi << 4) | lo)
    return bytes(out)


def base_meta(arch, extra):
    p = arch
    kvs = [
        ks("general.architecture", arch),
        ku(f"{p}.block_count", LAYERS), ku(f"{p}.context_length", 256),
        ku(f"{p}.embedding_length", E), ku(f"{p}.feed_forward_length", FF),
        ku(f"{p}.attention.head_count", HEADS), ku(f"{p}.attention.head_count_kv", KV),
        kf(f"{p}.attention.layer_norm_rms_epsilon", 1e-5),
        kf(f"{p}.rope.freq_base", 10000.0),
    ] + extra + [
        ks("tokenizer.ggml.model", "llama"), kas("tokenizer.ggml.tokens", VOCAB),
        kaf("tokenizer.ggml.scores", [0.0] * len(VOCAB)),
        kai("tokenizer.ggml.token_type", TTYPE), ku("tokenizer.ggml.bos_token_id", 1),
        ku("tokenizer.ggml.eos_token_id", 2), kb("tokenizer.ggml.add_bos_token", True),
    ]
    return kvs


def write(path, tensors, kvs):
    meta = b"".join(kvs)
    info, off = b"", 0
    for item in tensors:
        if len(item) == 3:
            name, dims, payload = item
            typ = T_F32
        else:
            name, dims, payload, typ = item
        info += s(name) + struct.pack("<I", len(dims))
        info += b"".join(struct.pack("<Q", d) for d in dims)
        info += struct.pack("<IQ", typ, off)
        off = (off + len(payload) + 31) & ~31
    head = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(kvs)) + meta + info
    with open(path, "wb") as f:
        f.write(head + b"\0" * ((-len(head)) % 32))
        for item in tensors:
            payload = item[2]
            f.write(payload)
            f.write(b"\0" * ((-len(payload)) % 32))
    print(f"wrote {path}")


# --- shared weights (generated once so all three files agree) -----------------
shared = [
    ("token_embd.weight", [E, len(VOCAB)], pack(flist(E * len(VOCAB)))),
    ("output_norm.weight", [E], ones(E)),
]
kv_dim = (E // HEADS) * KV
for i in range(LAYERS):
    shared += [
        (f"blk.{i}.attn_norm.weight", [E], ones(E)),
        (f"blk.{i}.attn_q.weight", [E, E], pack(flist(E * E))),
        (f"blk.{i}.attn_k.weight", [E, kv_dim], pack(flist(E * kv_dim))),
        (f"blk.{i}.attn_v.weight", [E, kv_dim], pack(flist(E * kv_dim))),
        (f"blk.{i}.attn_output.weight", [E, E], pack(flist(E * E))),
        (f"blk.{i}.ffn_norm.weight", [E], ones(E)),
    ]

# per-layer dense FFN weights (gate/up shared across experts; down is the oracle).
# Scaled up so the FFN materially drives the logits — otherwise the tiny random
# model is degenerate (argmax is FFN-independent) and the test cannot tell a
# correct MoE from a broken one.
ffn = {}
for i in range(LAYERS):
    ffn[i] = dict(gate=[v * 8 for v in flist(E * FF)],
                  up=[v * 8 for v in flist(E * FF)],
                  down=[v * 8 for v in flist(FF * E)])


def dense_ffn(i):
    return [
        (f"blk.{i}.ffn_gate.weight", [E, FF], pack(ffn[i]["gate"])),
        (f"blk.{i}.ffn_up.weight", [E, FF], pack(ffn[i]["up"])),
        (f"blk.{i}.ffn_down.weight", [FF, E], pack(ffn[i]["down"])),
    ]


def moe_ffn(i, n_expert, expert1_down):
    # gate_exps/up_exps: the shared gate/up, repeated per expert (ne {E, FF, n_expert})
    gate_exps = ffn[i]["gate"] * n_expert
    up_exps = ffn[i]["up"] * n_expert
    # down_exps ne {FF, E, n_expert}: expert 0 = dense down, expert 1 = expert1_down
    down_exps = list(ffn[i]["down"]) + list(expert1_down)
    return [
        (f"blk.{i}.ffn_gate_inp.weight", [E, n_expert], zeros(E * n_expert)),  # router
        (f"blk.{i}.ffn_gate_exps.weight", [E, FF, n_expert], pack(gate_exps)),
        (f"blk.{i}.ffn_up_exps.weight", [E, FF, n_expert], pack(up_exps)),
        (f"blk.{i}.ffn_down_exps.weight", [FF, E, n_expert], pack(down_exps)),
    ]


def moe_ffn_routed(i, n_expert):
    # A fixture that actually EXERCISES routing: a non-zero router and distinct
    # experts, so the softmax has real structure and the experts disagree.
    # Deliberately NOT dense-equivalent — this one exists for the fused-vs-eager
    # tolerance gate (test_moe_tol.c), which needs the two routing paths to have
    # something to differ about. The zero-router fixtures above are exactly
    # 0.5/0.5 either way and can only ever compare a path with itself.
    gate_exps, up_exps, down_exps = [], [], []
    for e in range(n_expert):
        gate_exps += [x * (1.0 + 0.25 * e) for x in ffn[i]["gate"]]
        up_exps += [x * (1.0 - 0.15 * e) for x in ffn[i]["up"]]
        down_exps += [x * (1.0 + 0.10 * e) for x in ffn[i]["down"]]
    return [
        (f"blk.{i}.ffn_gate_inp.weight", [E, n_expert], pack(flist(E * n_expert))),
        (f"blk.{i}.ffn_gate_exps.weight", [E, FF, n_expert], pack(gate_exps)),
        (f"blk.{i}.ffn_up_exps.weight", [E, FF, n_expert], pack(up_exps)),
        (f"blk.{i}.ffn_down_exps.weight", [FF, E, n_expert], pack(down_exps)),
    ]


def moe_ffn_split(i, n_expert, expert1_down):
    # legacy split layout: one 2D tensor per expert (older Mixtral GGUFs)
    downs = [ffn[i]["down"], expert1_down]
    tensors = [(f"blk.{i}.ffn_gate_inp.weight", [E, n_expert], zeros(E * n_expert))]
    for e in range(n_expert):
        tensors += [
            (f"blk.{i}.ffn_gate.{e}.weight", [E, FF], pack(ffn[i]["gate"])),
            (f"blk.{i}.ffn_up.{e}.weight", [E, FF], pack(ffn[i]["up"])),
            (f"blk.{i}.ffn_down.{e}.weight", [FF, E], pack(downs[e])),
        ]
    return tensors


dense = list(shared)
for i in range(LAYERS):
    dense += dense_ffn(i)
write(f"{OUT}.dense.gguf", dense, base_meta("llama", []))

# moe1: expert_used=1, expert 0 = dense, expert 1 = zeros
moe1 = list(shared)
for i in range(LAYERS):
    moe1 += moe_ffn(i, 2, [0.0] * (FF * E))
write(f"{OUT}.moe1.gguf", moe1,
      base_meta("llama", [ku("llama.expert_count", 2), ku("llama.expert_used_count", 1)]))

# moe2: expert_used=2, both experts = dense
moe2 = list(shared)
for i in range(LAYERS):
    moe2 += moe_ffn(i, 2, ffn[i]["down"])
write(f"{OUT}.moe2.gguf", moe2,
      base_meta("llama", [ku("llama.expert_count", 2), ku("llama.expert_used_count", 2)]))

# moe3: legacy SPLIT per-expert layout, expert_used=1, expert 0 = dense (like moe1)
moe3 = list(shared)
for i in range(LAYERS):
    moe3 += moe_ffn_split(i, 2, [0.0] * (FF * E))
write(f"{OUT}.moe3.gguf", moe3,
      base_meta("llama", [ku("llama.expert_count", 2), ku("llama.expert_used_count", 1)]))

# moe4: real router, distinct experts, top-2 of 4. Not dense-equivalent by
# design — its job is to give the fused-vs-eager routing comparison something
# to disagree about, which the zero-router fixtures structurally cannot.
moe4 = list(shared)
for i in range(LAYERS):
    moe4 += moe_ffn_routed(i, 4)
write(f"{OUT}.moe4.gguf", moe4,
      base_meta("llama", [ku("llama.expert_count", 4), ku("llama.expert_used_count", 2)]))

def moe_ffn_n(i, downs, probs_b=None):
    """N experts sharing the dense gate/up, each with its own `down` row block.

    Scaling only `down` keeps every expert a clean multiple of the dense FFN
    (down is linear, so down*k gives output*k exactly in fp32), which is what
    lets each router knob below be checked against the dense oracle.
    """
    n = len(downs)
    tensors = [
        (f"blk.{i}.ffn_gate_inp.weight", [E, n], zeros(E * n)),   # zero router
        (f"blk.{i}.ffn_gate_exps.weight", [E, FF, n], pack(ffn[i]["gate"] * n)),
        (f"blk.{i}.ffn_up_exps.weight", [E, FF, n], pack(ffn[i]["up"] * n)),
        (f"blk.{i}.ffn_down_exps.weight", [FF, E, n], pack(sum(downs, []))),
    ]
    if probs_b is not None:
        tensors.append((f"blk.{i}.exp_probs_b.weight", [n], pack(probs_b)))
    return tensors


def dn(i, k):
    """the dense down block scaled by k"""
    return [x * k for x in ffn[i]["down"]]


ZERO = None  # placeholder replaced per layer below


def emit(name, downs_fn, extra, probs_b=None):
    ts = list(shared)
    for i in range(LAYERS):
        ts += moe_ffn_n(i, downs_fn(i), probs_b)
    write(f"{OUT}.{name}.gguf", ts, base_meta("llama", extra))


ZEROS_FF_E = [0.0] * (FF * E)


# --- fixture self-checks. A fixture that CANNOT distinguish the knob it names
# is worse than no fixture: it reports a passing gate for a binary that does
# not implement the knob at all. Each check below models the runner's selection
# rule (src/model.c moe_route) independently and refuses to write a fixture
# whose knob makes no difference.

def _sel_topk(scores, k):
    """top-k by score, ties to the LOWEST index (moe_route's strict >)"""
    return sorted(range(len(scores)), key=lambda e: (-scores[e], e))[:k]


def _sel_grouped(scores, n_group, n_group_used, k):
    """group-limited top-k: keep the n_group_used groups with the largest sum
    of their top-2 scores, mask the rest out, then top-k over the survivors"""
    per = len(scores) // n_group
    gsum = []
    for g in range(n_group):
        blk = sorted(scores[g * per:(g + 1) * per], reverse=True)
        gsum.append(blk[0] + (blk[1] if per > 1 else 0.0))
    keep = set(sorted(range(n_group), key=lambda g: (-gsum[g], g))[:n_group_used])
    masked = [s if (e // per) in keep else -1e30 for e, s in enumerate(scores)]
    return _sel_topk(masked, k)


def check_group_sensitivity(probs_b, n_group, n_group_used, used, live):
    """Refuse a group-limited fixture that grouping does not change.

    The zero router gives every expert the SAME probability for every input
    (dot(0, x) == 0), so exp_probs_b alone decides selection and both answers
    are exact here — no probe input is needed. `live` is the set of experts
    whose down block is non-zero: the grouped pick must stay inside it (the
    fixture has to match the dense oracle) and the ungrouped pick must leave
    it (the miss has to be visible in the logits).
    """
    p = 1.0 / len(probs_b)                 # softmax over equal logits
    scores = [p + b for b in probs_b]
    grouped = _sel_grouped(scores, n_group, n_group_used, used)
    flat = _sel_topk(scores, used)
    if grouped == flat:
        sys.exit(f"make-test-moe: group fixture is degenerate — grouped and "
                 f"ungrouped selection are both {grouped}; a binary with no "
                 f"group-limited top-k would pass")
    if not set(grouped) <= live:
        sys.exit(f"make-test-moe: group fixture selects a zero expert "
                 f"{grouped} (live={sorted(live)}); it cannot match dense")
    if set(flat) <= live:
        sys.exit(f"make-test-moe: group fixture's ungrouped selection {flat} "
                 f"is all-live; the missed grouping would not change the logits")

# --- generalized-router knobs. Every fixture is built to come out EXACTLY
# equal to the dense oracle, so a knob that is ignored, applied twice, or
# applied to the wrong quantity shows up as a text mismatch.

# sigmoid gating: zero router -> sigmoid(0) = 0.5 for both, renormalized to
# 0.5/0.5. Both experts are the dense FFN, so 0.5y + 0.5y == y.
emit("gsigmoid", lambda i: [dn(i, 1.0), dn(i, 1.0)],
     [ku("llama.expert_count", 2), ku("llama.expert_used_count", 2),
      ku("llama.expert_gating_func", 2)])

# sqrt(softplus) gating: both experts score sqrt(ln 2); renormalization makes
# that 0.5/0.5 again. Checks the branch runs and still normalizes.
emit("gsqrtsp", lambda i: [dn(i, 1.0), dn(i, 1.0)],
     [ku("llama.expert_count", 2), ku("llama.expert_used_count", 2),
      ku("llama.expert_gating_func", 4)])

# softmax-over-weights: logits pass through ungated, and the softmax runs on
# the two SELECTED weights (0,0) -> 0.5/0.5.
emit("gsmw", lambda i: [dn(i, 1.0), dn(i, 1.0)],
     [ku("llama.expert_count", 2), ku("llama.expert_used_count", 2),
      ku("llama.expert_gating_func", 3)])

# expert_weights_norm = false: 4 experts, top-2, softmax gives 0.25 each and
# NOTHING renormalizes them. Experts are 2x the dense FFN, so
# 0.25*2y + 0.25*2y == y. If the renormalize ran anyway the weights would be
# 0.5 each and the output would come out 2x.
emit("gnonorm", lambda i: [dn(i, 2.0)] * 4,
     [ku("llama.expert_count", 4), ku("llama.expert_used_count", 2),
      kb("llama.expert_weights_norm", False)])

# expert_weights_scale = 2: weights 0.5/0.5 become 1.0/1.0 against half-sized
# experts, so 1.0*(y/2) + 1.0*(y/2) == y.
emit("gscale", lambda i: [dn(i, 0.5), dn(i, 0.5)],
     [ku("llama.expert_count", 2), ku("llama.expert_used_count", 2),
      kf("llama.expert_weights_scale", 2.0)])

# exp_probs_b steers SELECTION ONLY. Expert 0 is zeros and expert 1 is the
# dense FFN; the bias [-1, +1] must flip top-1 from index 0 to index 1. The
# weight still comes from the UNBIASED probability (0.5, renormalized to 1.0),
# so a bias that leaked into the weights would change the magnitude too.
emit("gprobsb", lambda i: [ZEROS_FF_E, dn(i, 1.0)],
     [ku("llama.expert_count", 2), ku("llama.expert_used_count", 1)],
     probs_b=[-1.0, 1.0])

# group-limited top-k: 4 experts in 2 groups, one group kept. The bias makes
# group 1 win on its top-2 sum, so selection must land on expert 2 — the only
# non-zero one. Getting the grouping wrong selects a zero expert and the
# output collapses.
#
# The bias is NOT free to be any group-1-favoring vector. The first version of
# this fixture used [0, 0, 1, 1]: group 1 won, but plain ungrouped top-1 also
# landed on expert 2 (same tie, same lowest-index tie-break), so a binary with
# no group-limited top-k at all produced the dense-identical output and passed.
# check_group_sensitivity below now proves the two selections disagree before
# the fixture is written.
GGROUP_PROBS_B = [0.0, 5.0, 4.0, 4.0]     # ungrouped top-1 -> expert 1 (zeros)
check_group_sensitivity(GGROUP_PROBS_B, n_group=2, n_group_used=1, used=1,
                        live={2})
emit("ggroup", lambda i: [ZEROS_FF_E, ZEROS_FF_E, dn(i, 1.0), ZEROS_FF_E],
     [ku("llama.expert_count", 4), ku("llama.expert_used_count", 1),
      ku("llama.expert_group_count", 2), ku("llama.expert_group_used_count", 1)],
     probs_b=GGROUP_PROBS_B)

# --- --prune-experts oracle-equivalence fixture. A zero router makes every
# expert's raw probability equal REGARDLESS OF INPUT (dot(0, x) == 0 for any
# x), so exp_probs_b alone decides top-k selection — PROVABLY, not just
# empirically for whatever prompt a test happens to try. Bias
# [1000, 500, 0, -1000] with top-2 of 4 always selects {0, 1} and NEVER
# {2, 3}, for every token at every position. Each expert's down projection
# is its own independent pseudo-random vector (NOT a scalar multiple of the
# others — a shared direction scaled differently survives the layer's
# post-FFN RMSNorm unchanged, which the first version of this fixture
# learned the hard way: it picked a real used expert and still produced a
# byte-identical generation). Distinct directions have no such escape:
# swapping which two get averaged changes the normalized output.
emit("pruneprobe", lambda i: [flist(FF * E) for _ in range(4)],
     [ku("llama.expert_count", 4), ku("llama.expert_used_count", 2)],
     probs_b=[1000.0, 500.0, 0.0, -1000.0])

# --- gating-function SENSITIVITY fixtures.
# The dense-oracle fixtures above cannot discriminate a gating function: with a
# zero router every expert scores the same, and renormalization then produces
# 0.5/0.5 whatever function ran. So these use the ROUTED fixture instead — a
# real router and experts that disagree — where each gating function must yield
# a visibly different answer. They are not dense-equivalent and are compared
# against each other, not against dense: the property is "this knob changes the
# arithmetic", which is exactly what the oracle fixtures cannot show.
for _tag, _fn in (("rsoft", 1), ("rsigmoid", 2), ("rsmw", 3), ("rsqrtsp", 4)):
    _ts = list(shared)
    for _i in range(LAYERS):
        _ts += moe_ffn_routed(_i, 4)
    write(f"{OUT}.{_tag}.gguf", _ts,
          base_meta("llama", [ku("llama.expert_count", 4),
                              ku("llama.expert_used_count", 2),
                              ku("llama.expert_gating_func", _fn)]))

# --- shared always-on expert. The routed experts are all zeros and the SHARED
# branch carries the dense FFN, so the routed path contributes nothing and the
# output must equal the dense oracle exactly. If the shared branch were
# ignored the output collapses to zero-FFN; if it were added twice it doubles.
#   shexp    ungated (DeepSeek shape)
#   shexpg   gated by a scalar sigmoid router (Qwen2-MoE shape). The router
#            weight is zero, so the logit is 0 and sigmoid(0) = 0.5 — the
#            shared FFN is therefore doubled to compensate, which only works
#            if the gate is really applied.
for _tag, _gated in (("shexp", False), ("shexpg", True)):
    _ts = list(shared)
    for _i in range(LAYERS):
        _scale = 2.0 if _gated else 1.0
        _ts += [
            (f"blk.{_i}.ffn_gate_inp.weight", [E, 2], zeros(E * 2)),
            (f"blk.{_i}.ffn_gate_exps.weight", [E, FF, 2], pack(ffn[_i]["gate"] * 2)),
            (f"blk.{_i}.ffn_up_exps.weight", [E, FF, 2], pack(ffn[_i]["up"] * 2)),
            (f"blk.{_i}.ffn_down_exps.weight", [FF, E, 2], zeros(FF * E * 2)),
            (f"blk.{_i}.ffn_gate_shexp.weight", [E, FF], pack(ffn[_i]["gate"])),
            (f"blk.{_i}.ffn_up_shexp.weight", [E, FF], pack(ffn[_i]["up"])),
            (f"blk.{_i}.ffn_down_shexp.weight", [FF, E],
             pack([x * _scale for x in ffn[_i]["down"]])),
        ]
        if _gated:
            _ts.append((f"blk.{_i}.ffn_gate_inp_shexp.weight", [E, 1], zeros(E)))
    write(f"{OUT}.{_tag}.gguf", _ts,
          base_meta("llama", [ku("llama.expert_count", 2),
                              ku("llama.expert_used_count", 1),
                              ku("llama.expert_shared_count", 1),
                              ku("llama.expert_shared_feed_forward_length", FF)]))


# --- gpt-oss advanced MoE fixture. This one is not dense-equivalent: the CPU
# backend is the oracle, and the Metal smoke compares the same file on CPU vs
# GPU. It covers the gpt-oss-specific contract: post_attention_norm as the FFN
# input norm, attention sinks, OAI SwiGLU, router bias, gate/up/down expert
# biases, SWA-pattern metadata, and MXFP4 expert tensors.
gptoss = [
    ("token_embd.weight", [E, len(VOCAB)], pack(flist(E * len(VOCAB)))),
    ("output_norm.weight", [E], ones(E)),
]
for _i in range(LAYERS):
    gptoss += [
        (f"blk.{_i}.attn_norm.weight", [E], ones(E)),
        (f"blk.{_i}.attn_q.weight", [E, E], pack(flist(E * E))),
        (f"blk.{_i}.attn_k.weight", [E, kv_dim], pack(flist(E * kv_dim))),
        (f"blk.{_i}.attn_v.weight", [E, kv_dim], pack(flist(E * kv_dim))),
        (f"blk.{_i}.attn_output.weight", [E, E], pack(flist(E * E))),
        (f"blk.{_i}.post_attention_norm.weight", [E], ones(E)),
        (f"blk.{_i}.attn_sinks.weight", [HEADS],
         pack([0.25, -0.15, 0.10, -0.05])),
        (f"blk.{_i}.ffn_gate_inp.weight", [E, 2], pack(flist(E * 2))),
        (f"blk.{_i}.ffn_gate_inp.bias", [2], pack([0.45, -0.20])),
        (f"blk.{_i}.ffn_gate_exps.weight", [E, FF, 2],
         mxfp4_payload(E, FF * 2, 11 + _i), T_MXFP4),
        (f"blk.{_i}.ffn_up_exps.weight", [E, FF, 2],
         mxfp4_payload(E, FF * 2, 37 + _i), T_MXFP4),
        (f"blk.{_i}.ffn_down_exps.weight", [FF, E, 2],
         mxfp4_payload(FF, E * 2, 71 + _i), T_MXFP4),
        (f"blk.{_i}.ffn_gate_exps.bias", [FF, 2],
         pack([0.03 * (((j + _i) % 5) - 2) for j in range(FF * 2)])),
        (f"blk.{_i}.ffn_up_exps.bias", [FF, 2],
         pack([0.02 * (((j + 2 * _i) % 7) - 3) for j in range(FF * 2)])),
        (f"blk.{_i}.ffn_down_exps.bias", [E, 2],
         pack([0.015 * (((j + 3 * _i) % 7) - 3) for j in range(E * 2)])),
    ]
write(f"{OUT}.gptoss-mxfp4.gguf", gptoss,
      base_meta("gpt-oss", [ku("gpt-oss.expert_count", 2),
                            ku("gpt-oss.expert_used_count", 2),
                            ku("gpt-oss.expert_feed_forward_length", FF),
                            ku("gpt-oss.attention.sliding_window", 8),
                            ku("gpt-oss.attention.sliding_window_pattern", 2)]))

# The same tensors routed TOP-1. At expert_used_count = 2 of 2 every expert
# runs whatever the router says, so ffn_gate_inp.bias can only reweight the
# two outputs -- a backend that drops it lands inside every tolerance in the
# tree. At top-1 the bias decides WHICH expert runs, which is what it does on
# the shipping gpt-oss models (top-4 of 32/128), and dropping it moves the
# logits by percent rather than by ulps. Measured on the CUDA fused MoE path:
# 0.0044 of logit range with the bias dropped, 3.8e-08 with it applied.
write(f"{OUT}.gptoss-top1.gguf", gptoss,
      base_meta("gpt-oss", [ku("gpt-oss.expert_count", 2),
                            ku("gpt-oss.expert_used_count", 1),
                            ku("gpt-oss.expert_feed_forward_length", FF),
                            ku("gpt-oss.attention.sliding_window", 8),
                            ku("gpt-oss.attention.sliding_window_pattern", 2)]))


# --- gemma-4 dual-branch MoE fixture. CPU is the oracle here too. The fixture
# exercises the gemma-only FFN shape: dense GELU branch plus routed GELU experts
# with fused gate_up_exps, router-input scale, down-expert scale, pre/post
# branch norms, post-attention/post-FFN norms, embedding scale and logit softcap.
gemma4 = [
    ("token_embd.weight", [E, len(VOCAB)], pack(flist(E * len(VOCAB)))),
    ("output_norm.weight", [E], ones(E)),
]
for _i in range(LAYERS):
    gemma4 += [
        (f"blk.{_i}.attn_norm.weight", [E], ones(E)),
        (f"blk.{_i}.attn_q.weight", [E, E], pack(flist(E * E))),
        (f"blk.{_i}.attn_k.weight", [E, kv_dim], pack(flist(E * kv_dim))),
        (f"blk.{_i}.attn_v.weight", [E, kv_dim], pack(flist(E * kv_dim))),
        (f"blk.{_i}.attn_output.weight", [E, E], pack(flist(E * E))),
        (f"blk.{_i}.attn_q_norm.weight", [E // HEADS], ones(E // HEADS)),
        (f"blk.{_i}.attn_k_norm.weight", [E // HEADS], ones(E // HEADS)),
        (f"blk.{_i}.post_attention_norm.weight", [E], ones(E)),
        (f"blk.{_i}.ffn_norm.weight", [E], ones(E)),
        (f"blk.{_i}.post_ffw_norm.weight", [E], ones(E)),
        (f"blk.{_i}.ffn_gate.weight", [E, FF], pack([v * 2 for v in flist(E * FF)])),
        (f"blk.{_i}.ffn_up.weight", [E, FF], pack([v * 2 for v in flist(E * FF)])),
        (f"blk.{_i}.ffn_down.weight", [FF, E], pack([v * 2 for v in flist(FF * E)])),
        (f"blk.{_i}.ffn_gate_inp.weight", [E, 2], pack(flist(E * 2))),
        (f"blk.{_i}.ffn_gate_inp.scale", [E],
         pack([0.9 + 0.01 * ((j + _i) % 5) for j in range(E)])),
        (f"blk.{_i}.ffn_gate_up_exps.weight", [E, 2 * FF, 2],
         pack([v * 2 for v in flist(E * 2 * FF * 2)])),
        (f"blk.{_i}.ffn_down_exps.weight", [FF, E, 2],
         pack([v * 2 for v in flist(FF * E * 2)])),
        (f"blk.{_i}.ffn_down_exps.scale", [2], pack([0.85, 1.15])),
        (f"blk.{_i}.post_ffw_norm_1.weight", [E], ones(E)),
        (f"blk.{_i}.pre_ffw_norm_2.weight", [E], ones(E)),
        (f"blk.{_i}.post_ffw_norm_2.weight", [E], ones(E)),
        (f"blk.{_i}.layer_output_scale.weight", [1], pack([0.92 + 0.03 * _i])),
    ]
write(f"{OUT}.gemma4-moe.gguf", gemma4,
      base_meta("gemma4", [ku("gemma4.expert_count", 2),
                           ku("gemma4.expert_used_count", 2),
                           ku("gemma4.expert_feed_forward_length", FF),
                           ku("gemma4.attention.key_length", E // HEADS),
                           ku("gemma4.rope.dimension_count", E // HEADS),
                           kf("gemma4.final_logit_softcapping", 30.0)]))

# gemma4 dual-branch MoE with the real 26B's HETEROGENEOUS attention: sliding
# layers (i%3 != 2) on smaller heads with fewer KV heads, full layers V-LESS
# (no attn_v — V is the raw K projection under the weightless V head norm).
# Same MoE FFN block as the uniform fixture; only attention geometry differs.
_HL = 3                                        # [S, S, F]
def _h_swa(i): return (i % 3) != 2
def _h_hd(i):  return 8 if _h_swa(i) else 16
def _h_kv(i):  return 1 if _h_swa(i) else 2
g4h = [
    ("token_embd.weight", [E, len(VOCAB)], pack(flist(E * len(VOCAB)))),
    ("output_norm.weight", [E], ones(E)),
]
for _i in range(_HL):
    _qd, _kd = HEADS * _h_hd(_i), _h_kv(_i) * _h_hd(_i)
    g4h += [
        (f"blk.{_i}.attn_norm.weight", [E], ones(E)),
        (f"blk.{_i}.attn_q.weight", [E, _qd], pack(flist(E * _qd))),
        (f"blk.{_i}.attn_k.weight", [E, _kd], pack(flist(E * _kd))),
        *([(f"blk.{_i}.attn_v.weight", [E, _kd], pack(flist(E * _kd)))]
          if _h_swa(_i) else []),
        (f"blk.{_i}.attn_output.weight", [_qd, E], pack(flist(_qd * E))),
        (f"blk.{_i}.attn_q_norm.weight", [_h_hd(_i)], ones(_h_hd(_i))),
        (f"blk.{_i}.attn_k_norm.weight", [_h_hd(_i)], ones(_h_hd(_i))),
        (f"blk.{_i}.post_attention_norm.weight", [E], ones(E)),
        (f"blk.{_i}.ffn_norm.weight", [E], ones(E)),
        (f"blk.{_i}.post_ffw_norm.weight", [E], ones(E)),
        (f"blk.{_i}.ffn_gate.weight", [E, FF], pack([v * 2 for v in flist(E * FF)])),
        (f"blk.{_i}.ffn_up.weight", [E, FF], pack([v * 2 for v in flist(E * FF)])),
        (f"blk.{_i}.ffn_down.weight", [FF, E], pack([v * 2 for v in flist(FF * E)])),
        (f"blk.{_i}.ffn_gate_inp.weight", [E, 2], pack(flist(E * 2))),
        (f"blk.{_i}.ffn_gate_inp.scale", [E],
         pack([0.9 + 0.01 * ((j + _i) % 5) for j in range(E)])),
        (f"blk.{_i}.ffn_gate_up_exps.weight", [E, 2 * FF, 2],
         pack([v * 2 for v in flist(E * 2 * FF * 2)])),
        (f"blk.{_i}.ffn_down_exps.weight", [FF, E, 2],
         pack([v * 2 for v in flist(FF * E * 2)])),
        (f"blk.{_i}.ffn_down_exps.scale", [2], pack([0.85, 1.15])),
        (f"blk.{_i}.post_ffw_norm_1.weight", [E], ones(E)),
        (f"blk.{_i}.pre_ffw_norm_2.weight", [E], ones(E)),
        (f"blk.{_i}.post_ffw_norm_2.weight", [E], ones(E)),
        (f"blk.{_i}.layer_output_scale.weight", [1], pack([0.92 + 0.03 * _i])),
    ]
_g4h_meta = [
    ks("general.architecture", "gemma4"),
    ku("gemma4.block_count", _HL), ku("gemma4.context_length", 256),
    ku("gemma4.embedding_length", E), ku("gemma4.feed_forward_length", FF),
    ku("gemma4.attention.head_count", HEADS),
    kau("gemma4.attention.head_count_kv", [_h_kv(i) for i in range(_HL)]),
    kf("gemma4.attention.layer_norm_rms_epsilon", 1e-5),
    kf("gemma4.rope.freq_base", 10000.0),
    ku("gemma4.expert_count", 2), ku("gemma4.expert_used_count", 2),
    ku("gemma4.expert_feed_forward_length", FF),
    ku("gemma4.attention.key_length", 16),
    ku("gemma4.attention.key_length_swa", 8),
    ku("gemma4.attention.sliding_window", 16),
    kab("gemma4.attention.sliding_window_pattern", [_h_swa(i) for i in range(_HL)]),
    ku("gemma4.rope.dimension_count", 16),
    ku("gemma4.rope.dimension_count_swa", 8),
    kf("gemma4.final_logit_softcapping", 30.0),
    ks("tokenizer.ggml.model", "llama"), kas("tokenizer.ggml.tokens", VOCAB),
    kaf("tokenizer.ggml.scores", [0.0] * len(VOCAB)),
    kai("tokenizer.ggml.token_type", TTYPE), ku("tokenizer.ggml.bos_token_id", 1),
    ku("tokenizer.ggml.eos_token_id", 2), kb("tokenizer.ggml.add_bos_token", True),
]
write(f"{OUT}.gemma4-moe-hetero.gguf", g4h, _g4h_meta)
