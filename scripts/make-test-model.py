#!/usr/bin/env python3
"""Generate a tiny random llama-architecture GGUF for CI smoke tests.

The model is ~1 MB of random weights with a byte-fallback SPM vocabulary:
output is gibberish, but loading, tokenization, the forward pass, sampling,
and JSON-constrained decoding all execute the same code paths as a real model.
"""
import struct
import sys

OUT = "test.gguf"
SUPPRESS_ALL_BUT_EOS = False
ZERO_FIRST_DIM = False
WRAP_FIRST_OFFSET = False
ARCH = "llama"
AGENT_PROFILE = False
AGENT_FEATURES = ["dense", "json_schema"]
MTP_LAYERS = 0   # extra trailing blocks declared as NextN/MTP predictor heads
# Gemma-4 E-series: per-layer embeddings plus a tail of layers that own no KV
# cache. Both mechanisms are structural, so a tiny random model exercises the
# load-time geometry, the aliased cache reads and the extra forward stage
# without needing the 5 GB real file.
APERTUS = None
ATTN_NOPE_STEP = 0
ATTN_TEMP_SCALE = 0.0
# Llama-4's own floor_scale. At 8192 the temperature is exactly 1.0 for every
# position below 8191, which is correct and makes the knob untestable on a
# short prompt -- so it is overridable, and a small value gives a fixture where
# the temperature is actually live at position 0.
ATTN_TEMP_FLOOR = 8192
SWA_WINDOW = 0
SWA_PATTERN = 0
ESERIES_SHARED_KV = 0
ESERIES_PLE = 0
FFN_WIDTHS = None  # per-layer FFN widths -> ARRAY-typed feed_forward_length
G4HETERO = False   # gemma4 heterogeneous attention geometry (26B/12B shape)
ACT_OVERFLOW = 0   # scale on ffn_gate weights, to drive the activation extreme
MUSE = False       # muse-glimmer: gated attention + QK/sandwich norms + NoPE
MUSE_GATE_FLAT = False  # zero the attn_gate weights (sigmoid -> flat 0.5)
MUSE_ALL_SWA = False    # pattern array all-sliding (every layer ropes)
GRANITE = False    # granite: four muP scalars, tied embeddings
GRANITE_RESID = 0.5  # residual_scale for the fixture (CLI-overridable)
QUANT = None       # --quant q8_0: store the 2-D matmul weights quantized
args = sys.argv[1:]
i = 0
while i < len(args):
    a = args[i]
    if a == "--suppress-all-but-eos":
        SUPPRESS_ALL_BUT_EOS = True
    elif a == "--zero-first-dim":
        ZERO_FIRST_DIM = True
    elif a == "--wrap-first-offset":
        WRAP_FIRST_OFFSET = True
    elif a == "--arch":
        i += 1
        ARCH = args[i]
    elif a == "--agent-profile":
        AGENT_PROFILE = True
    elif a == "--agent-feature":
        i += 1
        AGENT_PROFILE = True
        AGENT_FEATURES.append(args[i])
    elif a == "--apertus":
        # ungated MLP + xIELU. "IDENT" makes xIELU the identity
        # (alpha_p=0, alpha_n=0, beta=1) so the FFN is a plain up->down
        # linear map, which the runner's own dense path can be checked against.
        i += 1
        APERTUS = args[i]
        ARCH = "apertus"
    elif a == "--attn-knobs":
        # "STEP,TEMPSCALE[,FLOOR]" — NoPE every STEP-th layer, and the
        # Llama-4 attention temperature on those layers (0 disables the
        # temperature). FLOOR defaults to llama-4's 8192, which makes the
        # temperature a no-op below position 8191; pass a small FLOOR for a
        # fixture where it is live immediately.
        i += 1
        _parts = args[i].split(",")
        _st, _ts = _parts[0], _parts[1]
        ATTN_NOPE_STEP, ATTN_TEMP_SCALE = int(_st), float(_ts)
        if len(_parts) > 2:
            ATTN_TEMP_FLOOR = int(_parts[2])
    elif a == "--swa":
        # "WINDOW,PATTERN" — every PATTERN-th layer is full attention, the
        # others slide. This is intentionally arch-neutral so small Qwen-style
        # fixtures can exercise SWA without Gemma's unrelated GELU/norms.
        i += 1
        _win, _pat = args[i].split(",")
        SWA_WINDOW, SWA_PATTERN = int(_win), int(_pat)
    elif a == "--eseries":
        # --eseries SHARED_KV,PLE_DIM  (e.g. "3,16" on the default 6 layers)
        i += 1
        shared, ple = args[i].split(",")
        ESERIES_SHARED_KV, ESERIES_PLE = int(shared), int(ple)
        ARCH = "gemma4"
    elif a == "--act-overflow":
        # Scale ffn_gate so the gated activation's input is large. The GELU
        # tanh argument grows as x^3, and a fast-math tanh evaluated through
        # exp(2a) overflows to inf (then inf/inf = NaN) well before fp32 runs
        # out of range — a real gemma-3-4b hit exactly this on Metal and
        # emitted only token 0. Real models reach these magnitudes; the
        # default fixture weights (±0.04) never do.
        ACT_OVERFLOW = 400.0
    elif a == "--muse-glimmer":
        # muse-glimmer scaled down but shape-preserving: head_dim decoupled
        # from n_embd/n_head (real: 32x128=4096 vs n_embd 6656), afmoe-style
        # attn_gate, QK norms, sandwich norms, the real 3-local:1-global bool
        # pattern array (full layers NoPE), scaled+softcapped logits, and an
        # untied output tensor.
        MUSE = True
        ARCH = "muse-glimmer"
    elif a == "--muse-gate-flat":
        MUSE_GATE_FLAT = True
    elif a == "--muse-all-swa":
        MUSE_ALL_SWA = True
    elif a == "--granite":
        # IBM Granite dense: embedding_scale, a FIXED attention scale,
        # residual_scale on both branch outputs, divided logit scale, tied
        # embeddings (no output tensor). Scaled-down but shape-preserving.
        GRANITE = True
        ARCH = "granite"
    elif a == "--granite-resid":
        i += 1
        GRANITE_RESID = float(args[i])
    elif a == "--quant":
        # Store the 2-D matmul weights in a block-quantized type instead of
        # F32. An F32 fixture cannot reach a backend's quantized matvec at
        # all, so gates run on one are silent about every kernel that matters
        # for a real model -- see the CUDA decode-microbatch identity break,
        # which survived three weeks of a green `make test` because the only
        # model the batch gate ran on was F32.
        i += 1
        QUANT = args[i].lower()
        if QUANT not in ("q8_0",):
            sys.exit(f"--quant: unsupported type {QUANT!r} (have: q8_0)")
    elif a == "--gemma4-hetero":
        # the real gemma-4 26B/12B attention shape, scaled down: sliding
        # layers (i%3 != 2) rotate fewer dims on smaller heads with fewer KV
        # heads; full layers are V-LESS (no attn_v tensor — V is the raw K
        # projection under the weightless per-head V RMS norm). Exercises
        # per-layer head_dim/head_count_kv/rope_dim + V-less + sandwich
        # norms + per-layer output scale + logit softcap.
        G4HETERO = True
        ARCH = "gemma4"
    elif a == "--ffn-widths":
        # "W0,W1,..." one width per layer — emitted as an ARRAY-typed
        # feed_forward_length, the gemma-4 E2B export shape (real per-layer
        # variation 6144/12288); tensor shapes follow the per-layer widths
        i += 1
        FFN_WIDTHS = [int(w) for w in args[i].split(",")]
    elif a == "--mtp-layers":
        # emit N extra blocks and declare them as training-only MTP predictor
        # heads; the runner must exclude them and decode exactly as without
        i += 1
        MTP_LAYERS = int(args[i])
    else:
        OUT = a
    i += 1

N_EMBD, N_HEAD, N_KV, N_FF, N_LAYER = 64, 4, 2, 128, 2
if ESERIES_SHARED_KV or ESERIES_PLE or G4HETERO:
    # enough layers for a sliding/full pattern (with a shared-KV tail for
    # the E-series variants)
    N_LAYER = 6
MUSE_HD = 24         # decoupled head_dim: N_HEAD*24 = 96, not N_EMBD (64)
if MUSE:
    N_LAYER = 8      # two full periods of the 3-sliding:1-full pattern
def muse_swa(i):  return True if MUSE_ALL_SWA else (i % 4) != 3

# gemma4-hetero per-layer geometry: full-attention layers (i%3 == 2) use
# head_dim 32 / 2 KV heads / no V tensor; sliding layers head_dim 16 /
# 1 KV head / V present. Q width therefore varies 128 vs 64.
def g4_swa(i):  return (i % 3) != 2
def g4_hd(i):   return 16 if g4_swa(i) else 32
def g4_kv(i):   return 1 if g4_swa(i) else 2
VOCAB = ["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)]
TTYPE = [2, 3, 3] + [6] * 256  # unknown, control, control, bytes
N_VOCAB = len(VOCAB)

GGUF_U32, GGUF_F32, GGUF_STR, GGUF_ARR, GGUF_I32, GGUF_BOOL = 4, 6, 8, 9, 5, 7


def s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def kv_u32(k, v):  return s(k) + struct.pack("<II", GGUF_U32, v)
def kv_f32(k, v):  return s(k) + struct.pack("<If", GGUF_F32, v)
def kv_str(k, v):  return s(k) + struct.pack("<I", GGUF_STR) + s(v)
def kv_bool(k, v): return s(k) + struct.pack("<IB", GGUF_BOOL, 1 if v else 0)


def kv_arr_str(k, items):
    out = s(k) + struct.pack("<IIQ", GGUF_ARR, GGUF_STR, len(items))
    for it in items:
        out += s(it)
    return out


def kv_arr_f32(k, items):
    return (s(k) + struct.pack("<IIQ", GGUF_ARR, GGUF_F32, len(items)) +
            struct.pack(f"<{len(items)}f", *items))


def kv_arr_bool(k, items):
    return (s(k) + struct.pack("<IIQ", GGUF_ARR, GGUF_BOOL, len(items)) +
            bytes(1 if x else 0 for x in items))


def kv_arr_i32(k, items):
    return (s(k) + struct.pack("<IIQ", GGUF_ARR, GGUF_I32, len(items)) +
            struct.pack(f"<{len(items)}i", *items))


def kv_arr_u32(k, items):
    return (s(k) + struct.pack("<IIQ", GGUF_ARR, GGUF_U32, len(items)) +
            struct.pack(f"<{len(items)}I", *items))


# deterministic pseudo-random floats (no numpy dependency)
_state = 0x12345678


def rnd():
    global _state
    _state = (_state * 1103515245 + 12345) & 0x7FFFFFFF
    return (_state / 0x7FFFFFFF - 0.5) * 0.08


def tensor_data(n, scale=1.0):
    return struct.pack(f"<{n}f", *(rnd() * scale for _ in range(n)))


def ones(n):
    return struct.pack(f"<{n}f", *([1.0] * n))


tensors = [("token_embd.weight", [N_EMBD, N_VOCAB], tensor_data(N_EMBD * N_VOCAB)),
           ("output_norm.weight", [N_EMBD], ones(N_EMBD))]
# MTP predictor heads are counted inside block_count by the exports that carry
# them, exactly like the Qwen3.5 NextN blocks; the runner must strip them back
# off and decode as if they were never there.
if FFN_WIDTHS is not None and len(FFN_WIDTHS) != N_LAYER + MTP_LAYERS:
    sys.exit(f"--ffn-widths needs {N_LAYER + MTP_LAYERS} entries, "
             f"got {len(FFN_WIDTHS)}")

for i in range(N_LAYER + MTP_LAYERS):
    if G4HETERO:
        q_dim  = N_HEAD * g4_hd(i)
        kv_dim = g4_kv(i) * g4_hd(i)
    elif MUSE:
        q_dim  = N_HEAD * MUSE_HD
        kv_dim = N_KV * MUSE_HD
    else:
        q_dim  = N_EMBD
        kv_dim = N_KV * (N_EMBD // N_HEAD)
    N_FF_I = FFN_WIDTHS[i] if FFN_WIDTHS else N_FF
    tensors += [
        (f"blk.{i}.attn_norm.weight", [N_EMBD], ones(N_EMBD)),
        (f"blk.{i}.attn_q.weight", [N_EMBD, q_dim], tensor_data(N_EMBD * q_dim)),
        (f"blk.{i}.attn_k.weight", [N_EMBD, kv_dim], tensor_data(N_EMBD * kv_dim)),
        *([] if (G4HETERO and not g4_swa(i)) else
          [(f"blk.{i}.attn_v.weight", [N_EMBD, kv_dim], tensor_data(N_EMBD * kv_dim))]),
        (f"blk.{i}.attn_output.weight", [q_dim, N_EMBD], tensor_data(q_dim * N_EMBD)),
        (f"blk.{i}.ffn_norm.weight", [N_EMBD], ones(N_EMBD)),
        *([] if APERTUS else
          [(f"blk.{i}.ffn_gate.weight", [N_EMBD, N_FF_I],
            tensor_data(N_EMBD * N_FF_I, ACT_OVERFLOW or 1.0))]),
        (f"blk.{i}.ffn_up.weight", [N_EMBD, N_FF_I], tensor_data(N_EMBD * N_FF_I)),
        (f"blk.{i}.ffn_down.weight", [N_FF_I, N_EMBD], tensor_data(N_FF_I * N_EMBD)),
    ]
    if ESERIES_SHARED_KV or ESERIES_PLE or G4HETERO:
        head_dim = g4_hd(i) if G4HETERO else N_EMBD // N_HEAD
        tensors += [
            (f"blk.{i}.attn_q_norm.weight", [head_dim], ones(head_dim)),
            (f"blk.{i}.attn_k_norm.weight", [head_dim], ones(head_dim)),
            (f"blk.{i}.post_attention_norm.weight", [N_EMBD], ones(N_EMBD)),
            (f"blk.{i}.post_ffw_norm.weight", [N_EMBD], ones(N_EMBD)),
            (f"blk.{i}.layer_output_scale.weight", [1],
             struct.pack("<f", 0.91 + 0.02 * i) if G4HETERO else ones(1)),
        ]
    if MUSE:
        tensors += [
            (f"blk.{i}.attn_q_norm.weight", [MUSE_HD], ones(MUSE_HD)),
            (f"blk.{i}.attn_k_norm.weight", [MUSE_HD], ones(MUSE_HD)),
            (f"blk.{i}.attn_gate.weight", [N_EMBD, q_dim],
             struct.pack(f"<{N_EMBD * q_dim}f", *([0.0] * (N_EMBD * q_dim)))
             if MUSE_GATE_FLAT else tensor_data(N_EMBD * q_dim)),
            (f"blk.{i}.post_attention_norm.weight", [N_EMBD], ones(N_EMBD)),
            (f"blk.{i}.post_ffw_norm.weight", [N_EMBD], ones(N_EMBD)),
        ]
    if ESERIES_PLE:
        tensors += [
            (f"blk.{i}.inp_gate.weight", [N_EMBD, ESERIES_PLE],
             tensor_data(N_EMBD * ESERIES_PLE)),
            (f"blk.{i}.proj.weight", [ESERIES_PLE, N_EMBD],
             tensor_data(ESERIES_PLE * N_EMBD)),
            (f"blk.{i}.post_norm.weight", [N_EMBD], ones(N_EMBD)),
        ]

if MUSE or GRANITE:
    # muse: the real model's lm head is untied from the embeddings.
    # granite: the real models ARE tied, but a tied byte-vocab fixture locks
    # greedy argmax onto the input's last token (embedding self-similarity
    # survives any residual perturbation — even residual_scale 0), which
    # made the scalar-differential test vacuous. A random untied head makes
    # logits direction-sensitive; the tied layout is covered at real-model
    # certification.
    tensors.append(("output.weight", [N_EMBD, N_VOCAB],
                    tensor_data(N_EMBD * N_VOCAB)))

if ESERIES_PLE:
    width = ESERIES_PLE * N_LAYER
    tensors += [
        ("per_layer_token_embd.weight", [width, N_VOCAB],
         tensor_data(width * N_VOCAB)),
        ("per_layer_model_proj.weight", [N_EMBD, width],
         tensor_data(N_EMBD * width)),
        ("per_layer_proj_norm.weight", [ESERIES_PLE], ones(ESERIES_PLE)),
    ]

meta_kvs = [
    kv_str("general.architecture", ARCH),
    kv_u32(f"{ARCH}.block_count", N_LAYER + MTP_LAYERS),
    kv_u32(f"{ARCH}.context_length", 256),
    kv_u32(f"{ARCH}.embedding_length", N_EMBD),
    (kv_arr_u32(f"{ARCH}.feed_forward_length", FFN_WIDTHS) if FFN_WIDTHS
     else kv_u32(f"{ARCH}.feed_forward_length", N_FF)),
    kv_u32(f"{ARCH}.attention.head_count", N_HEAD),
    # G4HETERO publishes head_count_kv as a per-layer ARRAY below instead
    *([] if G4HETERO else [kv_u32(f"{ARCH}.attention.head_count_kv", N_KV)]),
    kv_f32(f"{ARCH}.attention.layer_norm_rms_epsilon", 1e-5),
    kv_f32(f"{ARCH}.rope.freq_base", 10000.0),
    kv_str("tokenizer.ggml.model", "llama"),
    kv_arr_str("tokenizer.ggml.tokens", VOCAB),
    kv_arr_f32("tokenizer.ggml.scores", [0.0] * N_VOCAB),
    kv_arr_i32("tokenizer.ggml.token_type", TTYPE),
    kv_u32("tokenizer.ggml.bos_token_id", 1),
    kv_u32("tokenizer.ggml.eos_token_id", 2),
    kv_bool("tokenizer.ggml.add_bos_token", True),
]
if ESERIES_SHARED_KV or ESERIES_PLE:
    # every third layer is a full-attention layer, the rest slide
    pattern = [(i % 3) != 2 for i in range(N_LAYER)]
    meta_kvs += [
        kv_u32(f"{ARCH}.attention.key_length", N_EMBD // N_HEAD),
        kv_u32(f"{ARCH}.attention.value_length", N_EMBD // N_HEAD),
        kv_u32(f"{ARCH}.attention.key_length_swa", N_EMBD // N_HEAD),
        kv_u32(f"{ARCH}.attention.sliding_window", 32),
        kv_arr_bool(f"{ARCH}.attention.sliding_window_pattern", pattern),
        kv_u32(f"{ARCH}.rope.dimension_count", N_EMBD // N_HEAD),
        kv_f32(f"{ARCH}.final_logit_softcapping", 30.0),
    ]
if ARCH == "gemma4" and not (ESERIES_SHARED_KV or ESERIES_PLE or G4HETERO):
    # gemma4's loader defaults attention.key_length to 512 when the key is
    # absent, because every official gemma-4 export publishes it. This
    # generator did not, so a plain `--arch gemma4` fixture declared a
    # geometry (4 heads x 512 = 2048) that its own attn_output tensor
    # ([64,64]) contradicted, and the loader refused it:
    #
    #   error: tensor attn_output in blk.0 has shape [64,64],
    #          expected [2048,>=64] for this model geometry
    #
    # So dense gemma4 had no working fixture at all — and because BOTH the
    # CPU and GPU arms failed identically, a cpu-vs-metal `cmp` of their
    # (empty) output passed. The --eseries and --gemma4-hetero branches below
    # publish these keys already, which is why only the plain arch was dark.
    meta_kvs += [
        kv_u32(f"{ARCH}.attention.key_length", N_EMBD // N_HEAD),
        kv_u32(f"{ARCH}.attention.value_length", N_EMBD // N_HEAD),
        kv_u32(f"{ARCH}.attention.key_length_swa", N_EMBD // N_HEAD),
        kv_u32(f"{ARCH}.rope.dimension_count", N_EMBD // N_HEAD),
    ]
if G4HETERO:
    pattern = [g4_swa(i) for i in range(N_LAYER)]
    meta_kvs += [
        kv_u32(f"{ARCH}.attention.key_length", 32),
        kv_u32(f"{ARCH}.attention.key_length_swa", 16),
        kv_arr_u32(f"{ARCH}.attention.head_count_kv",
                   [g4_kv(i) for i in range(N_LAYER)]),
        kv_u32(f"{ARCH}.attention.sliding_window", 32),
        kv_arr_bool(f"{ARCH}.attention.sliding_window_pattern", pattern),
        kv_u32(f"{ARCH}.rope.dimension_count", 32),
        kv_u32(f"{ARCH}.rope.dimension_count_swa", 16),
        kv_f32(f"{ARCH}.final_logit_softcapping", 30.0),
    ]
if GRANITE:
    # same keys and code paths as the real exports, but fixture-sane
    # magnitudes: the real muP values (embedding x12, attention x0.0078)
    # collapse a 64-dim random model into single-token degeneracy, which
    # made the residual-scale differential test vacuously pass-or-fail
    meta_kvs += [
        kv_f32(f"{ARCH}.embedding_scale", 2.0),
        kv_f32(f"{ARCH}.attention.scale", 0.25),
        kv_f32(f"{ARCH}.residual_scale", GRANITE_RESID),
        kv_f32(f"{ARCH}.logit_scale", 16.0),
    ]
if MUSE:
    meta_kvs += [
        kv_u32(f"{ARCH}.attention.key_length", MUSE_HD),
        kv_u32(f"{ARCH}.attention.value_length", MUSE_HD),
        kv_u32(f"{ARCH}.rope.dimension_count", MUSE_HD),
        kv_u32(f"{ARCH}.attention.sliding_window", 32),
        kv_arr_bool(f"{ARCH}.attention.sliding_window_pattern",
                    [muse_swa(i) for i in range(N_LAYER)]),
        kv_f32(f"{ARCH}.logit_scale", 0.5),
        kv_f32(f"{ARCH}.final_logit_softcapping", 30.0),
    ]
if ESERIES_SHARED_KV:
    meta_kvs.append(
        kv_u32(f"{ARCH}.attention.shared_kv_layers", ESERIES_SHARED_KV))
if ESERIES_PLE:
    meta_kvs.append(
        kv_u32(f"{ARCH}.embedding_length_per_layer_input", ESERIES_PLE))
if APERTUS:
    _id = APERTUS == "IDENT"
    meta_kvs += [
        kv_f32("xielu.alpha_n", 0.0 if _id else 0.8),
        kv_f32("xielu.alpha_p", 0.0 if _id else 0.8),
        kv_f32("xielu.beta",    1.0 if _id else 0.5),
        kv_f32("xielu.eps",     -1e-6),
    ]
if ATTN_NOPE_STEP:
    meta_kvs += [
        kv_u32(f"{ARCH}.attention.no_rope_layer_step", ATTN_NOPE_STEP),
        kv_u32(f"{ARCH}.attention.attn_temp_floor_scale", ATTN_TEMP_FLOOR),
        kv_f32(f"{ARCH}.attention.attn_temp_scale", ATTN_TEMP_SCALE),
        kv_f32(f"{ARCH}.attention.attn_temp_offset", 1.0),
    ]
if SWA_WINDOW:
    meta_kvs += [
        kv_u32(f"{ARCH}.attention.sliding_window", SWA_WINDOW),
        kv_u32(f"{ARCH}.attention.sliding_window_pattern", SWA_PATTERN or 2),
    ]
if MTP_LAYERS:
    meta_kvs.append(kv_u32(f"{ARCH}.nextn_predict_layers", MTP_LAYERS))
if AGENT_PROFILE:
    meta_kvs += [
        kv_u32("gridcore.agent.protocol_version", 1),
        kv_u32("gridcore.agent.tokenizer_version", 1),
        kv_str("gridcore.agent.schema_id", "gridcore.agent.action.v1"),
        kv_str("gridcore.agent.schema_digest", "a" * 64),
        kv_arr_str("gridcore.agent.required_features", AGENT_FEATURES),
    ]
if SUPPRESS_ALL_BUT_EOS:
    # every token except </s> is suppressed: greedy generation must emit EOS
    # immediately, so a completion prints only the echoed prompt
    meta_kvs.append(kv_arr_i32("tokenizer.ggml.suppress_tokens",
                               [i for i in range(N_VOCAB) if i != 2]))
meta = b"".join(meta_kvs)

# ---------------------------------------------------------------- quantization
GGML_F32, GGML_Q8_0 = 0, 8


def q8_0_row(vals):
    """One row of floats -> ggml q8_0 blocks (f16 scale + 32 int8 per 32)."""
    out = bytearray()
    for b in range(0, len(vals), 32):
        blk = vals[b:b + 32]
        amax = max(abs(v) for v in blk)
        d = amax / 127.0
        inv = (1.0 / d) if d else 0.0
        out += struct.pack("<e", d)
        for v in blk:
            q = int(round(v * inv))
            out.append(struct.pack("<b", -128 if q < -128 else 127 if q > 127 else q)[0])
    return bytes(out)


def quantize(name, ne, data):
    """(type, data) for one tensor under QUANT.

    Only the 2-D matmul weights are converted, and only when the row length is
    a whole number of blocks. token_embd stays F32 deliberately: this fixture
    exists to reach the quantized MATVEC kernels, and the fixture arch ties the
    lm head to the embedding, so quantizing it would drag the embedding-lookup
    dequant into a gate that is not about it. Norms are 1-D and never eligible.
    """
    if QUANT is None or len(ne) != 2 or ne[0] % 32 or name == "token_embd.weight":
        return GGML_F32, data
    vals = struct.unpack(f"<{len(data) // 4}f", data)
    rows = [vals[r * ne[0]:(r + 1) * ne[0]] for r in range(ne[1])]
    return GGML_Q8_0, b"".join(q8_0_row(r) for r in rows)


_typed = []
for name, ne, data in tensors:
    ttype, data = quantize(name, ne, data)
    _typed.append((name, ne, data, ttype))
tensors = _typed

header = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(meta_kvs))

info = b""
offset = 0
for index, (name, ne, data, ttype) in enumerate(tensors):
    if index == 0 and ZERO_FIRST_DIM:
        ne = [0, *ne[1:]]
    info += s(name) + struct.pack("<I", len(ne))
    for d in ne:
        info += struct.pack("<Q", d)
    stored_offset = (2**64 - 16) if index == 0 and WRAP_FIRST_OFFSET else offset
    info += struct.pack("<IQ", ttype, stored_offset)
    offset += len(data)
    offset = (offset + 31) & ~31

head = header + meta + info
pad = (-len(head)) % 32

with open(OUT, "wb") as f:
    f.write(head + b"\0" * pad)
    for _, _, data, _ in tensors:
        f.write(data)
        f.write(b"\0" * ((-len(data)) % 32))

print(f"wrote {OUT}")
