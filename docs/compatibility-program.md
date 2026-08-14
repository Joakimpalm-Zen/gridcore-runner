# Compatibility program

Runner is an inference engine. Its compatibility boundary is model execution
and the APIs consumed by Thane, SDKs, gateways, frameworks and agent clients. A
separate third-party web UI is therefore not part of this matrix.

## Real-model matrix

`tests/compatibility/models.json` pins one real GGUF for every claimed Runner
architecture. A filename is not evidence: `scripts/compat_matrix.py` hashes
each file before running it and emits a versioned JSON report. Model files are
not committed.

```sh
python3 scripts/compat_matrix.py --verify-files --load \
  --reference /path/to/llama-cli \
  --out tests/compatibility/out/model-load.json
```

The matrix separates independent claims: file/load, Hugging Face tokenizer
differential, llama.cpp reference generation, CPU/GPU identity, chat/tool use
and long context. A report only marks checks that actually ran. The 2026-07-22
run verified hashes and inference loads for all eight architecture targets and
CPU/CUDA token identity for the seven GPU-capable targets available at that
time. Qwen3.5/Ornith gained native CUDA Gated DeltaNet support on 2026-07-28;
its synthetic hybrid gate and local real-model Q4/Q8 smokes are CPU/GPU
greedy-identical, while regeneration of the pinned full matrix remains a
separate evidence run.

**European roster evidence run (2026-07-29).** Five European `llama`-path
models (EuroLLM-9B, Lucie-7B, Mistral-Nemo-12B, Teuken-7B, salamandra-7b)
were SHA-pinned into the manifest and evidenced:
`eu-roster-load-2026-07-29.json` (hash + load),
`eu-roster-checks-2026-07-29.json` (cpu_cuda / tokenizer / chat / tool,
with `not_executed` recorded where a check could not run) and
`reference-<family>.json` (the 8-token greedy sweep vs pinned b10076 — the
reference binary lives at `/home/lab/agent-torture-tools/llama/llama-b10076`
on the dev box, and a CPU-only build of the pinned source is reproducible
from the workspace checkout). Findings worth naming: Lucie's tokenizer
diverges on 259/721 corpus strings — **root-caused to the GGUF conversion,
not the engine**: the file exports Lucie's BPE tokenizer as SentencePiece
with all 65,024 merge ranks flattened to −1000, so the reference
tokenization is unreproducible from the artifact by any engine (Runner and
llama.cpp b10076 are token-identical on the file; OpenLLM-France's own
official GGUF carries the same defect — reported upstream as
[OpenLLM-France/Lucie-Training#3](https://github.com/OpenLLM-France/Lucie-Training/issues/3)). Lucie still does NOT hold the tokenizer check, because
the check certifies the shipped artifact against the HF reference — but the
failure names the right culprit. Teuken renders through the llama2 fallback and its chat row shows that
framing's artifacts, though the surface and tool calls work. (Corrected
2026-08-14: this previously read "Teuken's chat template emits artifacts
(`{Answer}`)". Teuken ships no chat template — its GGUF carries no
`tokenizer.chat_template` and it detects as `llama2`, the terminal
fallback — so the artifacts were runner's own framing, not Teuken's.) `reference_compare.py`
was also fixed in this run: Runner now rejects unknown model names, so the
script asks each server for its served model id instead of sending a
placeholder.

**Eager-routing pinning for MoE identity (since the 2026-07-29 device
routing).** MoE decode/prefill default to device-side routing; its softmax
arithmetic can only bit-match hosts whose `expf` is correctly rounded (UCRT
verified; a glibc/libmvec fast-math host is ~4 ulp and unreachable by any
device code). The certified byte-identity property is therefore defined over
the **eager path** (`RUNNER_MOE_EAGER=1`) — the unchanged v0.1.4 host-routing
arithmetic — and the harnesses pin it alongside `RUNNER_CUDA_TC=0`. The
fused default (fp32 device `expf` + per-element division, the body the
Blackwell splice-proof validated) is separately verified to the weaker
class: expert selection identical and `selw` within ~2 ulp of the host
reference (two independent 1-ulp sources: device-vs-host expf and
division-vs-reciprocal codegen; observed 1 ulp at the first routing on
both cert boxes). (A correctly-rounded double-exp + reciprocal-mirror
variant briefly made fused byte-identical on correctly-rounded hosts;
retired once certification pinned eager — the property was void on the
fast-math cert box, and the mirror's rcp.rn form measured 23% of MoE
decode on the Blackwell MIG.) `RUNNER_DEBUG_MOE` dumps both paths'
routing bits for re-verification.

**Scalar-path pinning (since the 2026-07-29 TC promotion).** The tensor-core
prefill GEMM is now the default on gated dense (Q4_K, arch) combos. It is
fp16-tile arithmetic held to a tolerance gate (`tests/test_tc_tol.c`), not to
byte identity — so all exact-identity evidence in this program (cpu_cuda,
greedy_reference, the reference comparison scripts) is defined over the
scalar path. **Metal gained the same shape in 0.1.11**: prompt processing
runs a tiled simdgroup-matrix GEMM under the same `test_tc_tol` gate, pinned
off by `RUNNER_METAL_MM=0` — which the Metal identity smokes now set, exactly
as the CUDA harnesses set `RUNNER_CUDA_TC=0`. And `compat_matrix.py`, `reference_compare.py` and
`compare_llamacpp.py` pin `RUNNER_CUDA_TC=0` when spawning Runner. Existing
recorded reports predate the promotion and were produced by the scalar path
they describe; they remain valid and reproducible. The promoted default is
certified separately, per (type, arch), by the tolerance gate's recorded
rows (see the TC spec).

Tokenizer references are exercised with the pinned `tokenizers` package and
the committed 721-string corpus. Install
`tests/compatibility/tokenizer-requirements.txt`, then run `scripts/difftok.py`
with the GGUF and immutable reference revision from the published report.

That run needs the network, and for a gated repo it needs credentials — which
is what has kept this check off contributors' machines and out of CI. The
reference tokenization of a fixed corpus is a constant, so it only has to be
fetched once: add `--capture PATH --ref-revision COMMIT` to an authenticated
run to write the reference ids out, then every later run replays them with
`--ref-ids PATH` and touches no network at all. A capture records the immutable
reference revision and the SHA-256 of the exact corpus it was taken against;
replay refuses a different corpus because ids compared to the wrong strings
would report confident nonsense.

Note that a capture cannot be reconstructed from the published compat reports.
Their `stdout_tail` records the diverging cases in full — string, runner ids
and reference ids — but it is a *tail*: of the four models diverging as of
2026-08-15, only phi-3.5-mini (2/721) has every case present. mistral-7b-v0.3
(44/721), salamandra-7b (16/721) and lucie-7b (259/721) are truncated to 14, 10
and 11 cases. The tail is enough to reproduce and debug a known divergence; it
is not enough to certify the other ~700 strings still agree, which is what the
check is for.

`scripts/reference_compare.py` gives Runner and llama.cpp equivalent raw
`/v1/completions` requests and compares exact generated UTF-8 at temperature
zero. This avoids CLI banners, prompt echo, ANSI output and chat-template
differences. The initial eight-token sweep is evidence, not a universal
equivalence claim: four architecture targets matched all five prompts, while
Llama 3, Qwen 3, Phi 3 and Gemma 4 had at least one divergence. Per-prompt
outputs are committed under `tests/compatibility/out/reference-*.json`.

`scripts/compare_llamacpp.py` is the reproducible performance/evidence harness
for current MoE and release-readiness comparisons. It runs Runner and a supplied
llama.cpp `llama-server` against the same GGUF, prompt, context and greedy
sampling settings, captures model hash, commits/versions, commands, hardware,
driver, throughput, time to first token, VRAM snapshots, generated tokens, raw
responses and top-logprob data where both endpoints expose it, then writes JSON
and Markdown. CI exercises its fixture mode. Real Qwen3-30B-A3B reports were
captured on 2026-07-28 against pinned llama.cpp `b10076` on CPU and a newer
`91d2fc3` build on the same Blackwell GPU. Runner CPU/GPU identity passed, but
both independent 128-token greedy comparisons diverged after a shared prefix;
the pinned CPU reference passes the committed semantic gate (55 shared tokens,
1.523 maximum common-token logprob delta; required 32 and 2.0 respectively).
Exact 128-token identity is required between Runner CPU and GPU, while the
independent-engine gate compares logits only over the shared history. The
committed reports under `tests/compatibility/out/qwen3-30b-a3b-*` record both
the passing gate and the exact point where generated text diverges.

Apertus joined forward-pass coverage on 2026-08-03. The architecture landed
2026-08-02 (`d7eda52`) but was only checked for shape, not for numbers, and it
was **wrong**: the first run against a real checkpoint produced fluent-looking
gibberish, because `ggml_xielu` transforms alpha_n and alpha_p before
`op_xielu` ever sees them and runner had transcribed only the leaf function.
See the CHANGELOG. It carries `load`, `cpu_cuda`, and `chat` but **not**
`greedy_reference`, for the same measured reason as the two models below: its
own floor is a 0.4596-nat max log-probability delta with 4 of 16 prompts
identical, against a 0.4148-nat cross-engine delta — the disagreement with
llama.cpp is smaller than the model's disagreement with itself under a KV
precision change. Evidence:
`out/sensitivity-apertus-2026-08-03.json`,
`out/divergence-apertus-2026-08-03.json`. CUDA support followed on 2026-08-03:
the pinned Q4_K_M artifact fully offloaded 32/32 layers on an RTX 3070 and
matched CPU greedy output on 5/5 prompts at 128 tokens. Evidence:
`out/apertus-cpu-cuda-2026-08-03.json`.

The manifest also, since 2026-08-01, deliberately omits `greedy_reference` for
`gemma-4-26b-a4b-it-q4_0`. This is a different kind of omission from Apertus:
the architecture *is* implemented and certified for `load`, `cpu_cuda` and
`chat`. What is unachievable is the check itself. That model is numerically
chaotic, and the way to establish that is to measure its floor before reading
any cross-engine number — `scripts/sensitivity_floor.py` compares an engine
against **itself** under a perturbation smaller than switching engines. On the
certified file at b10076, Runner disagrees with itself on 11 of 16 prompts when
only the KV cache precision changes, against the 9 it disagrees with llama.cpp
on. A perturbation strictly inside one build moves the output further than
changing engines does, so there is no threshold at which token-for-token
agreement would be evidence of anything. The cause is discrete top-8-of-128
expert routing over Q4_0 weights: at layer 2 the sixth and seventh selected
experts sat 0.0002 apart in weight, so a rounding difference flips an expert
and rewrites an eighth of the FFN output. The forward pass was checked directly
and does agree — layer 0 activations match the reference exactly, and the
pre-softmax router logits to ~0.1–0.5%.

Declaring the check anyway would have been worse than omitting it: under
`--require-complete` the manifest would assert a property no implementation can
satisfy, and a future reader would take the recorded failure for a defect. The
same instrument applies generally — a cross-engine logprob gap is uninterpretable
without the model's own sensitivity floor, so measure the floor first. Evidence:
`tests/compatibility/out/divergence-study-gemma4-moe-2026-08-01.json`.

The remaining chat/tool and long-context checks were run on 2026-07-23 through
the real `/v1/chat/completions` surface with fp16 KV.  Each pinned model saw a
needle near the middle of a measured 4K-token document and the same weather
tool request both without history and after 24 padded turns.  Six of eight
models passed all three assertions.  Ornith made both tool calls but emitted no
retrieval answer; Qwen 3 emitted a truncated short-tool argument and no answer
for the padded-tool or retrieval cases.  All 24 requests completed without a
server, protocol, schema or inference failure.  These are model-behavior
results, so successful execution is not reported as a quality pass.  Raw
prompts, token counts, replies and scores are committed in
`tests/compatibility/out/chat-tool-long-context-2026-07-23.json`.

## Library consumers

The optional gate starts one real Runner and exercises response parsing through
the pinned OpenAI and Anthropic Python/Node SDKs, LiteLLM and LangChain:

```sh
python3 -m venv .compat-venv
.compat-venv/bin/pip install -r tests/compatibility/requirements.txt
npm ci --ignore-scripts --prefix tests/compatibility/node
.compat-venv/bin/python scripts/consumer_compat.py
```

`make compat-consumers` runs the final command after dependencies are present.
CI runs this independently of the dependency-free Runner build and uploads the
machine-readable report.

## End-user agents

Installed clients are exercised against a real Qwen3-4B server and an isolated
sentinel fixture. Missing clients are recorded as `not_run`, never carried
forward as a fresh pass. Editor extensions such as Roo Code require a real
VS Code-compatible host; replaying a captured request is useful protocol
coverage but is not advertised as end-to-end client compatibility.

Thane is the UI consumer in scope. Its runner-client, gateway and CLI integration
tests are part of the evidence sweep; unrelated third-party web UIs are not.

Evidence lives under `tests/compatibility/out/`. Each aggregate record includes
the Runner commit, exact package/client versions, model hashes, hardware,
outcomes, exclusions and warnings. Historical reports are immutable.
