# Compatibility program

Runner is an inference engine. Its compatibility boundary is model execution
and the APIs consumed by Clu, SDKs, gateways, frameworks and agent clients. A
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
CPU/CUDA token identity for all seven GPU-capable targets. Qwen3.5/Ornith is
CPU-only by design.

The model manifest deliberately excludes Apertus from forward-pass coverage.
Runner supports its `tekken` tokenizer and chat template, but not the Apertus
tensor architecture; treating it as a Qwen2 model would be a false positive.

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

Clu is the UI consumer in scope. Its runner-client, gateway and CLI integration
tests are part of the evidence sweep; unrelated third-party web UIs are not.

Evidence lives under `tests/compatibility/out/`. Each aggregate record includes
the Runner commit, exact package/client versions, model hashes, hardware,
outcomes, exclusions and warnings. Historical reports are immutable.
