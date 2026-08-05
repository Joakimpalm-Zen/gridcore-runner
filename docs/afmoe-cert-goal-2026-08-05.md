# Goal: certify `afmoe` (Trinity-Nano) on this box — 2026-08-05 evening

You are on the Blackwell lab box (container, user `lab`, no sudo, 128 cores,
~200 GiB container limit). Read `~/workspace/Gridcore/MACHINE-NOTE.md` first:
borrowed machine, public repos only. Work one gate at a time; finish and
record before starting the next. Report > debug: if a gate fails twice,
write down exactly what happened and move on.

**CONTEXT HYGIENE (mandatory):** never cat/print a `.gguf`, `.jsonl`, or any
file > ~50 KB into the conversation. Inspect big things with `wc -l`,
`sha256sum`, `tail -5`. Long output goes to a file, then `tail`/`grep`.

## What this is

The runner gained a new architecture today: `afmoe` (Arcee Trinity family) —
branch `afmoe-trinity` of https://github.com/Joakimpalm-Zen/gridcore-runner.
Implementation is complete and smoke-tested on two ISAs; YOUR job is the
certification battery and the evidence record. Current evidence:
- greedy smoke passes on this box (Q8_0: "Paris. It is the largest city…")
  and on an 8 GB M1 (Q4_K_M, NEON path, ~5 tok/s);
- tokenizer differential vs the HF reference: **0/721 divergent** (after a
  whitespace-before-digit fix that is in the branch);
- `make test-quants-simd` green here.

## Machine state / cleanup first

- Models: `~/workspace/models/trinity-nano/` — `Q8_0` (6.5 GB, verified
  complete). `Q4_K_M` and `bf16` may be missing/partial: re-download from
  `arcee-ai/Trinity-Nano-Preview-GGUF` if a gate needs them (Q8_0 is the
  cert row; you may not need the others).
- **llama.cpp reference: use ONLY the prebuilt release binaries** at
  `~/workspace/Gridcore/lcpp-bin/llama-b10280/` (b10280). The source build
  in `~/workspace/Gridcore/llamacpp-fresh/` was compiled with the conda
  toolchain and **segfaults on every model** — delete that directory.
  Verify the prebuilt loads Trinity before relying on it (16-token greedy).
  If it ALSO crashes, download the previous release (b102xx) and note it.
- Delete leftovers from today's remote session: `~/workspace/Gridcore/`
  `afmoe-prep*.log bwcert* bwident* prebuilt*.log lcpp2.log lcpp.tgz
  llamacpp-build.log afmoe-wip.tgz bwcert.sh bwident.sh`; kill any stray
  `runner`/`llama-cli` processes you did not start.
- The working checkout for THIS goal: fresh clone of the public repo,
  branch `afmoe-trinity`, into `~/workspace/Gridcore/afmoe-cert/`. Build:
  `source /opt/conda/etc/profile.d/conda.sh && conda activate ccbuild`,
  then `make runner CC=x86_64-conda-linux-gnu-gcc -j` (the RUNNER builds
  fine with conda gcc; only llama.cpp did not). `./runner --version` must
  print 0.1.7-alpha.

## Gates, in order

1. **Suite regression:** `make test-quants-simd && ./test-quants-simd`;
   `python3 -m pytest -q tests/test_arch_admission.py` (needs `make runner`
   + `python3 scripts/make-test-model.py` deps; skip-note if pytest absent).
2. **Tokenizer:** `python3 scripts/difftok.py --gguf <Q8_0> --ref
   arcee-ai/Trinity-Nano-Preview` → expect **0/721**. Record the line.
3. **Greedy identity vs llama.cpp b10280** — the core gate. Protocol:
   prefer `llama-server` + `curl /completion` with `{"temperature":0,
   "n_predict":64}` over llama-cli (cli output formatting fights scripts;
   the server returns clean JSON). Runner side: `./runner -m <Q8_0> -p
   "<prompt>" -n 64 --temp 0 --gpu off` (its stdout echoes prompt +
   completion; strip the prompt). Compare COMPLETION TEXT exactly. Four
   prompts minimum, one per domain:
   a. "The capital of France is"
   b. "Write a python function that reverses a linked list."
   c. "In 1969, exactly 1234567 people watched as Apollo 11 landed. Summarize this event."
   d. "Beskriv kort hur en kvicksilvertermometer fungerar."
   Then TWO long runs (256 tokens): prompts (b) and (c).
   PASS = byte-identical completions on all six. A single mismatch = the
   gate FAILS: capture both outputs verbatim into the report, note the
   first divergent position, run the same prompt at -n <position+8> to
   confirm stability, and **STOP — do not modify engine code**. (Known
   acceptable exception: if llama.cpp emits a trailing EOS/newline the
   runner does not, trim trailing whitespace before comparing and say so.)
4. **Chat template smoke:** `./runner -m <Q8_0> --serve --port 8099 &`;
   curl an OpenAI chat completion ("What is 2+2? Answer briefly.",
   temperature 0). PASS = coherent answer, ChatML stop honored (no
   `<|im_end|>` leakage). Kill the server after.
5. **Perf row for the record:** `./runner -m <Q8_0> --bench-json --gpu off`
   once, quiet box. Record the JSON line.
6. **Secondary (unrelated to afmoe, do last): q4ne quality gate.**
   `~/workspace/Gridcore/afmoe/keep30-q4ne.gguf` (10.6 GB, this box built
   it today) vs `~/workspace/Gridcore/gridcore-runner/artifacts/
   gpt-oss-20b-keep30-MXFP4.gguf`:
   `python3 scripts/kld-compare.py --model-a <keep30> --model-b <q4ne>
   --runner ./runner --corpus tests/fixtures/mixed-corpus.txt
   --max-positions 400 --out q4ne-kld.json`.
   PASS bar: top-1 ≥ 97% AND mean KLD ≤ 0.05. Either miss = FAIL (record,
   don't rationalize). This decides whether q4ne ships as the 16 GB-Mac
   artifact.

## Deliverable

`docs/afmoe-cert-report-2026-08-05.md` on the `afmoe-trinity` branch:
per-gate verdicts with the raw evidence lines (shas, versions, the six
identity checksums, kld numbers), llama.cpp reference pinned as
`b10280 (61881b1f7)`, and a one-paragraph honest summary at the top.
If gates 1–5 all pass, ALSO update the README certified-architecture table
(follow the `qwen3moe` row's format; note "CPU only; Metal/CUDA refuse and
fall back") — the CHANGELOG Unreleased entry already exists. Commit to the
branch and push. **Do not merge to main and do not tag a release — the
owner does both.** If anything fails, push the report only.
