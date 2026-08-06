# Goal: artifact certification matrix — GPT-OSS × Gemma 4 derivative ecosystems

You are on the Blackwell lab box (container, user `lab`, no sudo, 128 cores,
~200 GiB container memory limit, big NVIDIA GPU). Read
`~/workspace/Gridcore/MACHINE-NOTE.md` first: borrowed machine, public repos
only. This session is meant to run LONG — the deliverable is a stream of
committed results, not one final report. **Commit and push after EVERY
artifact verdict.** If the session dies mid-list, everything already landed.

**CONTEXT HYGIENE (mandatory):** never cat/print a `.gguf`, `.jsonl`, or any
file > ~50 KB into the conversation. Inspect big things with `wc -l`,
`sha256sum`, `tail -5`. Long output goes to a file, then `tail`/`grep`.

## The thesis this session tests

The runner's README certifies **architectures** against one pinned artifact
each. The GPT-OSS and Gemma 4 ecosystems have exploded into derivatives —
QAT builds, mixed-tensor GGUFs, expert-pruned models, Frankenmodels,
fine-tunes, MTP drafters. "Architecture supported" and "artifact certified"
are different claims, and this session measures the gap: run the standard
gate battery against as many materially-different artifacts as time and
disk allow, and record a manifest-grade identity for each — e.g.
`GPTOSS / 24L / 32E / top4 / MXFP4_MOE / Harmony` — never just a family
name. A pruned 12-expert "gpt-oss" must not inherit certification because
its metadata says `gpt_oss`; whether it loads and what it scores is exactly
what we want to know. **Admission refusals and gate failures are results,
not setbacks — record them with the same care as passes.**

## Setup

- Fresh clone of https://github.com/Joakimpalm-Zen/gridcore-runner, branch
  `cert-matrix`, into `~/workspace/Gridcore/cert-matrix/`. Build:
  `source /opt/conda/etc/profile.d/conda.sh && conda activate ccbuild`,
  then `make runner CC=x86_64-conda-linux-gnu-gcc -j`. `./runner --version`
  must print 0.1.8-alpha (or later).
- **llama.cpp reference: ONLY the prebuilt release binaries** at
  `~/workspace/Gridcore/lcpp-bin/llama-b10280/` (pin: `b10280 (61881b1f7)`).
  Any conda-toolchain source build of llama.cpp segfaults on every model —
  do not build it from source. If the prebuilt refuses a NEW arch variant
  (possible for late-2026 releases), download the newest prebuilt release
  binary tarball, pin its tag in the report, and use it for that artifact
  only — note the split reference.
- `llama-quantize` lives at `~/.conda/envs/ccbuild/bin/` if requant is
  needed for prune experiments.
- Reference greedy runs via `llama-server` + `curl /completion` with
  `{"temperature":0, "cache_prompt":false}` — never llama-cli (interactive
  runaway), never with prompt cache (nondeterministic reference).
- Downloads: `huggingface-cli download <repo> <file> --local-dir ...` (or
  `curl -L` on resolved URLs). **Disk discipline: check `df -h
  ~/workspace` before every download; keep ≥ 25 GB free; at most two
  artifacts on disk at once; delete each artifact after its verdict lands
  (the sha in the report is the durable identity). The 120B tier is the
  only exception and has its own rules below.**

## Naming reality check

The roster below comes from ecosystem research pasted by the owner. Repo
names may be paraphrased. For each item: resolve the actual repo via the HF
API (`curl -s 'https://huggingface.co/api/models?search=<terms>&limit=20'`),
pin the exact `repo/filename` + sha256 in the report. If no plausible repo
exists, record **NOT FOUND** with the search terms used and move on — that
is a finding about the research, not a failure of yours.

## Gate battery (per artifact)

1. **Identity:** sha256 of the file; metadata dump via the runner's loader
   output (arch, layer count, expert count/active, quant types per tensor
   class — for MoE note whether expert FFNs are MXFP4_MOE vs requantized).
   Write the manifest line. This gate alone is valuable: Bartowski's
   "Q6_K" GPT-OSS keeps expert FFNs at MXFP4_MOE — record what each file
   ACTUALLY is, not what its name says.
2. **Admission:** does the runner load it at all? A clean refusal with a
   correct reason is a PASS for the refusal path — record the exact error.
   **Never modify engine/architecture code to make something load. STOP
   rule, no exceptions: implementation happens on the owner's machine,
   not here.** Metadata-dump + refusal reason = complete result.
3. **Tokenizer:** `python3 scripts/difftok.py --gguf <file> --ref <HF
   repo>` where a HF reference exists (official + major conversions).
   Expect 0 divergent. Fine-tunes that changed the tokenizer: record the
   count, not a rationalization.
4. **Reference gate — pick the bar the family has earned (do not chase
   what the README already documents as unattainable):**
   - `gpt-oss` family: agreement with llama.cpp sits at the model's own
     sensitivity floor; do NOT gate on token identity. Run
     `scripts/kld-compare.py --model-a <file> --model-b <same file>
     --runner ./runner` against llama.cpp on the SAME artifact (see the
     script's --help; it needs the exact model basename), 400 positions,
     mixed corpus. Bar: **top-1 ≥ 97%, mean KLD ≤ 0.05.**
   - `gemma4-moe`: same KLD protocol, same bar (README: "too numerically
     chaotic to gate on token identity").
   - `gemma4` dense / E-series and anything else: greedy token identity
     vs b10280, six prompts (four short domains + two 256-token runs, the
     afmoe protocol in docs/afmoe-cert-goal-2026-08-05.md gate 3). One
     mismatch = record first divergent position, then fall back to the
     KLD protocol and report both.
5. **cpu_cuda:** greedy 64 tokens `--gpu off` vs GPU run, byte-identical
   (`RUNNER_CUDA_TC=0` per the harness convention). MoE models where the
   README notes CUDA caveats: record, don't fight.
6. **Chat smoke:** `--serve`, one OpenAI chat completion, temperature 0.
   PASS = coherent answer, no template leakage. GPT-OSS derivatives:
   verify no Harmony channel markup (`<|channel|>`, `<|message|>`) leaks
   into content; Harmony is load-bearing for this family — a fine-tune
   that broke it is a real finding.
7. **Perf row:** `./runner -m <file> --bench-json` (GPU) and `--gpu off`
   (CPU) once each, quiet box. Record both JSON lines.

Verdict per artifact: **CERTIFIED / CERTIFIED-WITH-CAVEAT (say which gate
and why) / FAILED (say which gate) / REFUSED (admission, with reason) /
NOT FOUND.**

## Roster, in priority order

Work top to bottom; each tier is roughly "value per GB downloaded".
Skip nothing silently — an item you decide not to run gets a one-line
reason in the status table.

**Tier 1 — canonical + QAT (the certification-envelope core):**
1. ggml-org `gpt-oss-20b` MXFP4 GGUF — re-baseline on this box; fast, and
   every later gpt-oss row diffs against it.
2. Bartowski `gpt-oss-20b` **Q6_K_L** — the mixed-tensor trap: verify the
   expert FFNs are still MXFP4_MOE and the outer tensors changed; compare
   quality + size vs (1).
3. Unsloth `gpt-oss-20b` Q4_K_M-class GGUF — alternate conversion path.
4. Google/Unsloth `gemma-4-26B-A4B-it` **QAT Q4_0** GGUF — MoE + QAT; the
   single most important new artifact in the list.
5. Bartowski `gemma-4-26B-A4B-it` Q4_K_M (~17 GB) — post-training quant of
   the same MoE, head-to-head against the QAT build. This pair is the
   whole QAT-vs-PTQ story in one row.
6. `gemma-4-12B-it` QAT Q4_0 — dense unified arch, QAT variant of the
   already-pinned 12B.
7. `gemma-4-E2B-it` QAT Q4_0 — smallest effective/nested; quick row.
8. `gemma-4-31B-it` QAT Q4_0 (~17–18 GB) — flagship dense QAT.

**Tier 2 — mutations (derivative-topology compatibility):**
9. GPT-OSS **Nano 9B** GGUF (community prune: ~12 experts) — the "expert
   roster differs from family" admission test. Whatever happens is gold.
10. GPT-OSS 120B **REAP 58B** GGUF — expert-pruned 120B derivative;
    ~30 GB class at Q4. Same reason.
11. `gpt-oss-safeguard-20b` GGUF (find a conversion) — same architecture,
    specialized fine-tune; Harmony stress.
12. One GPT-OSS 20B coder fine-tune GGUF (DavidAU NEO family or similar) —
    chat + tool smoke emphasis.
13. One abliterated GPT-OSS 20B **MXFP4** GGUF (Huihui v2 class) —
    alignment-modified checkpoint on native expert format.
14. Gemma 4 `12B Coder` community fine-tune GGUF (the popular composer/
    coder 12B) — real-world fine-tune of the dense arch.
15. HauhauCS `gemma-4-26B-A4B` QAT uncensored GGUF — QAT + community
    mutation combo.
16. BrainStorm `GPT-OSS 36B` GGUF if one exists — Frankenmodel; NOT FOUND
    is an acceptable verdict.

**Tier 3 — speculative decoding (MTP):**
17. Gemma 4 26B-A4B (or 12B) + its **MTP drafter** GGUF (Unsloth packages
    drafters alongside). Runner flags: `--draft <drafter.gguf>
    --draft-k 4`. Gates: (a) drafter admission (unknown drafter arch →
    REFUSED is a fine result); (b) **losslessness: greedy output WITH
    draft must be byte-identical to WITHOUT** on the six-prompt protocol;
    (c) measured speedup (bench with/without). A speedup number with a
    losslessness proof is a README-grade result.

**Tier 4 — big iron (only after Tiers 1–2 are committed):**
18. `gpt-oss-120b` MXFP4 GGUF (~60 GB). Disk exception: allowed to be the
    only artifact on disk. GPU offload what fits, CPU the rest; full gate
    battery; delete immediately after.
19. `gpt-oss-safeguard-120b` — only if 18 went smoothly and time remains.
20. 220A20B expanded-expert FrankenMoE — metadata/admission row ONLY
    (download only if disk trivially allows; a REFUSED/loads-metadata
    result is all we want).

**Note-only (no download):** NVFP4, FP8, BnB, MLX artifacts — the runner
has no kernels for these formats; record one line each in the report
explaining what they are and why they're out of scope for a GGUF runtime.

## The 16 GB lever (the previous goal, revisited)

The standing target: a genuinely usable model inside a 16 GB Mac envelope.
Current holder: gpt-oss-20b keep-30 MXFP4 (ships with `iogpu.wired_limit_mb`
guidance); the q4ne sub-4-bit attempt FAILED its quality gate (22.5% top-1)
and is dead. Every CERTIFIED artifact above gets a 16 GB verdict:

- **fits** — file + KV at `-c 8192` residues under ~15 GiB; or
- **fits with levers** — `--kv q8`, context cap, or (MoE only) an expert
  prune: build keep-N variants with `--prune-experts` (see `--help`; plan
  JSONs via `scripts/` moe-prune tooling), requant if needed, then re-gate
  the pruned artifact against its OWN parent with kld-compare. **Quality
  bar is the same one that killed q4ne: top-1 ≥ 97%, mean KLD ≤ 0.05
  — do not rationalize a miss**; or
- **does not fit** — say so.

Prime candidates to actually sweep: Gemma 4 26B-A4B QAT Q4_0 (QAT weights
may tolerate pruning better than PTQ — this is the experiment), Nano 9B
(fits trivially — the question is whether its quality survived pruning),
REAP 58B (probably unreachable — prove it with numbers, one keep-N point
is enough).

**Live verification, not just arithmetic:** for the best 1–2 candidates,
run decode under a REAL 16 GiB cap using the ballast method (hard-won
rules, follow exactly): regulate against the CONTAINER cgroup `memory.max`
(200 GiB — `/proc/meminfo` shows the host's 250 GB and lies); fill ballast
with random bytes (zeroed bytearrays are lazy pages); pace the last GBs
slower than swap-out or the OOM killer wins; never set `oom_score_adj` on
the ballast; FREEZE the ballast size once at target — any release valve
lets page cache displace ballast 1:1 and un-caps the experiment (symptom:
impossibly warm tok/s); fill the 21 GB disk swap first, then stop touching
it. Usable bar: **≥ 5 tok/s sustained decode under the cap.** Record
tok/s cold and warm.

Deliverable: a ranked 16 GB-envelope table in the report with an explicit
recommendation — either "X beats keep-30 because [numbers]" or "no
candidate beats keep-30", with the same confidence either way.

## Commit discipline (the point of this session)

- `docs/cert-matrix-status.md`: one table row per roster item
  (# / artifact / resolved repo / verdict / one-line note). Update it
  with EVERY verdict.
- `docs/cert-matrix-2026-08-05.md`: one section per artifact — manifest
  line, sha, gate-by-gate evidence (raw numbers, not summaries), verdict.
- After each artifact: `git add docs/ && git commit -m "cert-matrix:
  <artifact> — <VERDICT>" && git push origin cert-matrix`. Never batch
  two artifacts into one commit. If push fails, keep committing locally
  and note the push failure in the status file.
- **Do not touch README.md or CHANGELOG.md** — the owner integrates
  certified rows after reviewing the reports (several rows will need
  caveat wording that is their call). Do not merge, do not tag.

## STOP rules

- No engine, architecture, kernel, or tokenizer code changes. Ever. A
  model that needs code changes gets a REFUSED/FAILED row with exact
  evidence (metadata dump, error text) — that row is the deliverable.
- A gate that fails twice gets recorded and you move on. Report > debug.
- Do not chase token identity for gpt-oss / gemma4-moe (documented floor
  and chaos caveats — the KLD bar IS the gate for those).
- Prune sweeps: maximum 3 keep-N points per model, then verdict. The
  16 GB lever must not starve the roster.
- Keep a running `disk`/`time` sanity check between tiers; if the box is
  under pressure, prefer finishing the current tier cleanly over starting
  the next.
