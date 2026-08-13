# Blackwell Q3_K rung results — 2026-08-17

## Verdict

**The sub-16 window fails at every mix point tried, rung closed.**

The least-coarse artifact below 16 GiB is N=11 (deepest 11 expert layers
Q3_K, remaining expert layers Q4_0). It passes the margin-qualified top-1
half of bar v2 at 98.75%, but fails the KLD half at 0.0528446 > 0.05. N=10
projects to 17,186,256,384 bytes (16.00595 GiB), so there is no higher-quality
mix inside the size limit.

No Hugging Face action was taken. All GGUF artifacts remained local. After
their gate JSONs were committed, the N=17, N=14, and N=12 FAIL artifacts were
deleted. The N=11 boundary FAIL remains local at
`/home/lab/workspace/models/Qwen3-30B-A3B-mix-q3k-q4/Qwen3-30B-A3B-mix-N11.gguf`.

## Protocol

- Source: verified Q8_0 GGUF, SHA-256
  `4ad960d180b16f56024f5b704697e5dd5b0837167c2e515ef0569abfc599743c`.
- Runner: `48fccf5`, built and run with the `ccbuild` conda toolchain.
- Scratch preflight: about 12 TiB free, above the required 80 GB.
- Plan: deepest N layers' three stacked expert tensors are Q3_K; all other
  expert tensors are Q4_0; all non-expert tensors remain Q8_0/F32 as sourced.
- Rule semantics: first substring match wins. Every plan was dry-run against
  the actual 579-tensor source index. Exact full tensor names matched the deep
  expert banks before the generic `_exps.weight` rule. All 435 non-expert
  tensors were unmatched.
- Integrity: every build preserved the unmatched tensor type, dimensions, and
  bytes. Each comparison covered 435 tensors / 1,674,993,664 bytes, and every
  resulting GGUF was complete with zero trailing slack.
- Zero point: Q8_0 versus itself immediately before each 400-position run,
  always 100/100 plain and margin-qualified top-1, mean KLD 0.0.
- Bar v2: 400 teacher-forced positions over
  `tests/fixtures/mixed-corpus.txt` using `scripts/kld-compare-raw.py`;
  PASS requires margin-qualified top-1 >=97% and mean KLD <=0.05. Plain top-1
  is reported beside the publication column.

## Results

| N deepest Q3_K layers | Bytes | GiB | Plain top-1 | Margin top-1 | Mean KLD | Verdict |
|---:|---:|---:|---:|---:|---:|:---|
| 17 | 16,624,743,936 | 15.4830 | 88.25% | 99.00% | 0.0550233 | FAIL |
| 14 | 16,865,392,128 | 15.7071 | 89.50% | 98.25% | 0.0542493 | FAIL |
| 12 | 17,025,824,256 | 15.8565 | 90.00% | 99.00% | 0.0526788 | FAIL |
| 11 | 17,106,040,320 | 15.9312 | 88.25% | 98.75% | 0.0528446 | FAIL — boundary |

All four failures are solely on KLD; their margin-qualified top-1 scores clear
97%. The non-monotonic plain/margin columns are reported as measured rather
than smoothed. The decisive size boundary is exact: N=11 has 73,828,864 bytes
of headroom, while N=10 exceeds 16 GiB by 6,387,200 bytes.

## Evidence and gates

Per-artifact plans, complete dry-run match listings, byte-integrity JSONs,
fresh zero-point JSONs, raw 400-position comparisons, and gate JSONs are in
`xyntetik-suite/docs/plans/blackwell-evidence-2026-08-17/`. Evidence was
committed and pushed section by section at suite commits `84687f4`, `3e9ef70`,
`3a17265`, and `aff8e04`.

Runner gates on the pulled `main` revision:

- `conda run -n ccbuild make test`: PASS, including GPU identity at a
  2.98e-7 mean-range fraction against the 2e-3 bound.
- `conda run -n ccbuild env PYTHON=python3 ./scripts/conformance.sh`:
  348 passed, 2 skipped.
- `conda run -n ccbuild python3 -m pytest tests/test_agent_torture.py
  tests/test_torture_compare.py`: 14 passed.
- `cpu_cuda` spot-check: not required; no quantizer or quant implementation
  changed.

README impact: checked; this run changes no public interface, default,
capability, or published artifact, so README.md remains accurate.
