# Metal decode at long context: the chunked path already captures it

*2026-08-14. Apple M1, 8 GB, 8 GPU cores, e2b-q40 (gemma-4 E2B, 35 layers,
8 heads, head_dim 512). Diagnosis. No code changed.*

The plan carried this as "the 8-threadgroup attention ceiling needs a
flash-shaped rewrite (KV range split across threadgroups with online-softmax
combination); ~10% beyond traffic remains unexplained". Measured first, per the
prior-art rule, because that rewrite already exists in the tree
(`k_attn_chunk` / `k_attn_combine`, online-softmax combine, selected at
`n == 1`). The conclusion is that it should not be rewritten.

## A measurement trap, first

`--bench-json -c N` does **not** vary the attention span. It caps its prompt at
`min(512, c/2)`, so decode always runs at a span of a few hundred tokens no
matter what `-c` says:

| `-c` | context | prompt_tokens | generated |
|---:|---:|---:|---:|
| 512 | 512 | 255 | 256 |
| 8192 | 8192 | 512 | 256 |

An initial sweep across `-c 512…8192` read 16.87 → 15.79 tok/s and looked like
"decode is flat in context". It was measuring the same short span four times.
Long-context decode has to be driven with a long **prompt**.

## What decode actually does at long context

Real spans, via `-f` with generated prompts, `-n 48`, `-c 8192`:

| prompt tokens | prefill tok/s | decode tok/s | vs shortest |
|---:|---:|---:|---:|
| 694 | 76.60 | 15.56 | — |
| 2,326 | 64.13 | 12.61 | −19 % |
| 4,641 | 48.31 | 9.81 | −37 % |
| 8,110 | 35.01 | 7.45 | **−52 %** |

Decode halves between ~700 and ~8,100 tokens. The problem the plan names is
real.

## The chunked path helps, and helps *less* as context grows

Round-robin, chunk path at its automatic setting versus `RUNNER_METAL_ATTN_CHUNK=0`
(single-pass kernel):

| span | chunked | single-pass | delta |
|---:|---:|---:|---:|
| 2,326 | 12.49 | 11.92 | **+4.8 %** |
| 4,641 | 9.84 | 9.55 | +3.0 % |
| 8,110 | 7.34 | 7.23 | +1.5 % |

It is earning its place, and the trend is the interesting part: the longer the
context, the *less* the split is worth. That is the opposite of what a
parallelism ceiling would do.

## Parallelism is not the ceiling

The "8-threadgroup" framing is stale. `METAL_ATTN_TARGET_GROUPS` is **256**, and
the sizing aims for `TARGET_GROUPS / n_head` chunks — for this model 256/8 = 32
chunks, floored at `METAL_ATTN_MIN_CHUNK` 128 tokens. So at an 8,110-token span
the chunked path already dispatches **32 chunks × 8 heads = 256 threadgroups**,
not 8.

Forcing the chunk size directly, at a 4,641-token span — smaller chunk means
more chunks means more threadgroups, spanning 9 to 73 chunks:

| `RUNNER_METAL_ATTN_CHUNK` | chunks | decode tok/s |
|---|---:|---:|
| 0 (path disabled) | 1 | 9.59 |
| 512 | 9 | 9.40 |
| 254 (≈ the auto choice) | 19 | 9.72 |
| 128 | 37 | 9.69 |
| 64 | 73 | 9.86 |

**An 8× change in threadgroup count moves decode by under 5 %**, and the whole
column sits inside a band the run-to-run spread nearly covers. There is no
parallelism headroom to recover. A flash-shaped rewrite would be rewriting the
kernel that already exists to chase a lever that measures flat.

## So the item closes as prior-art-already-covers-it

Both halves of the plan's premise were out of date: the flash-shaped path is
already implemented and selected by default, and the threadgroup count is 256
rather than 8. The measured gain from the split is +1.5 % to +4.8 %, and more
threadgroups do not add to it.

## What the remaining 52 % is, and what would settle it

Not attention parallelism. The candidates left are memory traffic and per-token
work that grows with span:

- **KV traffic.** This model's cache is ~70 KB per token (2 × 35 layers × 1 KV
  head × 512 head_dim × 2 bytes), so an 8,110-token span holds ~568 MB that
  decode reads every token, against ~1.3 GB of per-token weight traffic (the
  2.63 GB file minus the 1.32 GB embedding table, of which decode touches one
  row). That is roughly a 43 % traffic increase for a 52 % throughput loss —
  the right order, and not obviously the whole story.
- **The SWA correction is missing from that arithmetic and matters.** gemma-4
  slides most layers, so their KV read is capped at the window rather than
  growing with span. The honest position is that the traffic estimate above is
  an upper bound and the real figure needs the per-layer global/sliding split
  counted, which this session did not do.

The next measurement, if this is picked up, is that split — KV bytes actually
read per decode token, global layers versus sliding — against measured decode
time. Guessing at "~10 % unexplained" without it is what produced the stale
premise this document is correcting.
