// Token sampling and the per-family sampling presets.
#ifndef RUNNER_SAMPLE_H
#define RUNNER_SAMPLE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef struct {
    float temp, top_p, min_p, repeat_penalty;
    int top_k;
    uint64_t rng;
    int32_t recent[256];
    int n_recent, recent_head;
    // Ids exempt from the repeat penalty. A chat template puts its own turn
    // terminator in the prompt, and the prompt seeds the penalty window, so
    // penalising terminators can stop a model ever ending its turn.
    int32_t no_penalty[12];
    int n_no_penalty;
} sampler;

// validity filter for constrained sampling; return true if token is allowed
typedef bool (*sample_ok_fn)(void *ud, int token);

void sampler_reset(sampler *s);
void sampler_accept(sampler *s, int tok);
// Pick next token; ok==NULL means unconstrained; returns -1 if no token allowed.
//
// Contract: temp <= 0 returns the model's argmax (the first *valid* token in
// argmax order when constrained). The repeat penalty is a diversity knob and
// is not applied there — greedy decoding is a determinism request, and a
// caller who wants the most likely token does not want a different one back.
// Returns the chosen token id, -1 when no candidate is valid (a clean stop),
// or -2 on an allocation failure (the caller must treat this as an error, not
// a stop).
int  sample_pick(sampler *s, float *logits, int n_vocab, sample_ok_fn ok, void *ud);

// -------------------------------------------------- per-family sampling presets
//
// Model families publish recommended sampling settings and they differ a lot
// (Gemma 3 wants temp 1.0, SmolLM2 wants 0.2). One fixed set of defaults is
// wrong for most models, so runner picks a preset from the GGUF's
// `general.architecture` and `general.name`. Presets are visible rather than
// magic: logged at load, listed by --caps, reported by /v1/capabilities.

typedef struct {
    const char *name;    // preset id ("qwen3", "llama3", ..., "generic")
    const char *source;  // provenance of these numbers, printed by --caps
    float temp, top_p, min_p, repeat_penalty;
    int   top_k;
} sampler_preset;

// Values the caller set explicitly on the CLI. An explicit value always beats
// the preset, including an explicit zero.
typedef struct {
    bool  has_temp, has_top_p, has_min_p, has_top_k, has_repeat_penalty;
    float temp, top_p, min_p, repeat_penalty;
    int   top_k;
} sampler_override;

// Preset for a model. Never NULL: an unrecognised model gets "generic".
// Either string may be NULL.
const sampler_preset *sampler_preset_for(const char *arch, const char *name);
// combined matching identity: general.name + the load path's basename
// (quantizer metadata is unreliable; the filename carries the family name)
void sampler_ident(const char *name, const char *path, char *buf, size_t n);
// Enumerate the preset table; NULL past the end.
const sampler_preset *sampler_preset_at(int i);
// Apply preset then overrides to s. Touches only the five sampling knobs —
// rng state and the penalty window are left alone — and returns the preset
// used, so callers can report it.
const sampler_preset *sampler_resolve(sampler *s, const char *arch,
                                      const char *name,
                                      const sampler_override *ov);
// One-line human summary, e.g.
// "gemma3 (temp 1.00, top_p 0.95, top_k 64, min_p 0.00, repeat_penalty 1.10)"
void sampler_describe(const sampler *s, const sampler_preset *p,
                      char *buf, size_t cap);

#endif // RUNNER_SAMPLE_H
