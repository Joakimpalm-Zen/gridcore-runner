#ifndef RUNNER_ENVELOPE_H
#define RUNNER_ENVELOPE_H

// Measured-envelope manifest reader (slice 2 of the certified-envelope gate).
//
// scripts/certify-envelope.py emits a `<model>.envelope.json` sidecar that
// INDEXES the evidence a certification run produced for one artifact. This
// reads the sidecar sitting next to a loaded model and reports whether the
// CURRENT runtime (version + compiled backend) matches what was measured. It
// is READ + REPORT ONLY — no enforcement, no refusal; a matching-tuple
// three-state gate is the next slice. It never fails a load: an absent or
// malformed manifest is a state, not an error.
//
// Resolution is EXACT-MATCH by design (no closest-class fallback): the manifest
// only speaks for the runtime version and backend it was measured on. H8/H10
// wording — a configuration "matches a measured envelope", it is not "certified"
// as a standing property.

enum envelope_state {
    ENV_NONE = 0,      // no manifest beside the model
    ENV_CERTIFIED,     // manifest matches this version+backend and passed the gate
    ENV_OUTSIDE,       // manifest matches this version+backend, verdict outside-envelope
    ENV_EXPERIMENTAL,  // manifest exists but was measured on a different
                       // version/backend, or its verdict was experimental, or
                       // it could not be parsed
};

// `model_path`: the loaded GGUF path (the manifest is `<model_path>.envelope.json`).
// `runtime_version`: RUNNER_VERSION. `backend`: the compiled backend name
// ("metal" / "cuda" / "cpu"). Writes a one-line human summary into `out`
// (<= cap, always NUL-terminated when cap > 0) and returns the state.
int envelope_report(const char *model_path, const char *runtime_version,
                    const char *backend, char *out, int cap);

#endif
