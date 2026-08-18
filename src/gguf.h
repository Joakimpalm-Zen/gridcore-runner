// GGUF container: header, metadata key/values, tensor directory.
#ifndef RUNNER_GGUF_H
#define RUNNER_GGUF_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

enum gguf_val_type {
    GGUF_T_U8 = 0, GGUF_T_I8, GGUF_T_U16, GGUF_T_I16, GGUF_T_U32, GGUF_T_I32,
    GGUF_T_F32, GGUF_T_BOOL, GGUF_T_STR, GGUF_T_ARR, GGUF_T_U64, GGUF_T_I64,
    GGUF_T_F64,
};
typedef struct {
    uint64_t n;
    char    *s;             // NUL-terminated copy
} gg_str;

typedef struct {
    char       *key;
    uint32_t    type;       // gguf_val_type
    // scalar value (widened)
    union { uint64_t u64; int64_t i64; double f64; bool b; } v;
    uint64_t    raw;        // original scalar bits for F32/F64 finiteness checks
    gg_str      str;        // GGUF_T_STR
    // GGUF_T_ARR
    uint32_t    arr_type;
    uint64_t    arr_n;
    const void *arr_raw;    // scalar arrays: packed little-endian, points into mmap
    gg_str     *arr_str;    // string arrays: parsed copies
} gguf_kv;

typedef struct {
    char     name[128];
    uint32_t type;          // ggml_type
    uint32_t n_dims;
    uint64_t ne[4];         // ne[0] = row length (fastest dim)
    void    *data;
    uint64_t nbytes;
} gguf_tensor;

typedef struct {
    void       *map;
    size_t      map_size;
    void      **maps;       // split GGUF ownership; NULL for a single file
    size_t     *map_sizes;
    uint32_t    n_maps;
    uint64_t    mapped_size; // sum of every mapped part
    uint32_t    version;
    uint64_t    n_tensors, n_kv;
    gguf_kv    *kv;
    gguf_tensor *tensors;
    // Header-only parse (gguf_open_header): the data section was never
    // required to be present, every tensor's `data` is NULL, and `data_bytes`
    // is what the header SAYS the data section spans. Nothing may read tensor
    // contents through such a handle -- the NULL is deliberate, so a mistake
    // faults instead of reading whatever the mapping happens to hold.
    bool        header_only;
    uint64_t    data_bytes;
} gguf_file;

bool         gguf_open(gguf_file *g, const char *path);
void         gguf_close(gguf_file *g);
gguf_kv     *gguf_get(gguf_file *g, const char *key);
uint32_t     gguf_get_u32 (gguf_file *g, const char *key, uint32_t dflt);
uint32_t     gguf_get_u32_idx(gguf_file *g, const char *key, uint64_t idx,
                              uint32_t dflt);
float        gguf_get_f32 (gguf_file *g, const char *key, float dflt);
bool         gguf_get_bool(gguf_file *g, const char *key, bool dflt);
const char  *gguf_get_str (gguf_file *g, const char *key, const char *dflt);
gguf_tensor *gguf_find_tensor(gguf_file *g, const char *name);
uint64_t     gguf_mapped_size(const gguf_file *g);
// The mappings this handle owns: one for a single file, one per part for a
// split GGUF. Walk them with these rather than reading `map`/`map_size`
// directly -- those describe the FIRST part only, so anything that locks,
// unlocks or measures the whole model (mlock, munlock, residency) must
// iterate. Part `i` past the end is (NULL, 0).
uint32_t     gguf_map_count(const gguf_file *g);
void        *gguf_map_part(const gguf_file *g, uint32_t i, size_t *size);
// Parse metadata and tensor DESCRIPTORS from a file whose data section may be
// absent or short -- a header-only download, or the first few megabytes of a
// remote file. This is a separate read path on purpose: gguf_open() refuses a
// file whose data section does not cover the tensors it declares, and that
// refusal is a safety property (it is what stops a truncated or crafted file
// from pointing a tensor at memory outside the mapping). Nothing here weakens
// it. This path instead answers a different question -- "what would this file
// need if I downloaded it" -- and marks the handle so tensor data cannot be
// read through it. Split-GGUF continuation is deliberately not followed:
// callers get this part's header and the split metadata to reason with.
bool         gguf_open_header(gguf_file *g, const char *path);

#endif // RUNNER_GGUF_H
