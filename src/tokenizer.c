// Tokenizers: SentencePiece-style (llama) and byte-level BPE (gpt2).
#include <math.h>
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <limits.h>

enum { TT_NORMAL = 1, TT_UNKNOWN = 2, TT_CONTROL = 3, TT_USER_DEFINED = 4,
       TT_UNUSED = 5, TT_BYTE = 6 };

// ---------------------------------------------------------------- hashmap

static uint64_t fnv1a(const char *s, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) { h ^= (uint8_t)s[i]; h *= 0x100000001b3ull; }
    return h;
}

// Publishing the capacity before the table exists would leave hmap_put masking
// an index into a NULL `e`, so cap is set only once the allocation succeeded.
// hmap_get already treats a NULL table as empty.
static bool hmap_init(hmap *m, size_t expect) {
    size_t cap = 16;
    while (cap < expect * 2 && cap <= SIZE_MAX / 2) cap <<= 1;
    hmap_ent *e = calloc(cap, sizeof(hmap_ent));
    if (!e) { m->e = NULL; m->cap = 0; return false; }
    m->e = e;
    m->cap = cap;
    return true;
}

static void hmap_put(hmap *m, const char *k, size_t klen, int v) {
    size_t i = fnv1a(k, klen) & (m->cap - 1);
    while (m->e[i].key) {
        if (m->e[i].klen == klen && memcmp(m->e[i].key, k, klen) == 0) {
            return; // keep first entry (matches gguf duplicate handling)
        }
        i = (i + 1) & (m->cap - 1);
    }
    m->e[i] = (hmap_ent){ k, (uint32_t)klen, v };
}

static int hmap_get(const hmap *m, const char *k, size_t klen) {
    if (!m->e) return -1;
    size_t i = fnv1a(k, klen) & (m->cap - 1);
    while (m->e[i].key) {
        if (m->e[i].klen == klen && memcmp(m->e[i].key, k, klen) == 0)
            return m->e[i].val;
        i = (i + 1) & (m->cap - 1);
    }
    return -1;
}

// ---------------------------------------------------------------- utf8

static int u8_len(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

static uint32_t u8_decode(const char *s, int len) {
    const uint8_t *p = (const uint8_t *)s;
    switch (len) {
        case 2: return ((p[0] & 0x1F) << 6) | (p[1] & 0x3F);
        case 3: return ((p[0] & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
        case 4: return ((p[0] & 0x07) << 18) | ((p[1] & 0x3F) << 12) |
                       ((p[2] & 0x3F) << 6) | (p[3] & 0x3F);
        default: return p[0];
    }
}

static int u8_encode(uint32_t cp, char *out) {
    if (cp < 0x80)  { out[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

// ---------------------------------------------------------------- init

static int cmp_special(const void *a, const void *b, void *ctx) {
    tokenizer *t = ctx;
    uint64_t la = t->tokens[*(const int *)a].n, lb = t->tokens[*(const int *)b].n;
    return la < lb ? 1 : la > lb ? -1 : 0;
}

// qsort_r portability shim: simple insertion sort (special token lists are tiny)
static void sort_specials(tokenizer *t) {
    for (int i = 1; i < t->n_special; i++) {
        int key = t->special_ids[i], j = i - 1;
        while (j >= 0 && cmp_special(&t->special_ids[j], &key, t) > 0) {
            t->special_ids[j + 1] = t->special_ids[j];
            j--;
        }
        t->special_ids[j + 1] = key;
    }
}

// All-zero scores carry no ordering, so the merge loop in spm_encode would
// just take the leftmost candidate every round.
static bool spm_scores_degenerate(const tokenizer *t) {
    if (!t->scores) return true;
    for (int i = 0; i < t->n_vocab; i++)
        if (t->scores[i] != 0.0f) return false;
    return true;
}

// Rebuild scores from merge rank, highest score first, so spm_encode picks
// merges in BPE order. A piece that is no merge's result gets -inf: BPE only
// ever produces multi-character pieces through a merge, so anything absent
// here must never be merged into. Requires t->vocab to be populated.
//
// Returns false only when an allocation failed. A model that simply carries no
// merges is left exactly as it was and still reports success — that is not an
// error, and the caller must not abort the load over it.
static bool spm_scores_from_merges(tokenizer *t, gguf_file *g) {
    gguf_kv *mg = gguf_get(g, "tokenizer.ggml.merges");
    if (!mg || mg->type != GGUF_T_ARR || mg->arr_type != GGUF_T_STR || mg->arr_n == 0)
        return true;

    if (!t->scores) {
        t->scores = malloc(sizeof(float) * (size_t)t->n_vocab);
        if (!t->scores) return false;
    }
    for (int i = 0; i < t->n_vocab; i++) t->scores[i] = -FLT_MAX;

    char buf[512];
    for (uint64_t r = 0; r < mg->arr_n; r++) {
        const char *m = mg->arr_str[r].s;
        size_t n = mg->arr_str[r].n;
        // "left right": split on the first space, the pieces themselves use
        // U+2581 rather than a literal space
        const char *sep = memchr(m, ' ', n);
        if (!sep || n - 1 >= sizeof(buf)) continue;
        size_t left = (size_t)(sep - m);
        memcpy(buf, m, left);
        memcpy(buf + left, sep + 1, n - left - 1);
        int id = hmap_get(&t->vocab, buf, n - 1);
        if (id >= 0) t->scores[id] = -(float)r;
    }
    return true;
}

bool tokenizer_init(tokenizer *t, gguf_file *g) {
    memset(t, 0, sizeof(*t));

    const char *model = gguf_get_str(g, "tokenizer.ggml.model", "llama");
    if (strcmp(model, "llama") == 0) t->model = TOK_SPM;
    else if (strcmp(model, "gpt2") == 0) t->model = TOK_BPE;
    else if (strcmp(model, "gemma4") == 0) t->model = TOK_BPE_SPM;
    else {
        fprintf(stderr, "error: unsupported tokenizer model '%s'\n", model);
        return false;
    }

    // Split rules for the BPE families we have ground truth for. Everything
    // else, including a missing key, keeps the original GPT-2 behavior rather
    // than being silently retokenized by rules it was not checked against.
    const char *pre = gguf_get_str(g, "tokenizer.ggml.pre", "");
    if (strcmp(pre, "llama-bpe") == 0)   t->pre = TOK_PRE_LLAMA3;
    else if (strcmp(pre, "qwen2") == 0 ||
             strcmp(pre, "qwen35") == 0) t->pre = TOK_PRE_QWEN2;
    else if (strcmp(pre, "smollm") == 0) t->pre = TOK_PRE_SMOLLM;
    else if (strcmp(pre, "afmoe") == 0)  t->pre = TOK_PRE_AFMOE;
    else if (strcmp(pre, "tekken") == 0) t->pre = TOK_PRE_TEKKEN;
    else if (strcmp(pre, "llama4") == 0) t->pre = TOK_PRE_LLAMA4;
    else                                 t->pre = TOK_PRE_GPT2;

    gguf_kv *toks = gguf_get(g, "tokenizer.ggml.tokens");
    if (!toks || toks->type != GGUF_T_ARR || toks->arr_type != GGUF_T_STR) {
        fprintf(stderr, "error: no tokenizer vocabulary in model\n");
        return false;
    }
    // arr_n is a 64-bit count straight out of the file; a value past INT_MAX
    // would wrap n_vocab negative and turn every `sizeof(x) * n_vocab` below
    // into a huge or negative allocation size.
    if (toks->arr_n > INT_MAX) {
        fprintf(stderr, "error: tokenizer vocabulary is implausibly large\n");
        return false;
    }
    t->n_vocab = (int)toks->arr_n;
    t->tokens = toks->arr_str;

    gguf_kv *scores = gguf_get(g, "tokenizer.ggml.scores");
    if (scores && scores->type == GGUF_T_ARR && scores->arr_type == GGUF_T_F32 &&
        (int)scores->arr_n == t->n_vocab) {
        t->scores = malloc(sizeof(float) * (size_t)t->n_vocab);
        if (!t->scores) return false;
        memcpy(t->scores, scores->arr_raw, sizeof(float) * (size_t)t->n_vocab);
    }
    gguf_kv *tty = gguf_get(g, "tokenizer.ggml.token_type");
    if (tty && tty->type == GGUF_T_ARR && tty->arr_type == GGUF_T_I32 &&
        (int)tty->arr_n == t->n_vocab) {
        t->ttype = malloc(sizeof(int32_t) * (size_t)t->n_vocab);
        if (!t->ttype) return false;
        memcpy(t->ttype, tty->arr_raw, sizeof(int32_t) * (size_t)t->n_vocab);
    }

    t->bos_id = (int)gguf_get_u32(g, "tokenizer.ggml.bos_token_id", 1);
    t->eos_id = (int)gguf_get_u32(g, "tokenizer.ggml.eos_token_id", 2);
    t->unk_id = (int)gguf_get_u32(g, "tokenizer.ggml.unknown_token_id", -1u);
    t->add_bos = gguf_get_bool(g, "tokenizer.ggml.add_bos_token", t->model == TOK_SPM);
    t->add_space_prefix = gguf_get_bool(g, "tokenizer.ggml.add_space_prefix", true);

    if (!hmap_init(&t->vocab, (size_t)t->n_vocab)) return false;
    for (int i = 0; i < t->n_vocab; i++)
        hmap_put(&t->vocab, t->tokens[i].s, t->tokens[i].n, i);

    // Many conversions (TheBloke's GGUFs among them) write all-zero scores and
    // put the ordering in tokenizer.ggml.merges instead. Without this, every
    // score ties and "llama" encodes as "▁llam"+"a" instead of "▁ll"+"ama".
    // Models that carry neither usable scores nor merges are left as they were.
    if (t->model == TOK_SPM && spm_scores_degenerate(t) &&
        !spm_scores_from_merges(t, g))
        return false;

    // special tokens (control + user-defined) for input parsing
    t->special_ids = malloc(sizeof(int) * (size_t)t->n_vocab);
    if (!t->special_ids) return false;
    for (int i = 0; i < t->n_vocab; i++) {
        int tt = t->ttype ? t->ttype[i] : TT_NORMAL;
        if ((tt == TT_CONTROL || tt == TT_USER_DEFINED) && t->tokens[i].n > 0)
            t->special_ids[t->n_special++] = i;
    }
    sort_specials(t);

    if (t->model == TOK_BPE || t->model == TOK_BPE_SPM) {
        // GPT-2 byte <-> unicode mapping (unused by BPE_SPM, harmless)
        for (int i = 0; i < 512; i++) t->u2b[i] = -1;
        int n = 0;
        for (int b = 0; b < 256; b++) {
            int keep = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) ||
                       (b >= 0xAE && b <= 0xFF);
            int cp = keep ? b : 256 + n++;
            t->b2u[b] = cp;
            t->u2b[cp] = b;
        }
        // merge ranks
        gguf_kv *mg = gguf_get(g, "tokenizer.ggml.merges");
        if (!mg || mg->type != GGUF_T_ARR || mg->arr_type != GGUF_T_STR) {
            fprintf(stderr, "error: BPE tokenizer has no merges\n");
            return false;
        }
        if (!hmap_init(&t->merges, mg->arr_n)) return false;
        for (uint64_t i = 0; i < mg->arr_n; i++)
            hmap_put(&t->merges, mg->arr_str[i].s, mg->arr_str[i].n, (int)i);
    }
    return true;
}

void tokenizer_free(tokenizer *t) {
    free(t->scores);
    free(t->ttype);
    free(t->vocab.e);
    free(t->merges.e);
    free(t->merges_buf);
    free(t->special_ids);
    memset(t, 0, sizeof(*t));
}

// ---------------------------------------------------------------- SPM encode

// greedy highest-score bigram merging over a doubly linked symbol list
typedef struct { int start, len, prev, next; } sym_t;

// ------------------------------------------------------- merge candidates
//
// Both merge loops used to rescan every adjacent pair after every merge, which
// is O(n^2) in the length of one segment — and a segment is not a word. The
// SentencePiece path never splits at all, and the gemma-4 BPE path splits only
// on newline runs, so one long line is one unit. Measured on 4,000 characters
// of ordinary prose in a single line: 31 ms on the certified European
// SentencePiece models, 62 ms on gemma-4-E4B, against 0.31 ms on Qwen2.5,
// whose GPT-2 pre-tokenizer hands these loops a word at a time. At 16,000
// characters gemma-4 took 940 ms. Nine of the fourteen models on the bench box
// are affected, including five of the six certified European ones.
//
// The queue makes it O(n log n) and is required to be EXACT: the tokenizer
// feeds ids to the model, so any change of output would invalidate every
// greedy_reference certification. Ordering therefore reproduces the old scan
// exactly — best key first, and on a tie the leftmost pair, because the old
// loops compared with a strict `>` / `<` while walking left to right.
//
// `key` is "smaller is better": the negated piece score for SentencePiece, the
// merge rank for BPE. Ranks are far below 2^24 so a float holds them exactly.
typedef struct { float key; int l, r, len_l, len_r; } mcand;

static bool mc_better(const mcand *a, const mcand *b) {
    if (a->key != b->key) return a->key < b->key;
    return a->l < b->l;
}

// The queue starts in a fixed buffer. A GPT-2 style pre-tokenizer hands these
// loops a word at a time — three or four symbols — and for those the malloc
// was the whole cost: the first measured version of this change was 20% SLOWER
// on Qwen2.5 and gpt-oss while being 80x faster on gemma-4. Nothing allocates
// until a segment actually produces more candidates than fit here.
// Below this many symbols the rescan wins outright: a GPT-2 pre-tokenizer
// hands these loops three or four symbols, where the queue's push/pop
// bookkeeping costs more than re-reading every pair. Measured on a 4,000-char
// prompt: Qwen2.5 0.31 ms rescan against 0.39 ms queue. Both paths are
// verified byte-identical over the differential corpus, so which one runs is
// purely a cost decision.
#define MERGE_QUEUE_MIN 24
#define MH_FIXED 64

// Test hook: force the rescan (0) or the queue (1) regardless of segment
// length; -1 restores the length rule. The two paths are required to agree on
// every input, and tests/test_tokenizer_merge.c is that requirement -- without
// a way to run both on one binary the only gate would be a diff against an old
// build, which nobody runs.
static int g_merge_force = -1;
void tok_merge_force(int on) { g_merge_force = on; }
static bool use_merge_queue(int n_sym) {
    return g_merge_force >= 0 ? g_merge_force != 0 : n_sym >= MERGE_QUEUE_MIN;
}
typedef struct { mcand *e; int n, cap; mcand fixed[MH_FIXED]; } mheap;

static void mh_init(mheap *h) { h->e = h->fixed; h->n = 0; h->cap = MH_FIXED; }
static void mh_free(mheap *h) { if (h->e != h->fixed) free(h->e); }

static bool mh_push(mheap *h, mcand c) {
    if (h->n == h->cap) {
        int cap = h->cap * 2;
        mcand *e;
        if (h->e == h->fixed) {
            e = malloc(sizeof(mcand) * (size_t)cap);
            if (e) memcpy(e, h->fixed, sizeof(mcand) * (size_t)h->n);
        } else {
            e = realloc(h->e, sizeof(mcand) * (size_t)cap);
        }
        if (!e) return false;
        h->e = e;
        h->cap = cap;
    }
    int i = h->n++;
    h->e[i] = c;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (!mc_better(&h->e[i], &h->e[p])) break;
        mcand tmp = h->e[p]; h->e[p] = h->e[i]; h->e[i] = tmp;
        i = p;
    }
    return true;
}

static bool mh_pop(mheap *h, mcand *out) {
    if (h->n == 0) return false;
    *out = h->e[0];
    h->e[0] = h->e[--h->n];
    for (int i = 0;;) {
        int a = 2 * i + 1, b = a + 1, best = i;
        if (a < h->n && mc_better(&h->e[a], &h->e[best])) best = a;
        if (b < h->n && mc_better(&h->e[b], &h->e[best])) best = b;
        if (best == i) break;
        mcand tmp = h->e[best]; h->e[best] = h->e[i]; h->e[i] = tmp;
        i = best;
    }
    return true;
}

// A popped candidate is stale when either side has since been merged into
// something longer. Lengths grow strictly and indices never move, so the pair
// plus the two lengths it was pushed with identifies it uniquely — which is
// what lets superseded entries stay in the queue instead of being hunted down.
//
// That holds only because an ABSORBED symbol has its length zeroed. Absorbing
// updates the surviving left symbol and leaves the right one untouched, so a
// symbol that was itself swallowed as somebody's right-hand side would keep
// both its old length and its old `next` — and a candidate naming it would
// still pass this test and then merge a symbol no longer in the list. The
// rescanning loops this replaced could not hit that, because they only ever
// looked at symbols they could reach. Caught by the differential harness on
// EuroLLM, where "  index." tokenized as three ids instead of two.
#define MC_LIVE_SYM(c, SYM) \
    ((SYM)[(c).l].next == (c).r && (SYM)[(c).l].len == (c).len_l && \
     (SYM)[(c).r].len == (c).len_r)
#define MC_LIVE(c, LN, NEXT) \
    ((NEXT)[(c).l] == (c).r && (LN)[(c).l] == (c).len_l && (LN)[(c).r] == (c).len_r)

// The score floor the old scan started from. A piece that is no merge's result
// carries -FLT_MAX (see spm_scores_from_merges) and must never be merged;
// the old loop excluded it by initialising best_score to -1e30f, so the same
// constant is the admission test here.
#define SPM_SCORE_FLOOR (-1e30f)

static bool spm_push_pair(tokenizer *t, const char *text, const sym_t *sym,
                          mheap *h, int l) {
    if (l < 0) return true;
    int r = sym[l].next;
    if (r < 0) return true;
    int id = hmap_get(&t->vocab, text + sym[l].start, sym[l].len + sym[r].len);
    if (id < 0 || !(t->scores[id] > SPM_SCORE_FLOOR)) return true;
    mcand c = { -t->scores[id], l, r, sym[l].len, sym[r].len };
    return mh_push(h, c);
}

static int spm_encode(tokenizer *t, const char *text, size_t n,
                      int32_t *out, int cap, int n_out) {
    if (n == 0) return n_out;
    if (n > SIZE_MAX / sizeof(sym_t) - 1) { t->encode_oom = true; return n_out; }
    sym_t *sym = malloc(sizeof(sym_t) * (n + 1));
    if (!sym) { t->encode_oom = true; return n_out; }  // signalled via t; tok_encode returns -1
    int n_sym = 0;
    for (size_t i = 0; i < n; ) {
        int l = u8_len((uint8_t)text[i]);
        if ((size_t)(i + l) > n) l = 1;
        sym[n_sym] = (sym_t){ (int)i, l, n_sym - 1, n_sym + 1 };
        n_sym++;
        i += l;
    }
    if (n_sym > 0) sym[n_sym - 1].next = -1;

    // With no scores the old scan could never select a pair, so nothing merges.
    if (t->scores && !use_merge_queue(n_sym)) {
        for (;;) {
            float best_score = SPM_SCORE_FLOOR;
            int best = -1;
            for (int i = 0; i != -1 && sym[i].next != -1; i = sym[i].next) {
                int j = sym[i].next;
                int id = hmap_get(&t->vocab, text + sym[i].start,
                                  sym[i].len + sym[j].len);
                if (id >= 0 && t->scores[id] > best_score) {
                    best_score = t->scores[id];
                    best = i;
                }
            }
            if (best < 0) break;
            int j = sym[best].next;
            sym[best].len += sym[j].len;
            sym[best].next = sym[j].next;
            if (sym[j].next != -1) sym[sym[j].next].prev = best;
        }
    } else if (t->scores) {
        mheap h;
        mh_init(&h);
        bool oom = false;
        for (int i = 0; i != -1 && sym[i].next != -1; i = sym[i].next)
            if (!spm_push_pair(t, text, sym, &h, i)) { oom = true; break; }
        mcand c;
        while (!oom && mh_pop(&h, &c)) {
            if (!MC_LIVE_SYM(c, sym)) continue;   // superseded by an earlier merge
            int j = sym[c.l].next;
            sym[c.l].len += sym[j].len;
            sym[c.l].next = sym[j].next;
            if (sym[j].next != -1) sym[sym[j].next].prev = c.l;
            sym[j].len = 0;               // absorbed: fails MC_LIVE from here on
            // only the merged symbol's two neighbours are new pairs
            if (!spm_push_pair(t, text, sym, &h, sym[c.l].prev) ||
                !spm_push_pair(t, text, sym, &h, c.l)) { oom = true; break; }
        }
        mh_free(&h);
        if (oom) { free(sym); t->encode_oom = true; return n_out; }
    }

    for (int i = 0; i != -1; i = sym[i].next) {
        int id = hmap_get(&t->vocab, text + sym[i].start, sym[i].len);
        if (id >= 0) {
            if (n_out < cap) out[n_out++] = id;
        } else {
            // byte fallback
            for (int b = 0; b < sym[i].len; b++) {
                char name[8];
                snprintf(name, sizeof(name), "<0x%02X>", (uint8_t)text[sym[i].start + b]);
                int bid = hmap_get(&t->vocab, name, 6);
                if (bid < 0) bid = t->unk_id;
                if (bid >= 0 && n_out < cap) out[n_out++] = bid;
            }
        }
    }
    free(sym);
    return n_out;
}

static int spm_encode_text(tokenizer *t, const char *text, size_t n,
                           int32_t *out, int cap, int n_out, bool first_segment) {
    // replace ' ' with U+2581, optionally prefix a space
    if (n > (SIZE_MAX - 4) / 3) { t->encode_oom = true; return n_out; }
    char *buf = malloc(n * 3 + 4);
    if (!buf) { t->encode_oom = true; return n_out; }
    size_t m = 0;
    if (t->add_space_prefix && first_segment && n > 0) {
        memcpy(buf + m, "\xE2\x96\x81", 3); m += 3;
    }
    for (size_t i = 0; i < n; i++) {
        if (text[i] == ' ') { memcpy(buf + m, "\xE2\x96\x81", 3); m += 3; }
        else buf[m++] = text[i];
    }
    n_out = spm_encode(t, buf, m, out, cap, n_out);
    free(buf);
    return n_out;
}

// ---------------------------------------------------------------- BPE encode

static int cp_class(uint32_t c) {
    // 0 = letter, 1 = digit, 2 = other(punct/symbol), 3 = whitespace
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0x0B || c == 0x0C ||
        c == 0xA0 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 ||
        c == 0x202F || c == 0x205F || c == 0x3000)
        return 3;
    if (c >= '0' && c <= '9') return 1;
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80) return 0;
    return 2;
}

// Codepoints >= 0x80 are treated as letters, which is right for scripts but
// wrong for symbols. Only these ranges need excluding to match \p{L} on real
// text: emoji and dingbats otherwise glue onto adjacent words and shift every
// pre-token boundary after them. Full Unicode tables buy nothing beyond this.
static bool cp_symbol(uint32_t c) {
    return (c >= 0x2000 && c <= 0x206F) || (c >= 0x2100 && c <= 0x2BFF) ||
           (c >= 0x2E00 && c <= 0x2E7F) || (c >= 0x3000 && c <= 0x303F) ||
           (c >= 0xFE00 && c <= 0xFE0F) || (c >= 0x1F000 && c <= 0x1FAFF);
}

// Combining marks are Unicode category M, not L. Treating every non-symbol
// codepoint above ASCII as a letter incorrectly glued Indic/Thai vowel signs
// and viramas into \p{L}+ runs. These are the mark blocks exercised by the
// supported tokenizer corpus; expand alongside differential fixtures when a
// new script is admitted.
static bool cp_mark(uint32_t c) {
    return (c >= 0x0300 && c <= 0x036F) || // Combining Diacritical Marks
           (c >= 0x0900 && c <= 0x0903) || // Devanagari signs
           (c >= 0x093A && c <= 0x094F) ||
           (c >= 0x0951 && c <= 0x0957) ||
           (c >= 0x0962 && c <= 0x0963) ||
           c == 0x0E31 ||                  // Thai combining vowels/tones
           (c >= 0x0E34 && c <= 0x0E3A) ||
           (c >= 0x0E47 && c <= 0x0E4E);
}

static bool cp_letter(uint32_t c) {
    if (cp_class(c) == 3) return false;
    return ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= 0x80 && !cp_symbol(c) && !cp_mark(c)));
}

static bool cp_digit(uint32_t c)  { return c >= '0' && c <= '9'; }
static bool cp_space(uint32_t c)  { return cp_class(c) == 3; }
// \p{L} and \p{N} both excluded, i.e. the regex's [^\s\p{L}\p{N}] class
static bool cp_other(uint32_t c)  { return !cp_space(c) && !cp_letter(c) && !cp_digit(c); }

// case-insensitive 's 't 're 've 'm 'll 'd, as (?i:...) in the newer regexes
static int contraction_len(const uint32_t *cp, int i, int ncp) {
    if (cp[i] != '\'' || i + 1 >= ncp) return 0;
    uint32_t a = cp[i + 1] | 32, b = (i + 2 < ncp) ? (cp[i + 2] | 32) : 0;
    if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
    return 0;
}

// One pre-token of the newer BPE regex, returning the end index:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}{1,max_digits}
//   | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
static int pre_split_next(const uint32_t *cp, int i, int ncp, int max_digits) {
    int adv = contraction_len(cp, i, ncp);
    if (adv) return i + adv;

    // a single non-letter, non-digit, non-newline character may lead a letter run
    if (!cp_letter(cp[i]) && !cp_digit(cp[i]) && cp[i] != '\r' && cp[i] != '\n' &&
        i + 1 < ncp && cp_letter(cp[i + 1])) {
        int j = i + 1;
        while (j < ncp && cp_letter(cp[j])) j++;
        return j;
    }
    if (cp_letter(cp[i])) {
        int j = i;
        while (j < ncp && cp_letter(cp[j])) j++;
        return j;
    }
    if (cp_digit(cp[i])) {
        int j = i;
        while (j < ncp && cp_digit(cp[j]) && j - i < max_digits) j++;
        return j;
    }
    {   // optional leading space, then a run of symbols/punctuation
        int j = (cp[i] == ' ' && i + 1 < ncp && cp_other(cp[i + 1])) ? i + 1 : i;
        if (j < ncp && cp_other(cp[j])) {
            while (j < ncp && cp_other(cp[j])) j++;
            while (j < ncp && (cp[j] == '\r' || cp[j] == '\n')) j++;
            return j;
        }
    }
    {
        int j = i;
        while (j < ncp && cp_space(cp[j])) j++;
        // \s*[\r\n]+ runs through the last newline of the whitespace run, so
        // "\n\n" stays one pre-token; trailing spaces after it split off
        int last_nl = -1;
        for (int k = i; k < j; k++)
            if (cp[k] == '\r' || cp[k] == '\n') last_nl = k;
        if (last_nl >= 0) return last_nl + 1;
        // \s+(?!\S) keeps a trailing space for the next pre-token
        if (j < ncp && j - i > 1) j--;
        return j;
    }
}

// One pre-token of the original GPT-2 regex, bounded by end so a caller can
// restrict it to a segment:
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// The contractions are case-sensitive here, unlike the newer (?i:...) regexes.
static int gpt2_split_next(const uint32_t *cp, int i, int end) {
    if (cp[i] == '\'' && i + 1 < end) {
        uint32_t a = cp[i + 1], b = (i + 2 < end) ? cp[i + 2] : 0;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 2;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
            (a == 'l' && b == 'l')) return i + 3;
    }
    // an optional leading space joins a run of a single class
    int j = (cp[i] == ' ' && i + 1 < end && cp_class(cp[i + 1]) != 3) ? i + 1 : i;
    if (cp_class(cp[j]) != 3) {
        int cls = cp_class(cp[j]);
        int k = j;
        while (k < end && cp_class(cp[k]) == cls) k++;
        return k;
    }
    // \s+(?!\S) hands the final whitespace character to the next pre-token,
    // whatever that character is: "\t\tx" is "\t", "\t", "x", not "\t\t", "x"
    int k = i;
    while (k < end && cp_class(cp[k]) == 3) k++;
    if (k < end && k - i > 1) k--;
    return k;
}

// ---------------------------------------------------------------- tekken
//
// Cased-letter classification. The tekken regex is the only split rule here
// that distinguishes upper from lower case, so this is the only place a case
// table is needed. Everything outside these ranges is treated as caseless
// (\p{Lo}/\p{Lm}/\p{M}), which matches BOTH halves of the regex's letter
// classes and so behaves exactly like the plain \p{L}+ the other families use.
// That is the correct default: CJK, Thai, Arabic, Hebrew, Devanagari and the
// rest of the caseless scripts are the majority of what >= 0x80 contains.
// Latin Extended-A is laid out in capital/small pairs, but the parity of those
// pairs flips twice inside the block, so a plain (c & 1) test is wrong for
// exactly the ranges Polish, Czech and Slovak live in -- it reads "ź" (U+017A)
// as a capital and splits "łódź" into łód|ź. Returns 1 upper, -1 lower, 0 for
// the caseless/unpaired characters.
static int latin_ext_a_case(uint32_t c) {
    if (c == 0x138 || c == 0x149 || c == 0x17F) return -1; // ĸ, ŉ, ſ: no capital
    if (c == 0x178) return 1;                              // Ÿ, paired down at 0x00FF
    if (c <= 0x137) return (c & 1) == 0 ? 1 : -1;
    if (c <= 0x148) return (c & 1) == 1 ? 1 : -1;          // parity flips at Ĺ
    if (c <= 0x177) return (c & 1) == 0 ? 1 : -1;          // and back at Ŋ
    return (c & 1) == 1 ? 1 : -1;                          // and again at Ź
}

static bool cp_upper(uint32_t c) {
    if (c >= 'A' && c <= 'Z') return true;
    if (c < 0x80) return false;
    if (c >= 0xC0 && c <= 0xDE) return c != 0xD7;          // Latin-1, minus MULTIPLICATION SIGN
    if (c >= 0x100 && c <= 0x17F) return latin_ext_a_case(c) > 0;
    if (c >= 0x1E00 && c <= 0x1E95) return (c & 1) == 0;   // Latin Extended Additional
    if (c >= 0x1EA0 && c <= 0x1EFF) return (c & 1) == 0;   // (0x1E96..0x1E9F have no capital)
    if (c >= 0x386 && c <= 0x3AB) return true;             // Greek capitals
    if (c >= 0x400 && c <= 0x42F) return true;             // Cyrillic capitals
    return false;
}

static bool cp_lower(uint32_t c) {
    if (c >= 'a' && c <= 'z') return true;
    if (c < 0x80) return false;
    if (c >= 0xDF && c <= 0xFF) return c != 0xF7;          // Latin-1, minus DIVISION SIGN
    if (c >= 0x100 && c <= 0x17F) return latin_ext_a_case(c) < 0;
    if (c >= 0x1E00 && c <= 0x1E9F) return (c & 1) == 1 || c >= 0x1E96;
    if (c >= 0x1EA0 && c <= 0x1EFF) return (c & 1) == 1;
    if (c >= 0x3AC && c <= 0x3CE) return true;             // Greek smalls
    if (c >= 0x430 && c <= 0x45F) return true;             // Cyrillic smalls
    return false;
}

// [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}] -- a letter that is not lowercase
// Used only by tekken, and \p{M} is the reason they exist separately. The
// comments here have always spelled the class with \p{M} in it, but cp_letter
// deliberately excludes combining marks -- correct for llama3, qwen2 and
// smollm, whose regexes all say a plain \p{L}+, and wrong for tekken, which
// carries \p{M} in both its letter classes:
//
//   [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+
//
// so a Devanagari virama or a Thai vowel sign must stay INSIDE the run rather
// than ending it. Excluding them split three of the 721 differential strings
// away from the reference: नमस्ते, हिन्दी and สวัสดี.
static bool cp_letter_upperish(uint32_t c) {
    return (cp_letter(c) || cp_mark(c)) && !cp_lower(c);
}
// [\p{Ll}\p{Lm}\p{Lo}\p{M}] -- a letter that is not uppercase
static bool cp_letter_lowerish(uint32_t c) {
    return (cp_letter(c) || cp_mark(c)) && !cp_upper(c);
}

// The two letter alternatives of the tekken regex, tried in order at i:
//   [^\r\n\p{L}\p{N}]? [upperish]* [lowerish]+
//   [^\r\n\p{L}\p{N}]? [upperish]+ [lowerish]*
// Returns the end index, or i if neither matches.
//
// This is what makes tekken split on case: "camelCase" is camel|Case and
// "XMLHttpRequest" is XMLHttp|Request, where every other family in the lineup
// takes the whole run as one pre-token. Caseless scripts are unaffected,
// because a caseless letter satisfies both classes.
static int tekken_letters(const uint32_t *cp, int i, int ncp) {
    // the optional leading character: anything that is not CR/LF, letter or digit
    int s = i;
    if (!cp_letter(cp[i]) && !cp_digit(cp[i]) && cp[i] != '\r' && cp[i] != '\n')
        s = i + 1;
    // a mark can open the run under tekken's classes, so the fast bail has to
    // admit one too or the loops below never get to see it
    if (s >= ncp || !(cp_letter(cp[s]) || cp_mark(cp[s]))) return i;

    // upperish* is greedy, but must leave at least one lowerish for alt 1.
    // Backtracking is unnecessary: a lowerish that is also upperish (caseless)
    // is consumed by the greedy run and still satisfies the tail, and a
    // strictly-lowercase character stops the run on its own.
    int u = s;
    while (u < ncp && cp_letter_upperish(cp[u])) u++;

    int j = u;
    while (j < ncp && cp_letter_lowerish(cp[j])) j++;

    // alt 1 requires lowerish+; alt 2 requires upperish+. If the greedy
    // upperish run swallowed everything (all-caps, or caseless script), that is
    // alt 2 with an empty lowerish tail.
    if (j > u || u > s) return j;
    return i;
}

// One pre-token of the tekken regex (Mistral's tokenizer v3/v7 and Apertus):
//   [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+
//   | [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*
//   | \p{N} | ?[^\s\p{L}\p{N}]+[\r\n/]* | \s*[\r\n]+ | \s+(?!\S) | \s+
//
// Three things separate it from the llama3/qwen2 regex handled by
// pre_split_next: no contraction alternative (so "it's" is it|'s only because
// ' leads the letter run, and "it'S" is it|'S rather than it|'|S), single
// digits, and '/' joins the trailing run of a punctuation pre-token so that
// "//" in a comment or a URL path does not split away from its newline.
static int tekken_split_next(const uint32_t *cp, int i, int ncp) {
    int adv = tekken_letters(cp, i, ncp);
    if (adv > i) return adv;

    if (cp_digit(cp[i])) return i + 1;              // \p{N}, one digit at a time

    {   // optional leading space, then a run of symbols/punctuation
        int j = (cp[i] == ' ' && i + 1 < ncp && cp_other(cp[i + 1])) ? i + 1 : i;
        if (j < ncp && cp_other(cp[j])) {
            while (j < ncp && cp_other(cp[j])) j++;
            // [\r\n/]* -- '/' here, not in the llama3/qwen2 form
            while (j < ncp && (cp[j] == '\r' || cp[j] == '\n' || cp[j] == '/')) j++;
            return j;
        }
    }
    {   // \s*[\r\n]+ | \s+(?!\S) | \s+ -- identical to the other newer regexes
        int j = i;
        while (j < ncp && cp_space(cp[j])) j++;
        int last_nl = -1;
        for (int k = i; k < j; k++)
            if (cp[k] == '\r' || cp[k] == '\n') last_nl = k;
        if (last_nl >= 0) return last_nl + 1;
        if (j < ncp && j - i > 1) j--;
        return j;
    }
}

// One pre-token of the o200k-family regex (llama.cpp maps pre "llama4" onto
// PRE_TYPE_GPT4O; Muse Glimmer ships this):
//   [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//   | [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//   | \p{N}{1,3} | ?[^\s\p{L}\p{N}]+[\r\n/]* | \s*[\r\n]+ | \s+(?!\S) | \s+
//
// The letter alternatives are exactly tekken's (case-transition splitting),
// with the contraction attached as a SUFFIX of the letter run — not a leading
// alternative as in llama3, so "it's" stays one pre-token here where llama3
// gives it|'s. Digits run to three like llama3, and '/' joins the punctuation
// tail like tekken.
static int llama4_split_next(const uint32_t *cp, int i, int ncp) {
    int adv = tekken_letters(cp, i, ncp);
    if (adv > i) {
        int c = contraction_len(cp, adv, ncp);   // (?i:...) — already caseless
        return adv + c;
    }

    if (cp_digit(cp[i])) {
        int j = i;
        while (j < ncp && cp_digit(cp[j]) && j - i < 3) j++;
        return j;
    }

    {   // optional leading space, then a run of symbols/punctuation
        int j = (cp[i] == ' ' && i + 1 < ncp && cp_other(cp[i + 1])) ? i + 1 : i;
        if (j < ncp && cp_other(cp[j])) {
            while (j < ncp && cp_other(cp[j])) j++;
            while (j < ncp && (cp[j] == '\r' || cp[j] == '\n' || cp[j] == '/')) j++;
            return j;
        }
    }
    {   // \s*[\r\n]+ | \s+(?!\S) | \s+
        int j = i;
        while (j < ncp && cp_space(cp[j])) j++;
        int last_nl = -1;
        for (int k = i; k < j; k++)
            if (cp[k] == '\r' || cp[k] == '\n') last_nl = k;
        if (last_nl >= 0) return last_nl + 1;
        if (j < ncp && j - i > 1) j--;
        return j;
    }
}

// afmoe (Arcee Trinity) composes three passes (llama.cpp
// unicode_regex_split_custom_afmoe + llama-vocab.cpp pre "afmoe"), folded
// into one scanner because earlier passes only ever produce fragments the
// main regex cannot re-merge:
//   1. every maximal digit run splits into groups of three FROM THE RIGHT
//      (1234567 -> 1|234|567): first group len%3 (if nonzero), then threes;
//   2. CJK/Asian-script runs are isolated, so a leading space or Latin
//      letter can never glue onto them;
//   3. main regex:  [ascii-punct][A-Za-z]+
//                 | [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
//                 |  ?[\p{P}\p{S}]+[\r\n]*
//                 | \s*[\r\n]+ | \s+(?!\S) | \s+
// \p{P}\p{S} is approximated as cp_other minus marks, matching how this
// file approximates the classes everywhere else; the differential fixture
// set is the gate for the approximation.
static bool cp_afmoe_cjk(uint32_t c) {
    return (c >= 0x4E00 && c <= 0x9FFF) ||   // CJK Unified
           (c >= 0x3400 && c <= 0x4DBF) ||   // CJK Ext A
           (c >= 0xF900 && c <= 0xFAFF) ||   // CJK Compat Ideographs
           (c >= 0x3040 && c <= 0x309F) ||   // Hiragana
           (c >= 0x30A0 && c <= 0x30FF) ||   // Katakana
           (c >= 0xFF65 && c <= 0xFF9F) ||   // Halfwidth Katakana
           (c >= 0x2F00 && c <= 0x2FDF) ||   // Kangxi Radicals
           (c >= 0x0E40 && c <= 0x0E7F) ||   // Thai (trailing block)
           (c >= 0x0E80 && c <= 0x0EFF) ||   // Lao
           (c >= 0x1780 && c <= 0x17FF) ||   // Khmer
           (c >= 0x1000 && c <= 0x109F) ||   // Myanmar
           (c >= 0xAA60 && c <= 0xAA7F) ||   // Myanmar Ext A
           (c >= 0xA9E0 && c <= 0xA9FF) ||   // Myanmar Ext B
           (c >= 0xAC00 && c <= 0xD7AF) ||   // Hangul Syllables
           (c >= 0x1100 && c <= 0x11FF);     // Hangul Jamo
}
static bool cp_ascii_punct(uint32_t c) {
    return (c >= '!' && c <= '/') || (c >= ':' && c <= '@') ||
           (c >= '[' && c <= '`') || (c >= '{' && c <= '~');
}
static bool cp_punct_sym(uint32_t c) {       // ~ [\p{P}\p{S}]
    return cp_other(c) && !cp_mark(c);
}
static int afmoe_split_next(const uint32_t *cp, int i, int ncp) {
    if (cp_digit(cp[i])) {
        int L = 0;
        while (i + L < ncp && cp_digit(cp[i + L])) L++;
        int first = L % 3;
        return i + (first ? first : 3);
    }
    if (cp_afmoe_cjk(cp[i])) {
        int j = i;
        while (j < ncp && cp_afmoe_cjk(cp[j])) j++;
        return j;
    }
    // [ascii-punct][A-Za-z]+  (leads the alternation, so "'s" stays whole)
    if (cp_ascii_punct(cp[i]) && i + 1 < ncp &&
        ((cp[i + 1] >= 'a' && cp[i + 1] <= 'z') ||
         (cp[i + 1] >= 'A' && cp[i + 1] <= 'Z'))) {
        int j = i + 1;
        while (j < ncp && ((cp[j] >= 'a' && cp[j] <= 'z') ||
                           (cp[j] >= 'A' && cp[j] <= 'Z'))) j++;
        return j;
    }
    // [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+ — one optional "other" char
    // (space, mark; digits cannot reach here) then a letter/mark run that
    // stops at CJK, which pass 2 has already made its own fragment
    {
        int s = i;
        uint32_t c = cp[i];
        if (c != '\r' && c != '\n' && !cp_letter(c) && !cp_punct_sym(c) &&
            i + 1 < ncp)
            s = i + 1;
        if (s < ncp && (cp_letter(cp[s]) || cp_mark(cp[s])) &&
            !cp_afmoe_cjk(cp[s])) {
            int j = s;
            while (j < ncp && (cp_letter(cp[j]) || cp_mark(cp[j])) &&
                   !cp_afmoe_cjk(cp[j])) j++;
            if (j > s) return j;
        }
    }
    //  ?[\p{P}\p{S}]+[\r\n]*
    {
        int j = (cp[i] == ' ' && i + 1 < ncp && cp_punct_sym(cp[i + 1]))
                ? i + 1 : i;
        if (j < ncp && cp_punct_sym(cp[j])) {
            while (j < ncp && cp_punct_sym(cp[j]) && !cp_afmoe_cjk(cp[j])) j++;
            while (j < ncp && (cp[j] == '\r' || cp[j] == '\n')) j++;
            return j;
        }
    }
    // \s*[\r\n]+ | \s+(?!\S) | \s+
    {
        int j = i;
        while (j < ncp && cp_space(cp[j])) j++;
        int last_nl = -1;
        for (int k = i; k < j; k++)
            if (cp[k] == '\r' || cp[k] == '\n') last_nl = k;
        if (last_nl >= 0) return last_nl + 1;
        // the digit and CJK passes run BEFORE the main regex in the
        // reference, so a following digit/CJK char is a fragment boundary
        // there: \s+(?!\S) sees end-of-fragment and keeps every space
        if (j < ncp && j - i > 1 &&
            !cp_digit(cp[j]) && !cp_afmoe_cjk(cp[j])) j--;
        return j;
    }
}

// smollm runs a Digits(individual_digits) pass before the GPT-2 regex, so every
// digit stands alone and never takes a leading space. Splitting first also
// bounds the regex: in "  12" the space run ends at the digit and stays whole,
// where "  leading" gives a space back to the word.
static int smollm_split_next(const uint32_t *cp, int i, int ncp) {
    if (cp_digit(cp[i])) return i + 1;
    int seg = i;
    while (seg < ncp && !cp_digit(cp[seg])) seg++;
    return gpt2_split_next(cp, i, seg);
}

// BPE merge within one pre-token (already byte->unicode mapped, utf8 string)
// Same candidate rule as the SentencePiece side, keyed on merge rank.
static bool bpe_push_pair(tokenizer *t, const char *w, const int *st,
                          const int *ln, const int *next, mheap *h, int l) {
    if (l < 0) return true;
    int r = next[l];
    if (r < 0) return true;
    int kl = ln[l] + 1 + ln[r];
    if (kl >= 512) return true;      // the old scan skipped over-long keys too
    char key[512];
    memcpy(key, w + st[l], (size_t)ln[l]);
    key[ln[l]] = ' ';
    memcpy(key + ln[l] + 1, w + st[r], (size_t)ln[r]);
    int rank = hmap_get(&t->merges, key, (size_t)kl);
    if (rank < 0) return true;
    mcand c = { (float)rank, l, r, ln[l], ln[r] };
    return mh_push(h, c);
}

static int bpe_word(tokenizer *t, const char *w, int n, int32_t *out, int cap, int n_out) {
    if (n <= 0) return n_out;
    size_t max_sym = (size_t)n + 1;
    // one block, four slices: the linked list needs two arrays the compacting
    // version did not, and four separate mallocs per pre-token showed up
    int *mem = malloc(sizeof(int) * max_sym * 4);
    if (!mem) { t->encode_oom = true; return n_out; }
    int *st = mem, *ln = mem + max_sym, *next = mem + 2 * max_sym,
        *prev = mem + 3 * max_sym;
    int ns = 0;
    for (int i = 0; i < n; ) {
        int l = u8_len((uint8_t)w[i]);
        if (i + l > n) l = 1;
        st[ns] = i; ln[ns] = l;
        prev[ns] = ns - 1; next[ns] = ns + 1;
        ns++;
        i += l;
    }
    if (ns > 0) next[ns - 1] = -1;
    // A linked list rather than a compacted array: the queue holds indices, so
    // they have to stay put. Order along the list is still left to right, which
    // is what makes "smaller index" the same tie-break the old scan had.
    if (!use_merge_queue(ns)) {
        char key[512];
        while (ns > 1) {
            int best_rank = INT_MAX, best = -1;
            for (int i = 0; i != -1 && next[i] != -1; i = next[i]) {
                int j = next[i];
                int kl = ln[i] + 1 + ln[j];
                if (kl >= (int)sizeof(key)) continue;
                memcpy(key, w + st[i], (size_t)ln[i]);
                key[ln[i]] = ' ';
                memcpy(key + ln[i] + 1, w + st[j], (size_t)ln[j]);
                int r = hmap_get(&t->merges, key, (size_t)kl);
                if (r >= 0 && r < best_rank) { best_rank = r; best = i; }
            }
            if (best < 0) break;
            int j = next[best];
            ln[best] += ln[j];
            next[best] = next[j];
            if (next[j] != -1) prev[next[j]] = best;
            ns--;
        }
    } else {
        mheap h;
        mh_init(&h);
        bool oom = false;
        for (int i = 0; i != -1 && next[i] != -1; i = next[i])
            if (!bpe_push_pair(t, w, st, ln, next, &h, i)) { oom = true; break; }
        mcand c;
        while (!oom && mh_pop(&h, &c)) {
            if (!MC_LIVE(c, ln, next)) continue;
            int j = next[c.l];
            ln[c.l] += ln[j];
            next[c.l] = next[j];
            if (next[j] != -1) prev[next[j]] = c.l;
            ln[j] = 0;                    // absorbed: fails MC_LIVE from here on
            if (!bpe_push_pair(t, w, st, ln, next, &h, prev[c.l]) ||
                !bpe_push_pair(t, w, st, ln, next, &h, c.l)) { oom = true; break; }
        }
        mh_free(&h);
        if (oom) { free(mem); t->encode_oom = true; return n_out; }
    }
    for (int i = 0; i != -1; i = next[i]) {
        int id = hmap_get(&t->vocab, w + st[i], ln[i]);
        if (id < 0) {
            // fall back to per-character lookup, then to the <0xNN> byte
            // pieces. Silently dropping a character loses input outright:
            // gemma4's vocabulary has no literal CR or U+00A0 piece, and both
            // must decompose to their UTF-8 bytes the way the reference does.
            // A gpt2-style BPE vocabulary carries no <0xNN> pieces and its
            // byte->unicode alphabet already covers every byte, so this path
            // stays unreachable there.
            for (int j = 0; j < ln[i]; ) {
                int l = u8_len((uint8_t)w[st[i] + j]);
                int cid = hmap_get(&t->vocab, w + st[i] + j, l);
                if (cid >= 0) {
                    if (n_out < cap) out[n_out++] = cid;
                } else {
                    for (int b = 0; b < l; b++) {
                        char name[8];
                        snprintf(name, sizeof(name), "<0x%02X>",
                                 (uint8_t)w[st[i] + j + b]);
                        int bid = hmap_get(&t->vocab, name, 6);
                        if (bid < 0) bid = t->unk_id;
                        if (bid >= 0 && n_out < cap) out[n_out++] = bid;
                    }
                }
                j += l;
            }
        } else if (n_out < cap) {
            out[n_out++] = id;
        }
    }
    free(mem);
    return n_out;
}

static int bpe_encode_text(tokenizer *t, const char *text, size_t n,
                           int32_t *out, int cap, int n_out) {
    if (n == 0) return n_out;
    if (n > SIZE_MAX / sizeof(size_t) - 2 || n > (SIZE_MAX - 8) / 2) return n_out;
    // decode to codepoints, remembering byte offsets
    uint32_t *cp = malloc(sizeof(uint32_t) * (n + 1));
    size_t *off = malloc(sizeof(size_t) * (n + 2));
    // byte->unicode expands ascii <0x80 to <=2 bytes
    char *word = malloc(n * 2 + 8);
    if (!cp || !off || !word) { free(cp); free(off); free(word); t->encode_oom = true; return n_out; }
    int ncp = 0;
    for (size_t i = 0; i < n; ) {
        int l = u8_len((uint8_t)text[i]);
        if (i + l > n) l = 1;
        off[ncp] = i;
        cp[ncp++] = u8_decode(text + i, l);
        i += l;
    }
    off[ncp] = n;

    // The split rules come from tokenizer.ggml.pre; anything unrecognised keeps
    // the original GPT-2 regex it has always used.
    int max_digits = t->pre == TOK_PRE_LLAMA3 ? 3 : t->pre == TOK_PRE_QWEN2 ? 1 : 0;
    for (int i = 0; i < ncp; ) {
        int end = t->pre == TOK_PRE_TEKKEN    ? tekken_split_next(cp, i, ncp)
                : t->pre == TOK_PRE_LLAMA4    ? llama4_split_next(cp, i, ncp)
                : max_digits                  ? pre_split_next(cp, i, ncp, max_digits)
                : t->pre == TOK_PRE_SMOLLM    ? smollm_split_next(cp, i, ncp)
                : t->pre == TOK_PRE_AFMOE     ? afmoe_split_next(cp, i, ncp)
                                              : gpt2_split_next(cp, i, ncp);
        if (end <= i) end = i + 1; // never stall
        // map the original bytes of this pre-token through byte->unicode
        size_t b0 = off[i], b1 = off[end];
        int wl = 0;
        for (size_t b = b0; b < b1; b++)
            wl += u8_encode((uint32_t)t->b2u[(uint8_t)text[b]], word + wl);
        n_out = bpe_word(t, word, wl, out, cap, n_out);
        i = end;
    }
    free(cp); free(off); free(word);
    return n_out;
}

// gemma4: SPM-style BPE — the normalizer replaces spaces with U+2581 and BPE
// merges run over raw UTF-8 with no byte-encoding; only newline runs split
// pre-tokens (merge keys never contain newlines)
static int bpe_spm_encode_text(tokenizer *t, const char *text, size_t n,
                               int32_t *out, int cap, int n_out) {
    if (n == 0) return n_out;
    if (n > (SIZE_MAX - 4) / 3) { t->encode_oom = true; return n_out; }
    char *buf = malloc(n * 3 + 4);
    if (!buf) { t->encode_oom = true; return n_out; }
    size_t m = 0;
    for (size_t i = 0; i < n; i++) {
        if (text[i] == ' ') { memcpy(buf + m, "\xE2\x96\x81", 3); m += 3; }
        else buf[m++] = text[i];
    }
    size_t i = 0;
    while (i < m) {
        bool nl = buf[i] == '\n';
        size_t j = i;
        while (j < m && (buf[j] == '\n') == nl) j++;
        n_out = bpe_word(t, buf + i, (int)(j - i), out, cap, n_out);
        i = j;
    }
    free(buf);
    return n_out;
}

// ---------------------------------------------------------------- public api

int tok_encode(tokenizer *t, const char *text, int32_t *out, int cap,
               bool add_bos, bool parse_special) {
    int n_out = 0;
    t->encode_oom = false;   // set by a helper that had to drop a segment
    if (add_bos && t->add_bos && t->bos_id >= 0 && n_out < cap)
        out[n_out++] = t->bos_id;

    size_t n = strlen(text);
    size_t seg = 0;   // start of pending plain-text segment
    bool first = true;
    for (size_t i = 0; i < n; ) {
        int matched = -1;
        if (parse_special) {
            for (int s = 0; s < t->n_special; s++) {
                gg_str *tok = &t->tokens[t->special_ids[s]];
                if (tok->n <= n - i && memcmp(text + i, tok->s, tok->n) == 0) {
                    matched = t->special_ids[s];
                    break;
                }
            }
        }
        if (matched >= 0) {
            if (i > seg) {
                n_out = t->model == TOK_SPM
                    ? spm_encode_text(t, text + seg, i - seg, out, cap, n_out, first)
                    : t->model == TOK_BPE_SPM
                    ? bpe_spm_encode_text(t, text + seg, i - seg, out, cap, n_out)
                    : bpe_encode_text(t, text + seg, i - seg, out, cap, n_out);
                first = false;
            }
            if (n_out < cap) out[n_out++] = matched;
            i += t->tokens[matched].n;
            seg = i;
            first = false;
        } else {
            i++;
        }
    }
    if (n > seg) {
        n_out = t->model == TOK_SPM
            ? spm_encode_text(t, text + seg, n - seg, out, cap, n_out, first)
            : t->model == TOK_BPE_SPM
            ? bpe_spm_encode_text(t, text + seg, n - seg, out, cap, n_out)
            : bpe_encode_text(t, text + seg, n - seg, out, cap, n_out);
    }
    // A dropped segment would look like a legitimately shorter prompt; refuse
    // to return a silently truncated tokenization.
    if (t->encode_oom) return -1;
    return n_out;
}

bool tok_is_control(tokenizer *t, int id) {
    if (id < 0 || id >= t->n_vocab) return false;
    int tt = t->ttype ? t->ttype[id] : TT_NORMAL;
    return tt == TT_CONTROL;
}

const char *tok_raw(tokenizer *t, int id) {
    if (id < 0 || id >= t->n_vocab) return NULL;
    return t->tokens[id].s;
}

int tok_find(tokenizer *t, const char *s) {
    return hmap_get(&t->vocab, s, strlen(s));
}

// Encode raw bytes verbatim: no BOS, no specials, and — the point — none of
// tok_encode's segment-start normalization (SPM's leading-space prefix), so
// the token list round-trips to exactly the input bytes wherever the vocab
// can express them. Grammar fast-forward drafts through this; a vocab that
// cannot round-trip some byte simply yields a shorter (or empty) draft.
int tok_encode_raw(tokenizer *t, const char *text, int n,
                   int32_t *out, int cap) {
    t->encode_oom = false;
    if (n <= 0) return 0;
    return t->model == TOK_SPM
        ? spm_encode_text(t, text, n, out, cap, 0, false)
        : t->model == TOK_BPE_SPM
        ? bpe_spm_encode_text(t, text, n, out, cap, 0)
        : bpe_encode_text(t, text, n, out, cap, 0);
}

int tok_decode(tokenizer *t, int id, char *buf, int cap) {
    if (id < 0 || id >= t->n_vocab) return 0;
    int tt = t->ttype ? t->ttype[id] : TT_NORMAL;
    if (tt == TT_CONTROL || tt == TT_UNUSED) return 0;
    gg_str *tok = &t->tokens[id];

    if (t->model == TOK_SPM || t->model == TOK_BPE_SPM) {
        if (tt == TT_BYTE) {
            // "<0xXX>"
            if (tok->n == 6 && cap >= 1) {
                buf[0] = (char)strtol(tok->s + 3, NULL, 16);
                return 1;
            }
            return 0;
        }
        int m = 0;
        for (size_t i = 0; i < tok->n && m < cap - 3; ) {
            if (i + 3 <= tok->n && memcmp(tok->s + i, "\xE2\x96\x81", 3) == 0) {
                buf[m++] = ' ';
                i += 3;
            } else {
                buf[m++] = tok->s[i++];
            }
        }
        return m;
    }

    // BPE: map codepoints back to bytes
    int m = 0;
    for (size_t i = 0; i < tok->n && m < cap; ) {
        int l = u8_len((uint8_t)tok->s[i]);
        if (i + l > tok->n) l = 1;
        uint32_t c = u8_decode(tok->s + i, l);
        if (c < 512 && t->u2b[c] >= 0) buf[m++] = (char)t->u2b[c];
        else {
            // not in byte-alphabet (e.g. user-defined token text): copy as-is
            for (int j = 0; j < l && m < cap; j++) buf[m++] = tok->s[i + j];
        }
        i += l;
    }
    return m;
}
