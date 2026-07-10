// Tokenizers: SentencePiece-style (llama) and byte-level BPE (gpt2).
#include "runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

enum { TT_NORMAL = 1, TT_UNKNOWN = 2, TT_CONTROL = 3, TT_USER_DEFINED = 4,
       TT_UNUSED = 5, TT_BYTE = 6 };

// ---------------------------------------------------------------- hashmap

static uint64_t fnv1a(const char *s, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) { h ^= (uint8_t)s[i]; h *= 0x100000001b3ull; }
    return h;
}

static void hmap_init(hmap *m, size_t expect) {
    size_t cap = 16;
    while (cap < expect * 2) cap <<= 1;
    m->cap = cap;
    m->e = calloc(cap, sizeof(hmap_ent));
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

bool tokenizer_init(tokenizer *t, gguf_file *g) {
    memset(t, 0, sizeof(*t));

    const char *model = gguf_get_str(g, "tokenizer.ggml.model", "llama");
    if (strcmp(model, "llama") == 0) t->model = TOK_SPM;
    else if (strcmp(model, "gpt2") == 0) t->model = TOK_BPE;
    else {
        fprintf(stderr, "error: unsupported tokenizer model '%s'\n", model);
        return false;
    }

    gguf_kv *toks = gguf_get(g, "tokenizer.ggml.tokens");
    if (!toks || toks->type != GGUF_T_ARR || toks->arr_type != GGUF_T_STR) {
        fprintf(stderr, "error: no tokenizer vocabulary in model\n");
        return false;
    }
    t->n_vocab = (int)toks->arr_n;
    t->tokens = toks->arr_str;

    gguf_kv *scores = gguf_get(g, "tokenizer.ggml.scores");
    if (scores && scores->type == GGUF_T_ARR && scores->arr_type == GGUF_T_F32 &&
        (int)scores->arr_n == t->n_vocab) {
        t->scores = malloc(sizeof(float) * t->n_vocab);
        memcpy(t->scores, scores->arr_raw, sizeof(float) * t->n_vocab);
    }
    gguf_kv *tty = gguf_get(g, "tokenizer.ggml.token_type");
    if (tty && tty->type == GGUF_T_ARR && tty->arr_type == GGUF_T_I32 &&
        (int)tty->arr_n == t->n_vocab) {
        t->ttype = malloc(sizeof(int32_t) * t->n_vocab);
        memcpy(t->ttype, tty->arr_raw, sizeof(int32_t) * t->n_vocab);
    }

    t->bos_id = (int)gguf_get_u32(g, "tokenizer.ggml.bos_token_id", 1);
    t->eos_id = (int)gguf_get_u32(g, "tokenizer.ggml.eos_token_id", 2);
    t->unk_id = (int)gguf_get_u32(g, "tokenizer.ggml.unknown_token_id", -1u);
    t->add_bos = gguf_get_bool(g, "tokenizer.ggml.add_bos_token", t->model == TOK_SPM);
    t->add_space_prefix = gguf_get_bool(g, "tokenizer.ggml.add_space_prefix", true);

    hmap_init(&t->vocab, t->n_vocab);
    for (int i = 0; i < t->n_vocab; i++)
        hmap_put(&t->vocab, t->tokens[i].s, t->tokens[i].n, i);

    // special tokens (control + user-defined) for input parsing
    t->special_ids = malloc(sizeof(int) * t->n_vocab);
    for (int i = 0; i < t->n_vocab; i++) {
        int tt = t->ttype ? t->ttype[i] : TT_NORMAL;
        if ((tt == TT_CONTROL || tt == TT_USER_DEFINED) && t->tokens[i].n > 0)
            t->special_ids[t->n_special++] = i;
    }
    sort_specials(t);

    if (t->model == TOK_BPE) {
        // GPT-2 byte <-> unicode mapping
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
        hmap_init(&t->merges, mg->arr_n);
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

static int spm_encode(tokenizer *t, const char *text, size_t n,
                      int32_t *out, int cap, int n_out) {
    if (n == 0) return n_out;
    sym_t *sym = malloc(sizeof(sym_t) * (n + 1));
    int n_sym = 0;
    for (size_t i = 0; i < n; ) {
        int l = u8_len((uint8_t)text[i]);
        if ((size_t)(i + l) > n) l = 1;
        sym[n_sym] = (sym_t){ (int)i, l, n_sym - 1, n_sym + 1 };
        n_sym++;
        i += l;
    }
    if (n_sym > 0) sym[n_sym - 1].next = -1;

    for (;;) {
        float best_score = -1e30f;
        int best = -1;
        for (int i = 0; i != -1 && sym[i].next != -1; i = sym[i].next) {
            int j = sym[i].next;
            int id = hmap_get(&t->vocab, text + sym[i].start, sym[i].len + sym[j].len);
            if (id >= 0 && t->scores && t->scores[id] > best_score) {
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
    char *buf = malloc(n * 3 + 4);
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

// BPE merge within one pre-token (already byte->unicode mapped, utf8 string)
static int bpe_word(tokenizer *t, const char *w, int n, int32_t *out, int cap, int n_out) {
    int max_sym = n + 1;
    int *st = malloc(sizeof(int) * max_sym), *ln = malloc(sizeof(int) * max_sym);
    int ns = 0;
    for (int i = 0; i < n; ) {
        int l = u8_len((uint8_t)w[i]);
        if (i + l > n) l = 1;
        st[ns] = i; ln[ns] = l; ns++;
        i += l;
    }
    char key[512];
    while (ns > 1) {
        int best_rank = INT_MAX, best = -1;
        for (int i = 0; i < ns - 1; i++) {
            int kl = ln[i] + 1 + ln[i + 1];
            if (kl >= (int)sizeof(key)) continue;
            memcpy(key, w + st[i], ln[i]);
            key[ln[i]] = ' ';
            memcpy(key + ln[i] + 1, w + st[i + 1], ln[i + 1]);
            int r = hmap_get(&t->merges, key, kl);
            if (r >= 0 && r < best_rank) { best_rank = r; best = i; }
        }
        if (best < 0) break;
        ln[best] += ln[best + 1];
        memmove(st + best + 1, st + best + 2, sizeof(int) * (ns - best - 2));
        memmove(ln + best + 1, ln + best + 2, sizeof(int) * (ns - best - 2));
        ns--;
    }
    for (int i = 0; i < ns; i++) {
        int id = hmap_get(&t->vocab, w + st[i], ln[i]);
        if (id < 0) {
            // fall back to per-character lookup
            for (int j = 0; j < ln[i]; ) {
                int l = u8_len((uint8_t)w[st[i] + j]);
                int cid = hmap_get(&t->vocab, w + st[i] + j, l);
                if (cid >= 0 && n_out < cap) out[n_out++] = cid;
                j += l;
            }
        } else if (n_out < cap) {
            out[n_out++] = id;
        }
    }
    free(st); free(ln);
    return n_out;
}

static int bpe_encode_text(tokenizer *t, const char *text, size_t n,
                           int32_t *out, int cap, int n_out) {
    if (n == 0) return n_out;
    // decode to codepoints, remembering byte offsets
    uint32_t *cp = malloc(sizeof(uint32_t) * (n + 1));
    size_t *off = malloc(sizeof(size_t) * (n + 2));
    int ncp = 0;
    for (size_t i = 0; i < n; ) {
        int l = u8_len((uint8_t)text[i]);
        if (i + l > n) l = 1;
        off[ncp] = i;
        cp[ncp++] = u8_decode(text + i, l);
        i += l;
    }
    off[ncp] = n;

    char *word = malloc(n * 2 + 8); // byte->unicode expands ascii <0x80 to <=2 bytes

    // simplified GPT-2 pre-tokenizer
    int i = 0;
    while (i < ncp) {
        int start = i;
        // contractions: 's 't 'm 'd 're 've 'll
        if (cp[i] == '\'' && i + 1 < ncp) {
            uint32_t a = cp[i + 1], b = i + 2 < ncp ? cp[i + 2] : 0;
            int adv = 0;
            if (a == 's' || a == 't' || a == 'm' || a == 'd') adv = 2;
            else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
                     (a == 'l' && b == 'l')) adv = 3;
            if (adv) { i += adv; goto emit; }
        }
        {
            // optional single leading space attached to a letter/digit/other run
            int j = (cp[i] == ' ' && i + 1 < ncp && cp_class(cp[i + 1]) != 3) ? i + 1 : i;
            if (cp_class(cp[j]) != 3) {
                int cls = cp_class(cp[j]);
                i = j;
                while (i < ncp && cp_class(cp[i]) == cls) i++;
                goto emit;
            }
            // whitespace run
            int k = i;
            while (k < ncp && cp_class(cp[k]) == 3) k++;
            if (k < ncp && k - i > 1 && cp[k - 1] == ' ') k--; // leave one ' ' for next word
            i = k;
        }
    emit:
        if (i > start) {
            // map original bytes through byte->unicode
            size_t b0 = off[start], b1 = off[i];
            int wl = 0;
            for (size_t b = b0; b < b1; b++)
                wl += u8_encode((uint32_t)t->b2u[(uint8_t)text[b]], word + wl);
            n_out = bpe_word(t, word, wl, out, cap, n_out);
        } else {
            i++; // safety
        }
    }
    free(cp); free(off); free(word);
    return n_out;
}

// ---------------------------------------------------------------- public api

int tok_encode(tokenizer *t, const char *text, int32_t *out, int cap,
               bool add_bos, bool parse_special) {
    int n_out = 0;
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
            : bpe_encode_text(t, text + seg, n - seg, out, cap, n_out);
    }
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

int tok_decode(tokenizer *t, int id, char *buf, int cap) {
    if (id < 0 || id >= t->n_vocab) return 0;
    int tt = t->ttype ? t->ttype[id] : TT_NORMAL;
    if (tt == TT_CONTROL || tt == TT_UNUSED) return 0;
    gg_str *tok = &t->tokens[id];

    if (t->model == TOK_SPM) {
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
