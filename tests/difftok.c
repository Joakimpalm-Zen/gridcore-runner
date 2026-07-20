// difftok: dump Runner's token ids for a corpus, for differential testing
// against the HuggingFace reference tokenizer.
//
//   difftok <model.gguf> <corpus.txt>
//
// The corpus is one JSON string literal per line (see
// scripts/tokenizer-corpus.py), so leading/trailing whitespace and newlines
// survive the file round trip -- those are exactly the cases that separate
// tokenizer implementations.
//
// Output is one line per corpus line: space-separated decimal token ids.
// Encoding runs with add_bos=false and parse_special=true. That is the exact
// counterpart of `tokenizers`' `encode(s, add_special_tokens=False)`: that flag
// only disables the post-processor (the BOS/EOS template), it does NOT stop
// added/special tokens in the *input text* from being matched. Passing
// parse_special=false here instead scores every "<s>"-shaped corpus string as a
// divergence, which is a harness artifact and not a tokenizer difference.
//
// BOS is left off on both sides: its placement is a chat-template concern, and
// folding it in would hide real merge differences behind a constant offset.
//
// scripts/difftok.py drives this and does the comparison.
#include "runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Decode one JSON string literal (including the surrounding quotes) in place.
// Returns the decoded length, or -1 if the line is not a JSON string.
static long unjson(const char *src, char *dst, size_t cap) {
    const char *p = src;
    while (*p == ' ' || *p == '\t') p++;
    if (*p != '"') return -1;
    p++;
    size_t n = 0;
    while (*p && *p != '"') {
        if (n + 4 >= cap) return -1;
        if (*p != '\\') { dst[n++] = *p++; continue; }
        p++;
        switch (*p) {
        case '"':  dst[n++] = '"';  p++; break;
        case '\\': dst[n++] = '\\'; p++; break;
        case '/':  dst[n++] = '/';  p++; break;
        case 'b':  dst[n++] = '\b'; p++; break;
        case 'f':  dst[n++] = '\f'; p++; break;
        case 'n':  dst[n++] = '\n'; p++; break;
        case 'r':  dst[n++] = '\r'; p++; break;
        case 't':  dst[n++] = '\t'; p++; break;
        case 'u': {
            p++;
            unsigned cp = 0;
            for (int i = 0; i < 4; i++) {
                char c = *p++;
                cp <<= 4;
                if (c >= '0' && c <= '9') cp |= (unsigned)(c - '0');
                else if (c >= 'a' && c <= 'f') cp |= (unsigned)(c - 'a' + 10);
                else if (c >= 'A' && c <= 'F') cp |= (unsigned)(c - 'A' + 10);
                else return -1;
            }
            if (cp >= 0xD800 && cp < 0xDC00 && p[0] == '\\' && p[1] == 'u') {
                unsigned lo = 0;
                const char *q = p + 2;
                for (int i = 0; i < 4; i++) {
                    char c = *q++;
                    lo <<= 4;
                    if (c >= '0' && c <= '9') lo |= (unsigned)(c - '0');
                    else if (c >= 'a' && c <= 'f') lo |= (unsigned)(c - 'a' + 10);
                    else if (c >= 'A' && c <= 'F') lo |= (unsigned)(c - 'A' + 10);
                    else return -1;
                }
                if (lo >= 0xDC00 && lo < 0xE000) {
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    p = q;
                }
            }
            if (cp < 0x80) dst[n++] = (char)cp;
            else if (cp < 0x800) {
                dst[n++] = (char)(0xC0 | (cp >> 6));
                dst[n++] = (char)(0x80 | (cp & 0x3F));
            } else if (cp < 0x10000) {
                dst[n++] = (char)(0xE0 | (cp >> 12));
                dst[n++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                dst[n++] = (char)(0x80 | (cp & 0x3F));
            } else {
                dst[n++] = (char)(0xF0 | (cp >> 18));
                dst[n++] = (char)(0x80 | ((cp >> 12) & 0x3F));
                dst[n++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                dst[n++] = (char)(0x80 | (cp & 0x3F));
            }
            break;
        }
        default: return -1;
        }
    }
    if (*p != '"') return -1;
    dst[n] = '\0';
    return (long)n;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <corpus.txt>\n", argv[0]);
        return 2;
    }

    gguf_file g;
    if (!gguf_open(&g, argv[1])) {
        fprintf(stderr, "difftok: cannot open %s\n", argv[1]);
        return 1;
    }
    tokenizer t;
    if (!tokenizer_init(&t, &g)) {
        fprintf(stderr, "difftok: cannot init tokenizer from %s\n", argv[1]);
        return 1;
    }

    FILE *f = fopen(argv[2], "rb");
    if (!f) {
        fprintf(stderr, "difftok: cannot open %s\n", argv[2]);
        return 1;
    }

    // Corpus strings are short by construction; the cap is generous.
    static char line[1 << 16];
    static char text[1 << 16];
    static int32_t ids[1 << 15];

    while (fgets(line, sizeof(line), f)) {
        size_t len = strlen(line);
        while (len && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = '\0';
        if (len == 0) continue;

        if (unjson(line, text, sizeof(text)) < 0) {
            fprintf(stderr, "difftok: bad corpus line: %s\n", line);
            return 1;
        }
        int n = tok_encode(&t, text, ids, (int)(sizeof(ids) / sizeof(*ids)),
                           false, true);
        for (int i = 0; i < n; i++) printf("%s%d", i ? " " : "", ids[i]);
        printf("\n");
    }

    fclose(f);
    tokenizer_free(&t);
    gguf_close(&g);
    return 0;
}
