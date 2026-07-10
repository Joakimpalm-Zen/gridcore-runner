// runner — CLI: one-shot completion and interactive chat.
#include "runner.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// ---------------------------------------------------------------- sampling

typedef struct {
    float temp, top_p, repeat_penalty;
    int top_k;
    uint64_t rng;
    int32_t recent[256];
    int n_recent, recent_head;
} sampler;

static uint64_t rng_next(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return *s = x;
}
static float rng_f32(uint64_t *s) {
    return (rng_next(s) >> 40) / 16777216.0f;
}

typedef struct { float p; int id; } cand_t;
static int cand_cmp(const void *a, const void *b) {
    float d = ((const cand_t *)b)->p - ((const cand_t *)a)->p;
    return d > 0 ? 1 : d < 0 ? -1 : 0;
}

static void sampler_accept(sampler *s, int tok) {
    s->recent[s->recent_head] = tok;
    s->recent_head = (s->recent_head + 1) % 256;
    if (s->n_recent < 256) s->n_recent++;
}

static int sample(sampler *s, float *logits, int n_vocab) {
    if (s->repeat_penalty != 1.0f) {
        for (int i = 0; i < s->n_recent; i++) {
            int tok = s->recent[i];
            if (logits[tok] > 0) logits[tok] /= s->repeat_penalty;
            else                 logits[tok] *= s->repeat_penalty;
        }
    }
    if (s->temp <= 0) {
        int best = 0;
        for (int i = 1; i < n_vocab; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    cand_t *c = malloc(sizeof(cand_t) * n_vocab);
    for (int i = 0; i < n_vocab; i++) c[i] = (cand_t){ logits[i] / s->temp, i };
    qsort(c, n_vocab, sizeof(cand_t), cand_cmp);

    int k = (s->top_k > 0 && s->top_k < n_vocab) ? s->top_k : n_vocab;
    // softmax over top-k
    float mx = c[0].p, sum = 0;
    for (int i = 0; i < k; i++) { c[i].p = expf(c[i].p - mx); sum += c[i].p; }
    for (int i = 0; i < k; i++) c[i].p /= sum;
    // top-p
    if (s->top_p < 1.0f) {
        float cum = 0;
        int cut = k;
        for (int i = 0; i < k; i++) {
            cum += c[i].p;
            if (cum >= s->top_p) { cut = i + 1; break; }
        }
        k = cut;
        cum = 0;
        for (int i = 0; i < k; i++) cum += c[i].p;
        for (int i = 0; i < k; i++) c[i].p /= cum;
    }
    float r = rng_f32(&s->rng), cum = 0;
    int pick = c[k - 1].id;
    for (int i = 0; i < k; i++) {
        cum += c[i].p;
        if (r < cum) { pick = c[i].id; break; }
    }
    free(c);
    return pick;
}

// ---------------------------------------------------------------- chat templates

enum { TMPL_CHATML, TMPL_LLAMA2, TMPL_LLAMA3, TMPL_ZEPHYR, TMPL_RAW };

static int detect_template(const char *tmpl, tokenizer *tok) {
    if (tmpl) {
        if (strstr(tmpl, "<|im_start|>"))       return TMPL_CHATML;
        if (strstr(tmpl, "<|start_header_id|>")) return TMPL_LLAMA3;
        if (strstr(tmpl, "<|user|>"))            return TMPL_ZEPHYR;
        if (strstr(tmpl, "[INST]"))              return TMPL_LLAMA2;
    }
    if (tok_find(tok, "<|im_start|>") >= 0)       return TMPL_CHATML;
    if (tok_find(tok, "<|start_header_id|>") >= 0) return TMPL_LLAMA3;
    if (tok_find(tok, "<|user|>") >= 0)            return TMPL_ZEPHYR;
    return TMPL_LLAMA2;
}

static const char *template_name(int t) {
    switch (t) {
        case TMPL_CHATML: return "chatml";  case TMPL_LLAMA2: return "llama2";
        case TMPL_LLAMA3: return "llama3";  case TMPL_ZEPHYR: return "zephyr";
        default: return "raw";
    }
}

// render one user turn (plus system prompt on the first turn) up to the point
// where the assistant should start writing
static void render_turn(char *buf, size_t cap, int tmpl, const char *sys, const char *msg) {
    switch (tmpl) {
        case TMPL_CHATML:
            if (sys) snprintf(buf, cap,
                "<|im_start|>system\n%s<|im_end|>\n"
                "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", sys, msg);
            else snprintf(buf, cap,
                "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", msg);
            break;
        case TMPL_LLAMA3:
            if (sys) snprintf(buf, cap,
                "<|start_header_id|>system<|end_header_id|>\n\n%s<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n", sys, msg);
            else snprintf(buf, cap,
                "<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n", msg);
            break;
        case TMPL_ZEPHYR:
            if (sys) snprintf(buf, cap,
                "<|system|>\n%s</s>\n<|user|>\n%s</s>\n<|assistant|>\n", sys, msg);
            else snprintf(buf, cap, "<|user|>\n%s</s>\n<|assistant|>\n", msg);
            break;
        case TMPL_LLAMA2:
            if (sys) snprintf(buf, cap,
                "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]", sys, msg);
            else snprintf(buf, cap, "[INST] %s [/INST]", msg);
            break;
        default:
            snprintf(buf, cap, "%s", msg);
    }
}

// ---------------------------------------------------------------- misc

// resolve "-m name[:tag]" through the local Ollama model store
// (~/.ollama/models or $OLLAMA_MODELS): manifests point at GGUF blobs
static char *ollama_resolve(const char *arg) {
    if (strstr(arg, ".gguf") || arg[0] == '/' || arg[0] == '.' || arg[0] == '~')
        return NULL; // clearly a file path
    char base[1024];
    const char *env = getenv("OLLAMA_MODELS");
    if (env) snprintf(base, sizeof(base), "%s", env);
    else {
        const char *home = getenv("HOME");
        if (!home) return NULL;
        snprintf(base, sizeof(base), "%s/.ollama/models", home);
    }

    char name[256];
    snprintf(name, sizeof(name), "%s", arg);
    char *tag = strchr(name, ':');
    if (tag) *tag++ = 0;
    else tag = "latest";
    const char *ns = "library", *mdl = name;
    char *slash = strchr(name, '/');
    if (slash) { *slash = 0; ns = name; mdl = slash + 1; }

    char mpath[1400];
    snprintf(mpath, sizeof(mpath), "%s/manifests/registry.ollama.ai/%s/%s/%s",
             base, ns, mdl, tag);
    FILE *f = fopen(mpath, "rb");
    if (!f) return NULL;
    char manifest[65536];
    size_t n = fread(manifest, 1, sizeof(manifest) - 1, f);
    fclose(f);
    manifest[n] = 0;

    // the model layer's mediaType precedes its digest within the layer object
    char *p = strstr(manifest, "vnd.ollama.image.model");
    if (!p) return NULL;
    p = strstr(p, "sha256:");
    if (!p || (size_t)(p - manifest) + 7 + 64 > n) return NULL;
    char hex[65];
    memcpy(hex, p + 7, 64);
    hex[64] = 0;
    for (int i = 0; i < 64; i++)
        if (!((hex[i] >= '0' && hex[i] <= '9') || (hex[i] >= 'a' && hex[i] <= 'f')))
            return NULL;
    char *blob = malloc(2048);
    snprintf(blob, 2048, "%s/blobs/sha256-%s", base, hex);
    if (access(blob, R_OK) == 0) {
        fprintf(stderr, "resolved ollama model %s -> %s\n", arg, blob);
        return blob;
    }
    free(blob);
    return NULL;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static char *unescape(const char *s) {
    char *out = malloc(strlen(s) + 1);
    size_t m = 0;
    for (size_t i = 0; s[i]; i++) {
        if (s[i] == '\\' && s[i + 1]) {
            i++;
            switch (s[i]) {
                case 'n': out[m++] = '\n'; break;
                case 't': out[m++] = '\t'; break;
                case 'r': out[m++] = '\r'; break;
                case '\\': out[m++] = '\\'; break;
                default: out[m++] = '\\'; out[m++] = s[i]; break;
            }
        } else out[m++] = s[i];
    }
    out[m] = 0;
    return out;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s -m model.gguf [options]\n\n"
        "options:\n"
        "  -m PATH|NAME   GGUF file, or an Ollama model name (e.g. llama3.2:1b)\n"
        "                 resolved from the local Ollama store\n"
        "  -p TEXT        prompt (one-shot completion; \\n etc. are unescaped)\n"
        "  -f FILE        read prompt from file (appended after -p text)\n"
        "  -i             interactive chat mode\n"
        "  -n N           max tokens to generate (default 256, -1 = until EOS)\n"
        "  -c N           context length (default: min(model max, 4096));\n"
        "                 beyond the training context, YaRN rope scaling is\n"
        "                 applied automatically\n"
        "  -b N           prompt batch size (default 64)\n"
        "  -t N           threads (default: min(8, cpus))\n"
        "  -s N           RNG seed (default: time)\n"
        "  --rope-scale F force linear rope position scaling by F\n"
        "  --rope-base F  override rope frequency base\n"
        "  --temp F       temperature (default 0.8, 0 = greedy)\n"
        "  --top-k N      top-k (default 40)\n"
        "  --top-p F      top-p (default 0.95)\n"
        "  --repeat-penalty F  (default 1.1)\n"
        "  --system TEXT  system prompt for chat mode\n"
        "  --chat-template chatml|llama2|llama3|zephyr  (default: auto)\n"
        "  --no-bos       do not add BOS token\n"
        "  --ignore-eos   keep generating past end-of-text tokens\n"
        "  -v             verbose model info\n",
        prog);
}

typedef struct {
    model_t *m;
    tokenizer *tok;
    sampler *smp;
    int pos;           // next free KV slot
    int stop_ids[8];
    int n_stop;
    bool ignore_eos;
    bool hit_stop;     // last generate() ended on a stop token
} engine;

static bool is_stop(engine *e, int id) {
    for (int i = 0; i < e->n_stop; i++) if (e->stop_ids[i] == id) return true;
    return false;
}

// feed tokens through the model in batches (no sampling); returns logits of
// the last token
static float *feed(engine *e, const int32_t *toks, int n) {
    float *logits = NULL;
    model_t *m = e->m;
    for (int i = 0; i < n; ) {
        int chunk = n - i < m->n_batch ? n - i : m->n_batch;
        if (e->pos + chunk > m->n_ctx) return NULL;
        bool last = (i + chunk == n);
        logits = model_forward_batch(m, toks + i, chunk, e->pos, last);
        e->pos += chunk;
        i += chunk;
        if (n > 512 && (i % 512 < m->n_batch || last))
            fprintf(stderr, "\rprompt: %d/%d tokens%s", i, n, last ? "\n" : "");
    }
    return logits;
}

// generate until stop token / limit, streaming to stdout; returns tokens generated
static int generate(engine *e, float *logits, int max_new, double *gen_time) {
    char buf[512];
    int n_gen = 0;
    e->hit_stop = false;
    double t0 = now_s();
    while ((max_new < 0 || n_gen < max_new) && e->pos < e->m->n_ctx) {
        int tok = sample(e->smp, logits, e->m->n_vocab);
        sampler_accept(e->smp, tok);
        if (is_stop(e, tok) && !e->ignore_eos) { e->hit_stop = true; break; }
        int n = tok_decode(e->tok, tok, buf, sizeof(buf));
        fwrite(buf, 1, n, stdout);
        fflush(stdout);
        n_gen++;
        logits = model_forward(e->m, tok, e->pos++);
    }
    *gen_time = now_s() - t0;
    return n_gen;
}

int main(int argc, char **argv) {
    const char *model_path = NULL, *prompt = NULL, *system_prompt = NULL;
    const char *tmpl_arg = NULL, *prompt_file = NULL;
    int n_predict = 256, n_threads = 0, tmpl = -1;
    bool interactive = false, verbose = false, no_bos = false, ignore_eos = false;
    model_params mp = {0};
    sampler smp = { .temp = 0.8f, .top_p = 0.95f, .repeat_penalty = 1.1f,
                    .top_k = 40, .rng = 0 };

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        #define NEXT (i + 1 < argc ? argv[++i] : (usage(argv[0]), exit(1), (char*)0))
        if      (!strcmp(a, "-m")) model_path = NEXT;
        else if (!strcmp(a, "-p")) prompt = NEXT;
        else if (!strcmp(a, "-f")) prompt_file = NEXT;
        else if (!strcmp(a, "-n")) n_predict = atoi(NEXT);
        else if (!strcmp(a, "-c")) mp.n_ctx = atoi(NEXT);
        else if (!strcmp(a, "-b")) mp.n_batch = atoi(NEXT);
        else if (!strcmp(a, "-t")) n_threads = atoi(NEXT);
        else if (!strcmp(a, "-s")) smp.rng = strtoull(NEXT, NULL, 10);
        else if (!strcmp(a, "-i")) interactive = true;
        else if (!strcmp(a, "-v")) verbose = true;
        else if (!strcmp(a, "--temp")) smp.temp = atof(NEXT);
        else if (!strcmp(a, "--top-k")) smp.top_k = atoi(NEXT);
        else if (!strcmp(a, "--top-p")) smp.top_p = atof(NEXT);
        else if (!strcmp(a, "--repeat-penalty")) smp.repeat_penalty = atof(NEXT);
        else if (!strcmp(a, "--rope-scale")) mp.rope_scale = atof(NEXT);
        else if (!strcmp(a, "--rope-base")) mp.rope_base = atof(NEXT);
        else if (!strcmp(a, "--system")) system_prompt = NEXT;
        else if (!strcmp(a, "--chat-template")) tmpl_arg = NEXT;
        else if (!strcmp(a, "--no-bos")) no_bos = true;
        else if (!strcmp(a, "--ignore-eos")) ignore_eos = true;
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown option %s\n", a); usage(argv[0]); return 1; }
    }
    if (!model_path) { usage(argv[0]); return 1; }
    if (prompt_file) {
        FILE *pf = fopen(prompt_file, "rb");
        if (!pf) { fprintf(stderr, "error: cannot open %s\n", prompt_file); return 1; }
        fseek(pf, 0, SEEK_END);
        long fsz = ftell(pf);
        fseek(pf, 0, SEEK_SET);
        char *fbuf = malloc((prompt ? strlen(prompt) : 0) + fsz + 2);
        size_t off = 0;
        if (prompt) { strcpy(fbuf, prompt); off = strlen(prompt); }
        off += fread(fbuf + off, 1, fsz, pf);
        fbuf[off] = 0;
        fclose(pf);
        prompt = fbuf;
    }
    if (!prompt && !interactive) {
        fprintf(stderr, "error: need -p PROMPT or -i\n");
        usage(argv[0]);
        return 1;
    }
    if (smp.rng == 0) smp.rng = (uint64_t)time(NULL) ^ 0x9E3779B97F4A7C15ull;

    if (n_threads <= 0) {
        long nc = sysconf(_SC_NPROCESSORS_ONLN);
        n_threads = nc > 8 ? 8 : (int)nc;
    }
    mp.verbose = verbose;
    mp.n_threads = n_threads;

    f16_init();

    char *resolved = ollama_resolve(model_path);
    if (resolved) model_path = resolved;

    model_t m;
    double t0 = now_s();
    if (!model_load(&m, model_path, &mp)) return 1;

    tokenizer tok;
    if (!tokenizer_init(&tok, &m.gf)) return 1;
    fprintf(stderr, "loaded %s | %s | %d layers | ctx %d | %d threads | %.2fs\n",
            model_path, m.arch, m.n_layer, m.n_ctx, n_threads, now_s() - t0);

    engine e = { .m = &m, .tok = &tok, .smp = &smp, .pos = 0,
                 .ignore_eos = ignore_eos };
    e.stop_ids[e.n_stop++] = tok.eos_id;
    const char *stops[] = { "<|im_end|>", "<|eot_id|>", "<|end_of_text|>",
                            "<|endoftext|>", "</s>" };
    for (size_t i = 0; i < sizeof(stops) / sizeof(*stops); i++) {
        int id = tok_find(&tok, stops[i]);
        if (id >= 0 && !is_stop(&e, id) && e.n_stop < 8) e.stop_ids[e.n_stop++] = id;
    }

    size_t tok_cap = (prompt ? strlen(prompt) : 0) + m.n_ctx + 32;
    int32_t *toks = malloc(sizeof(int32_t) * tok_cap);
    double ptime, gtime;
    int n_prompt, n_gen;

    if (!interactive) {
        // one-shot completion
        char *p = unescape(prompt);
        n_prompt = tok_encode(&tok, p, toks, (int)tok_cap, !no_bos, true);
        if (n_prompt == 0) { fprintf(stderr, "error: empty prompt\n"); return 1; }
        if (n_prompt >= m.n_ctx) {
            fprintf(stderr, "error: prompt is %d tokens but context is %d — "
                    "rerun with -c %d or larger\n", n_prompt, m.n_ctx,
                    (n_prompt + n_predict + 1023) / 1024 * 1024);
            return 1;
        }
        if (verbose) {
            fprintf(stderr, "prompt tokens (%d):", n_prompt);
            for (int i = 0; i < n_prompt && i < 64; i++) fprintf(stderr, " %d", toks[i]);
            fprintf(stderr, "\n");
        }
        t0 = now_s();
        float *logits = feed(&e, toks, n_prompt);
        ptime = now_s() - t0;
        if (!logits) { fprintf(stderr, "error: prompt exceeds context\n"); return 1; }
        for (int i = 0; i < n_prompt; i++) sampler_accept(&smp, toks[i]);

        if (!prompt_file) printf("%s", p); // don't echo huge file prompts
        free(p);
        fflush(stdout);
        n_gen = generate(&e, logits, n_predict, &gtime);
        printf("\n");
        if (e.hit_stop) fprintf(stderr, "[end of text]\n");
        fprintf(stderr, "\nprompt: %d tok, %.2f tok/s | gen: %d tok, %.2f tok/s\n",
                n_prompt, n_prompt / (ptime > 0 ? ptime : 1e-9),
                n_gen, n_gen / (gtime > 0 ? gtime : 1e-9));
        return 0;
    }

    // interactive chat
    if (tmpl_arg) {
        if      (!strcmp(tmpl_arg, "chatml")) tmpl = TMPL_CHATML;
        else if (!strcmp(tmpl_arg, "llama2")) tmpl = TMPL_LLAMA2;
        else if (!strcmp(tmpl_arg, "llama3")) tmpl = TMPL_LLAMA3;
        else if (!strcmp(tmpl_arg, "zephyr")) tmpl = TMPL_ZEPHYR;
        else if (!strcmp(tmpl_arg, "raw"))    tmpl = TMPL_RAW;
    }
    if (tmpl < 0)
        tmpl = detect_template(gguf_get_str(&m.gf, "tokenizer.chat_template", NULL), &tok);
    if (!system_prompt) system_prompt = "You are a helpful assistant.";
    fprintf(stderr, "chat mode (template: %s) — Ctrl-D or /exit to quit\n\n",
            template_name(tmpl));

    char line[8192], rendered[16384];
    bool first_turn = true;
    for (;;) {
        fprintf(stderr, "> ");
        if (!fgets(line, sizeof(line), stdin)) break;
        line[strcspn(line, "\n")] = 0;
        if (!strcmp(line, "/exit") || !strcmp(line, "/quit")) break;
        if (!line[0]) continue;

        render_turn(rendered, sizeof(rendered), tmpl,
                    first_turn ? system_prompt : NULL, line);
        n_prompt = tok_encode(&tok, rendered, toks, (int)tok_cap, first_turn && !no_bos, true);
        first_turn = false;

        if (e.pos + n_prompt >= m.n_ctx) {
            fprintf(stderr, "[context full — restart to continue]\n");
            break;
        }
        t0 = now_s();
        float *logits = feed(&e, toks, n_prompt);
        ptime = now_s() - t0;
        if (!logits) { fprintf(stderr, "[context full]\n"); break; }
        for (int i = 0; i < n_prompt; i++) sampler_accept(&smp, toks[i]);

        n_gen = generate(&e, logits, n_predict, &gtime);
        printf("\n");
        fprintf(stderr, "[%d tok, %.1f tok/s]\n",
                n_gen, n_gen / (gtime > 0 ? gtime : 1e-9));
    }
    return 0;
}
