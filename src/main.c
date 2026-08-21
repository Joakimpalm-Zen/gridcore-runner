// runner — CLI: one-shot completion, interactive chat, and server launcher.
#include "runner.h"
#include "instances.h"
#include "tray.h"

#include "compat.h"
#include "json.h"
#include "envelope.h"
#include "server.h"
// for render_prompt_alloc: chat mode renders the same prompts the chat route
// does, so it uses the same measured-size renderer rather than a second one
#include "api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>

// Bare-invocation tray launch needs to know whether a human is at the other
// end: a piped/redirected probe (CI, scripts, `runner | head`) must get usage
// text, never a GUI event loop.
#ifdef _WIN32
#include <io.h>
#define RUNNER_TTY(f) _isatty(_fileno(f))
#else
#include <unistd.h>
#define RUNNER_TTY(f) isatty(fileno(f))
#endif

// quantize_gguf is declared in runner.h

// ---------------------------------------------------------------- misc

// CLI flags share the one strict parser (compat.c) with the env overrides; the
// only difference is a hard exit on a bad flag versus a warn-and-default on a
// bad env var.
static long long int_arg(const char *opt, const char *s, long long min,
                         long long max) {
    long long v;
    if (!parse_i64(s, min, max, &v)) {
        fprintf(stderr, "error: %s expects an integer in [%lld, %lld]\n",
                opt, min, max);
        exit(1);
    }
    return v;
}

static uint64_t u64_arg(const char *opt, const char *s) {
    uint64_t v;
    if (!parse_u64(s, 0, UINT64_MAX, &v)) {
        fprintf(stderr, "error: %s expects an unsigned integer\n", opt);
        exit(1);
    }
    return v;
}

static double float_arg(const char *opt, const char *s, double min, double max) {
    double v;
    if (!parse_f64(s, min, max, &v)) {
        fprintf(stderr, "error: %s expects a finite number in [%g, %g]\n",
                opt, min, max);
        exit(1);
    }
    return v;
}

// Read a whole file into a NUL-terminated malloc'd buffer, or NULL on any
// error: open, seek, a negative or oversized length (ftell can return -1 and a
// hostile/special file can be enormous), allocation, or a read error. The size
// cap also keeps the later size arithmetic well clear of overflow. *out_len
// gets the byte count (excluding the terminator).
#define MAX_INPUT_FILE (256u * 1024u * 1024u)
static char *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0 || (unsigned long)sz > MAX_INPUT_FILE || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return NULL;
    }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)sz, f);
    bool err = ferror(f) != 0;
    fclose(f);
    if (err) { free(buf); return NULL; }
    buf[got] = 0;
    if (out_len) *out_len = got;
    return buf;
}

// One-shot and interactive CLI state has a shorter lifetime than the process
// in embedding/tests. Keep its teardown explicit instead of relying on exit(2)
// to reclaim it; that also makes the documented sanitizer command a real leak
// gate. Server mode owns and frees the same objects inside server_run().
static void cli_cleanup(engine *e, int32_t *toks, tokenizer *tok, model_t *m) {
    if (e) {
        schema_free((snode *)e->schema);
        if (e->dm) { model_free(e->dm); free(e->dm); }
        free(e->hist);
    }
    free(toks);
    tokenizer_free(tok);
    model_free(m);
    prefix_cache_clear();
}

// --tool-info resolves, from the model's chat template alone, which native
// tool-call protocol family the runtime speaks for this model and whether that
// protocol is native (vs the generic marker/envelope fallback). BOTH are read
// from template.c truth -- the protocol tool_decl_native() selects for the
// family, plus the one family (ornith/qwen3_xml) whose native declaration
// tools_render_for renders instead and which tool_decl_native deliberately
// leaves alone -- never a static arch->family table, which would drift from the
// renderer. `native` is set to whether that family has a native tool protocol.
static const char *tool_family_for(int tmpl, bool *native) {
    tool_envelope env = {0};
    bool skip_generic = false;
    // strict + atem_tool_calling: ask for each family's NATIVE default, so the
    // protocol tool_decl_native selects is the one this model was trained on.
    tool_decl_native(tmpl, true, true, NULL, &env, &skip_generic);
    switch (env.proto) {
        case TP_ATEM:    *native = true; return "atem";
        case TP_HARMONY: *native = true; return "harmony";
        case TP_GEMMA4:  *native = true; return "gemma4";
        default: break;
    }
    if (tmpl == TMPL_ORNITH) { *native = true; return "qwen3_xml"; }
    *native = false;
    return "generic";
}

// One line of chat input, newline stripped, in a buffer that grows to fit it.
// NULL at EOF with nothing read, or on allocation failure (*oom set then).
//
// A fixed buffer here is not a truncation, it is a SPLIT: fgets stops at the
// buffer and the remainder of the same paste arrives as the next line, so
// pasting a file into chat asked two unrelated questions, the second starting
// mid-word. Callers free the result.
static char *read_line(FILE *f, bool *oom) {
    size_t cap = 1024, n = 0;
    char *buf = malloc(cap);
    if (!buf) { *oom = true; return NULL; }
    for (;;) {
        if (!fgets(buf + n, (int)(cap - n), f)) {
            if (n == 0) { free(buf); return NULL; }
            return buf;                      // EOF after a newline-less tail
        }
        n += strlen(buf + n);
        if (n && buf[n - 1] == '\n') { buf[n - 1] = 0; return buf; }
        if (n + 1 < cap) return buf;         // short read: EOF, no newline
        // fgets filled the buffer, so the line continues past it. The cap
        // keeps cap - n inside the int fgets takes; a gigabyte-long single
        // chat line fails loudly rather than overflowing that conversion.
        if (cap > (size_t)INT_MAX / 2) { free(buf); *oom = true; return NULL; }
        char *bigger = realloc(buf, cap * 2);
        if (!bigger) { free(buf); *oom = true; return NULL; }
        buf = bigger;
        cap *= 2;
    }
}

static char *unescape(const char *s) {
    char *out = malloc(strlen(s) + 1);
    if (!out) { fprintf(stderr, "error: out of memory\n"); exit(1); }
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
        "  -m PATH        GGUF model file\n"
        "  -p TEXT        prompt (one-shot completion; \\n etc. are unescaped)\n"
        "  -f FILE        read prompt from file (appended after -p text)\n"
        "  -i             interactive chat mode\n"
        "  --serve        HTTP server mode (OpenAI-compatible API)\n"
        "  --tray         BE the desktop tray/menu-bar controller (macOS,\n"
        "                 Windows) instead of running a model. Required when\n"
        "                 there is no terminal — launchd, Task Scheduler, a\n"
        "                 service wrapper — since the launches below all need\n"
        "                 one. Bare `runner` at a terminal, or a double-click,\n"
        "                 does the same thing without the flag\n"
        "  --no-tray      opt out of the tray entirely. The tray otherwise\n"
        "                 follows any session you sit with (--serve, -i) at a\n"
        "                 terminal and is LEFT RUNNING afterwards, so the next\n"
        "                 model can be loaded from it. One-shot -p runs and\n"
        "                 tooling modes never raise one\n"
        "  --port N       server port (default 8080)\n"
        "  --parallel N   parallel inference slots in server mode (default 1)\n"
        "                 -m \"name=path,name2=path2\" serves multiple models,\n"
        "                 loading the requested one on demand (swap mode)\n"
        "  --ttl N        swap mode: unload an idle model after N seconds\n"
        "                 (default 300, 0 = never)\n"
        "  --force-uncertified  load a model whose measured-envelope sidecar\n"
        "                 records an OUTSIDE-envelope verdict for this runtime\n"
        "                 (refused by default) — loads with a loud warning\n"
        "  --json         constrain output to a single valid JSON object\n"
        "  --json-schema F constrain output to the JSON Schema in file F\n"
        "  --quantize OUT rewrite the model to OUT.gguf (see --quant) and exit\n"
        "  --quant T      quantize target: q8_0 | q4_0 | f16 (default q4_0,\n"
        "                 or unchanged per-tensor if --prune-experts alone)\n"
        "  --type-plan PLAN.json  per-tensor precision while rewriting:\n"
        "                 {\"default\":\"keep\",\"rules\":[{\"match\":\"_exps.weight\",\n"
        "                 \"type\":\"q4_0\"}]}. First matching rule wins; types are\n"
        "                 keep|q8_0|q4_0|q3_k|f16. Per-tensor is the finest GGUF can\n"
        "                 express: experts are stored stacked, one tensor per\n"
        "                 layer, so per-EXPERT precision is not representable\n"
        "  --prune-experts LIST.json  drop MoE experts per layer while\n"
        "                 rewriting: {\"layer_N\":[kept expert ids...]};\n"
        "                 a layer absent from the file keeps every expert.\n"
        "                 Combine with --quant to also requantize survivors\n"
        "  -n N           max tokens to generate (default 256, -1 = until EOS)\n"
        "  -c N           context length (default: min(model max, 4096));\n"
        "                 beyond the training context, YaRN rope scaling is\n"
        "                 applied automatically\n"
        "  -b N           prompt batch size (default 64)\n"
        "  -t N           threads (default: physical cores, capped at 64)\n"
        "  -s N           RNG seed (default: time)\n"
        "  --think        request the model family's thinking prompt shape\n"
        "  --no-think     request its non-thinking shape; with neither flag\n"
        "                 runner renders what the family's own reference\n"
        "                 template renders, which differs per family\n"
        "  --temp F       temperature (0 = greedy: returns the model's argmax\n"
        "                 with no repeat penalty applied)\n"
        "  --top-k N      top-k (0 = off)\n"
        "  --top-p F      top-p\n"
        "  --min-p F      min-p vs the top candidate (0 = off)\n"
        "  --repeat-penalty F  penalty on recently emitted tokens (1 = off)\n"
        "                 the five options above default to the model family's\n"
        "                 published settings; see --caps for the preset table\n"
        "  --rope-scale F force linear rope position scaling by F\n"
        "  --rope-base F  override rope frequency base\n"
        "  --system TEXT  system prompt for chat mode\n"
        "  --chat-template chatml|chatml-think|llama2|llama3|mistral|mistral-v1|\n"
        "                 mistral-nemo|zephyr|phi3|gemma|gemma4|gemma4-mainline|\n"
        "                 apertus|ornith|muse|granite|harmony|raw\n"
        "                 (default: auto). Applies to chat and --serve; not\n"
        "                 valid with a multi-model -m swap set\n"
        "  --no-bos       do not add BOS token\n"
        "  --ignore-eos   keep generating past end-of-text tokens\n"
        "  --gpu auto|off GPU offload if a backend is available (default auto)\n"
        "  --gpu-layers N force N leading layers on the GPU, rest on CPU\n"
        "                 (0 = no GPU, same as --gpu off; omit for auto-fit)\n"
        "  --cpu-moe [N|auto]  keep sparse MoE expert FFNs in RAM while CUDA\n"
        "                 runs attention and other dense tensors on the GPU.\n"
        "                 Bare = every expert layer on the host; N = only the\n"
        "                 deepest N expert layers (the rest stay on the GPU);\n"
        "                 auto = fill the VRAM budget with expert banks first\n"
        "  --wait-for-vram [S]  when another registered runner is holding the\n"
        "                 GPU, queue for up to S seconds (default 300) instead\n"
        "                 of refusing. Without it a runner that does not fit\n"
        "                 fails immediately, naming who holds what.\n"
        "  --vram-priority N  advisory small-integer priority (default 0,\n"
        "                 also RUNNER_VRAM_PRIORITY). Shown in the refusal\n"
        "                 listing; among several --wait-for-vram waiters on\n"
        "                 one GPU, a higher N is admitted first once space\n"
        "                 frees, but only among waiters that currently fit.\n"
        "                 No effect without --wait-for-vram.\n"
        "  --yield-on-request  in --serve, release the resident model at the\n"
        "                 next idle point between requests when another\n"
        "                 process has asked it to (vram_request_yield). A\n"
        "                 request, never preemption: nothing forces this.\n"
        "  --reserve P    cap this runner at P%% of TOTAL VRAM and RAM. With\n"
        "                 -c 0 the context is auto-fit to whatever the\n"
        "                 reservation leaves after the weights\n"
        "  --reserve-vram P / --reserve-ram P   as --reserve, one budget only\n"
        "  --reserve-cpu P  size the thread count as P%% of cores\n"
        "  --kv f16|q8    KV cache storage (default f16). q8 roughly halves the\n"
        "                 cache, so it about doubles the context that fits, but\n"
        "                 it is lossy: output differs from an f16 cache\n"
        "  --mlock        wire the weights into RAM so the OS cannot evict them.\n"
        "                 Off by default and allowed to fail: on a machine with\n"
        "                 little headroom, locking the model can cause the very\n"
        "                 pressure it avoids. Without it a loaded machine can\n"
        "                 page the weights out and every token reads from disk\n"
        "  --moe-prefetch on|off|auto   hand routed experts to the OS as whole\n"
        "                 blocks before the FFN reads them (MoE models only).\n"
        "                 auto (default): on for Apple Silicon when the weights\n"
        "                 exceed available RAM — measured 1.26-1.43x decode\n"
        "                 there — and off elsewhere, where the same A/B\n"
        "                 measured nothing on fast storage\n"
        "  --draft PATH   small same-vocab GGUF for speculative decoding\n"
        "                 (one-shot, chat, and single-model --serve)\n"
        "  --draft-k N    draft tokens per round (default 4)\n"
        "  --bench-json   run a small decode benchmark and print JSON metrics\n"
        "  --score        teacher-forced scoring: per-token log P(token|prefix)\n"
        "                 over the raw -p/-f text (no template, no sampling),\n"
        "                 printed as JSON with NLL and perplexity\n"
        "  --lora FILE    load a LoRA adapter GGUF beside the frozen base\n"
        "                 (CPU dense projections; fails closed otherwise)\n"
        "  --lora-scale F multiply the adapter's trained alpha/r (default 1.0)\n"
        "  --caps         print machine capabilities as JSON and exit\n"
        "  --tool-info    load -m MODEL and print its native tool-call protocol\n"
        "                 as one JSON line, then exit\n"
        "  --fit PATH     estimate whether a GGUF fits this machine and exit;\n"
        "                 reads only the header, so a partial download works\n"
        "  --version      print the runner version and exit\n"
        "  --parent-pid N exit when process N dies (supervisor cleanup)\n"
        "  -v             verbose model info\n",
        prog);
}

static int stdout_cb(void *ud, const char *bytes, int n) {
    (void)ud;
    fwrite(bytes, 1, n, stdout);
    fflush(stdout);
    return 0;
}

static int discard_cb(void *ud, const char *bytes, int n) {
    (void)ud; (void)bytes; (void)n;
    return 0;
}

// chat-mode display: thinking channels are shown between [thinking] markers
// instead of leaking the architecture's raw thinking tags
typedef struct {
    think_split ts;
    bool in_think;
} chat_out;

static int chat_emit(void *ud, int reasoning, const char *bytes, int n) {
    chat_out *c = ud;
    if (reasoning && !c->in_think)  { fputs("[thinking]\n", stdout);   c->in_think = true; }
    if (!reasoning && c->in_think)  { fputs("\n[/thinking]\n", stdout); c->in_think = false; }
    fwrite(bytes, 1, n, stdout);
    fflush(stdout);
    return 0;
}

static int chat_cb(void *ud, const char *bytes, int n) {
    chat_out *c = ud;
    return think_feed(&c->ts, bytes, n, chat_emit, c);
}

// --fit: what a GGUF would cost on this machine, from its HEADER alone.
//
// The point is to answer "should I download this" before spending the
// bandwidth, so it reads through gguf_open_header() and works on a file that
// is only a few megabytes of header. It does NOT fetch anything: the runner
// does not speak HTTP and a fit check is not a reason to teach it. The README
// documents the ranged-read one-liner that turns a URL into a header file.
static void fit_gib(char *buf, size_t n, uint64_t bytes) {
    snprintf(buf, n, "%.2f GiB", (double)bytes / (double)(1ull << 30));
}

static int run_fit_check(const char *path, int n_ctx_want) {
    gguf_file g;
    if (!gguf_open_header(&g, path)) return 1;   // already reported why

    model_fit f;
    if (!model_fit_report(&g, n_ctx_want, &f)) {
        fprintf(stderr, "error: %s does not carry the geometry a fit check "
                        "needs (architecture, block_count, head counts)\n", path);
        gguf_close(&g);
        return 1;
    }

    char w[32], h[32], kf[32], kq[32], av[32];
    fit_gib(w, sizeof w, f.weights);
    fit_gib(h, sizeof h, f.hot);
    fit_gib(kf, sizeof kf, f.kv_f16);
    fit_gib(kq, sizeof kq, f.kv_q8);
    fit_gib(av, sizeof av, f.available);

    uint32_t split = gguf_get_u32(&g, "split.count", 0);
    printf("fit: %s\n", path);
    printf("  model         %s, %d layers", f.arch, f.n_layer);
    if (f.sparse) printf(", MoE %d experts, %d used", f.n_expert, f.n_expert_used);
    printf("\n");
    if (split > 1)
        printf("  NOTE          this is part 1 of a %u-part split; the sizes "
               "below are THIS PART only\n", split);
    printf("  weights       %s\n", w);
    if (f.sparse)
        printf("  hot set       %s  (only the routed experts a token actually "
               "uses)\n", h);
    printf("  kv cache      %s at ctx %d, f16", kf, f.n_ctx);
    if (f.kv_q8) printf("   |  %s with --kv q8", kq);
    else         printf("   |  --kv q8 unavailable (head_dim is not a multiple of 32)");
    printf("\n");
    if (f.available) printf("  available RAM %s right now\n", av);
    else             printf("  available RAM unknown on this platform\n");

    const char *v = model_fit_verdict(&f);
    printf("  verdict       %s", v);
    if (!f.available) {
        printf("\n                available RAM could not be read, so this is "
               "sizes only\n");
    } else if (!strcmp(v, "FITS")) {
        char slack[32];
        fit_gib(slack, sizeof slack, f.available - f.hot - f.kv_f16);
        printf(" — %s to spare at ctx %d\n", slack, f.n_ctx);
    } else if (!strcmp(v, "FITS WITH --kv q8")) {
        char over[32];
        fit_gib(over, sizeof over, f.hot + f.kv_f16 - f.available);
        printf(" — an f16 cache is %s over; q8 halves it\n", over);
    } else {
        char over[32];
        fit_gib(over, sizeof over, f.hot + (f.kv_q8 ? f.kv_q8 : f.kv_f16)
                                   - f.available);
        printf(" — %s over even with the smallest cache. Every token would "
               "page from disk.\n", over);
        printf("                fixes: a smaller quantization, a shorter -c, "
               "or --gpu-layers to split it\n");
    }
    if (f.sparse)
        printf("  note          the verdict uses the hot set; a sparse MoE runs "
               "usefully while the file exceeds RAM\n");
    printf("  note          KV is an upper bound: models with per-layer KV "
           "geometry (shared KV, MLA) use less\n");

    gguf_close(&g);
    return 0;
}

int main(int argc, char **argv) {
    const char *model_path = NULL, *prompt = NULL, *system_prompt = NULL;
    char *owned_prompt = NULL;
    const char *tmpl_arg = NULL, *prompt_file = NULL, *schema_file = NULL;
    const char *quant_out = NULL, *quant_type = NULL, *prune_experts = NULL;
    const char *type_plan = NULL;
    int n_predict = 256, n_threads = 0, tmpl = -1, reserve_cpu_pct = 0;
    int port = 8080, parallel = 1, ttl = -1; // -1: 300 for swap mode, never for single
    long parent_pid = 0;
    const char *draft_path = NULL;
    int draft_k = 4;
    bool interactive = false, verbose = false, no_bos = false;
    bool seed_given = false;
    bool ignore_eos = false, json_mode = false, serve = false, caps = false;
    bool tool_info = false;
    const char *fit_path = NULL;
    bool no_tray = false;
    bool force_uncertified = false;
    int thinking = THINK_DEFAULT;
    bool bench_json = false;
    bool score = false;
    const char *lora_path = NULL;
    float lora_scale = 1.0f;
    model_params mp = {0};
    // RUNNER_VRAM_PRIORITY sets the baseline; --vram-priority (parsed below)
    // overrides it, same precedence as every other env/flag pair in runner.
    mp.vram_priority = (int)env_i64("RUNNER_VRAM_PRIORITY", -1000, 1000, 0);
    // The four sampling knobs are filled in from the model's family preset
    // once the model is known; anything named on the command line is recorded
    // here and wins over the preset.
    sampler smp = { .rng = 0 };
    sampler_override ov = {0};

    // A bare `runner` — double-clicked, or typed with no arguments — starts
    // the tray on the platforms that have one. The conditions are deliberately
    // narrow: literally zero arguments (so no scripted invocation can ever be
    // affected — scripts always pass something), a real terminal on both
    // stdin and stdout (so pipes and CI probes still get usage text), and a
    // platform with a tray backend (Linux keeps printing usage). --no-tray is
    // the standing opt-out: it suppresses this launch and nothing else, for
    // wrappers that want bare-invocation behavior guaranteed text-only.
#if defined(__APPLE__) || defined(_WIN32)
    if (argc == 1 && RUNNER_TTY(stdin) && RUNNER_TTY(stdout))
        return tray_main();
#endif

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        #define NEXT (i + 1 < argc ? argv[++i] : (usage(argv[0]), exit(1), (char*)0))
        if      (!strcmp(a, "-m")) model_path = NEXT;
        else if (!strcmp(a, "-p")) prompt = NEXT;
        else if (!strcmp(a, "-f")) prompt_file = NEXT;
        else if (!strcmp(a, "-n")) n_predict = (int)int_arg(a, NEXT, -1, INT_MAX);
        else if (!strcmp(a, "-c")) mp.n_ctx = (int)int_arg(a, NEXT, 0, INT_MAX);
        else if (!strcmp(a, "-b")) mp.n_batch = (int)int_arg(a, NEXT, 0, INT_MAX);
        else if (!strcmp(a, "-t")) n_threads = (int)int_arg(a, NEXT, 0, INT_MAX);
        else if (!strcmp(a, "-s")) { smp.rng = u64_arg(a, NEXT); seed_given = true; }
        else if (!strcmp(a, "-i")) interactive = true;
        else if (!strcmp(a, "-v")) verbose = true;
        else if (!strcmp(a, "--serve")) serve = true;
        // gemma-4 renders a non-thinking prompt by default, matching its
        // reference template; --think opts into the thinking shape
        else if (!strcmp(a, "--think")) thinking = THINK_ON;
        else if (!strcmp(a, "--no-think")) thinking = THINK_OFF;
        else if (!strcmp(a, "--tray")) return tray_main();
        // The standing opt-out, and no longer a no-op: it suppresses both the
        // bare-invocation launch above and the tray that otherwise follows a
        // --serve or -i session below.
        else if (!strcmp(a, "--no-tray")) no_tray = true;
        else if (!strcmp(a, "--force-uncertified")) force_uncertified = true;
        else if (!strcmp(a, "--port")) port = (int)int_arg(a, NEXT, 1, 65535);
        else if (!strcmp(a, "--parallel")) parallel = (int)int_arg(a, NEXT, 1, 16);
        else if (!strcmp(a, "--ttl")) ttl = (int)int_arg(a, NEXT, -1, INT_MAX);
        else if (!strcmp(a, "--json")) json_mode = true;
        else if (!strcmp(a, "--json-schema")) schema_file = NEXT;
        else if (!strcmp(a, "--quantize")) quant_out = NEXT;
        else if (!strcmp(a, "--quant")) quant_type = NEXT;
        else if (!strcmp(a, "--prune-experts")) prune_experts = NEXT;
        else if (!strcmp(a, "--type-plan")) type_plan = NEXT;
        else if (!strcmp(a, "--temp")) { ov.temp = (float)float_arg(a, NEXT, 0, FLT_MAX); ov.has_temp = true; }
        else if (!strcmp(a, "--top-k")) { ov.top_k = (int)int_arg(a, NEXT, 0, INT_MAX); ov.has_top_k = true; }
        else if (!strcmp(a, "--top-p")) { ov.top_p = (float)float_arg(a, NEXT, 0, 1); ov.has_top_p = true; }
        else if (!strcmp(a, "--min-p")) { ov.min_p = (float)float_arg(a, NEXT, 0, 1); ov.has_min_p = true; }
        else if (!strcmp(a, "--repeat-penalty")) { ov.repeat_penalty = (float)float_arg(a, NEXT, FLT_MIN, FLT_MAX); ov.has_repeat_penalty = true; }
        else if (!strcmp(a, "--rope-scale")) mp.rope_scale = (float)float_arg(a, NEXT, 0, FLT_MAX);
        else if (!strcmp(a, "--rope-base")) mp.rope_base = (float)float_arg(a, NEXT, 0, FLT_MAX);
        else if (!strcmp(a, "--system")) system_prompt = NEXT;
        else if (!strcmp(a, "--chat-template")) tmpl_arg = NEXT;
        else if (!strcmp(a, "--no-bos")) no_bos = true;
        else if (!strcmp(a, "--ignore-eos")) ignore_eos = true;
        else if (!strcmp(a, "--gpu")) {
            const char *v = NEXT;
            if (!strcmp(v, "auto")) mp.gpu_mode = GPU_AUTO;
            else if (!strcmp(v, "off")) mp.gpu_mode = GPU_OFF;
            else { fprintf(stderr, "error: --gpu expects auto or off\n"); return 1; }
        }
        else if (!strcmp(a, "--kv")) {
            const char *v = NEXT;
            if (!strcmp(v, "q8")) mp.kv_q8 = true;
            else if (!strcmp(v, "f16")) mp.kv_q8 = false;
            else { fprintf(stderr, "error: --kv expects f16 or q8\n"); return 1; }
        }
        else if (!strcmp(a, "--mlock")) mp.mlock = true;
        else if (!strcmp(a, "--moe-prefetch")) {
            const char *v = NEXT;
            if (!strcmp(v, "on")) mp.moe_prefetch = 1;
            else if (!strcmp(v, "off")) mp.moe_prefetch = -1;
            else if (!strcmp(v, "auto")) mp.moe_prefetch = 0;
            else { fprintf(stderr, "error: --moe-prefetch expects on, off or auto\n"); return 1; }
        }
        else if (!strcmp(a, "--draft"))   draft_path = NEXT;
        else if (!strcmp(a, "--draft-k")) draft_k = (int)int_arg(a, NEXT, 1, 15);
        else if (!strcmp(a, "--bench-json")) bench_json = true;
        else if (!strcmp(a, "--score")) score = true;
        else if (!strcmp(a, "--lora")) lora_path = NEXT;
        else if (!strcmp(a, "--lora-scale"))
            lora_scale = (float)float_arg(a, NEXT, 0, FLT_MAX);
        else if (!strcmp(a, "--reserve")) mp.reserve_vram_pct = mp.reserve_ram_pct = (int)int_arg(a, NEXT, 0, 100);
        else if (!strcmp(a, "--reserve-vram")) mp.reserve_vram_pct = (int)int_arg(a, NEXT, 0, 100);
        else if (!strcmp(a, "--gpu-layers")) {
            // An explicit 0 means what it says: no layers on the GPU. It used
            // to be the "auto-fit" sentinel, which read as CPU but silently
            // ran full-GPU — that exact misreading produced vacuous
            // CPU-vs-GPU certification evidence (GPU compared with GPU).
            // Auto-fit is the default; omit the flag to get it.
            int n = (int)int_arg(a, NEXT, 0, 100000);
            if (n == 0) mp.gpu_mode = GPU_OFF;
            else mp.gpu_layers_override = n;
        }
        else if (!strcmp(a, "--cpu-moe")) {
            // Optional argument: a layer count or "auto". A bare --cpu-moe
            // keeps every expert FFN on the host (the original behaviour), so
            // the argument is only consumed when it is unmistakably ours —
            // every other CLI token starts with '-' or is a flag's operand.
            mp.cpu_moe = true;
            mp.cpu_moe_layers = CPU_MOE_ALL;
            const char *nx = i + 1 < argc ? argv[i + 1] : NULL;
            if (nx && !strcmp(nx, "auto")) {
                mp.cpu_moe_layers = CPU_MOE_AUTO;
                i++;
            } else if (nx && nx[0] >= '0' && nx[0] <= '9') {
                mp.cpu_moe_layers = (int)int_arg(a, argv[++i], 0, INT_MAX);
            }
        }
        else if (!strcmp(a, "--reserve-ram")) mp.reserve_ram_pct = (int)int_arg(a, NEXT, 0, 100);
        else if (!strcmp(a, "--reserve-cpu")) reserve_cpu_pct = (int)int_arg(a, NEXT, 0, 100);
        else if (!strcmp(a, "--wait-for-vram")) {
            // optional argument: a bare --wait-for-vram waits the default, and
            // anything that is not a number is left for the next iteration to
            // parse as its own flag
            mp.vram_wait_secs = 300;
            if (i + 1 < argc && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9')
                mp.vram_wait_secs = (int)int_arg(a, argv[++i], 1, 86400);
        }
        else if (!strcmp(a, "--vram-priority"))
            mp.vram_priority = (int)int_arg(a, NEXT, -1000, 1000);
        else if (!strcmp(a, "--yield-on-request")) mp.yield_on_request = true;
        else if (!strcmp(a, "--parent-pid")) parent_pid = (long)int_arg(a, NEXT, 1, LONG_MAX);
        else if (!strcmp(a, "--caps")) caps = true;
        // Per-MODEL evidence: the native tool-call protocol resolved from this
        // model's chat template (a build property --caps cannot carry). Loads
        // the model, prints one JSON line, exits.
        else if (!strcmp(a, "--tool-info")) tool_info = true;
        else if (!strcmp(a, "--fit")) fit_path = NEXT;
        else if (!strcmp(a, "--version")) { printf("runner %s\n", RUNNER_VERSION); return 0; }
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown option %s\n", a); usage(argv[0]); return 1; }
    }
    plat_parent_watch(parent_pid);
    if (n_threads <= 0 && reserve_cpu_pct > 0) {
        n_threads = plat_cpu_count() * reserve_cpu_pct / 100;
        if (n_threads < 1) n_threads = 1;
    }
    if (fit_path) return run_fit_check(fit_path, mp.n_ctx);

    if (caps) {
        char gname[128];
        bool has_gpu = gpu_available(gname, sizeof(gname));
        printf("{\"version\":\"%s\",\"os\":\"%s\",\"arch\":\"%s\",\"cpu_cores\":%d,"
               "\"ram_bytes\":%llu,\"ram_available_bytes\":%llu,\"gpu\":",
               RUNNER_VERSION,
#if defined(_WIN32)
               "windows",
#elif defined(__APPLE__)
               "macos",
#else
               "linux",
#endif
#if defined(__aarch64__) || defined(__arm64__)
               "arm64",
#elif defined(__x86_64__) || defined(_M_X64)
               "x86_64",
#else
               "other",
#endif
               plat_cpu_count(), (unsigned long long)plat_ram_bytes(),
               // Not total RAM: "model file size <= RAM" is the test that
               // passes right before a machine starts thrashing. A launcher
               // needs to know what is actually free plus reclaimable, so it
               // can refuse a model that will not stay resident.
               (unsigned long long)plat_ram_available_bytes());
        if (has_gpu) {
            size_t vfree = 0, vtotal = 0;
            bool vm = gpu_mem_info(&vfree, &vtotal);
#ifdef __APPLE__
            printf("{\"backend\":\"metal\",\"name\":\"%s\",\"unified_memory\":true",
                   gname);
#else
            printf("{\"backend\":\"cuda\",\"name\":\"%s\",\"unified_memory\":false",
                   gname);
            printf(",\"min_compute_capability\":\"%s\",\"min_gpu\":\"%s\","
                   "\"ptx_target\":\"%s\"",
                   RUNNER_CUDA_MIN_CC, RUNNER_CUDA_MIN_GPU,
                   RUNNER_CUDA_PTX_TARGET);
#endif
            if (vm)
                printf(",\"vram_bytes\":%llu,\"vram_free_bytes\":%llu",
                       (unsigned long long)vtotal, (unsigned long long)vfree);
            // Read this alongside gpu_quants. "This backend has a sparse-MoE
            // path" and "this exact MoE variant's router/layout/activation is
            // implemented" are different claims; gpu_init() still guards the
            // latter at load.
            printf(",\"moe\":%s", gpu_moe_ok() ? "true" : "false");
            printf(",\"eseries\":%s", gpu_eseries_ok() ? "true" : "false");
            printf(",\"kv_q8\":%s", gpu_kv_q8_ok() ? "true" : "false");
            // Metal only: a single allocation above this fails even when the
            // file fits in RAM, so ram_available_bytes alone overstates what
            // the GPU path can load. Same argument as moe/kv_q8: don't let a
            // scheduler read capability across a capacity cliff.
            // Proves the binary carries the kernels currently in the tree.
            // make test-shader-embed compares this to a fresh hash of
            // src/kernels.metal.
            const char *ssha = gpu_shader_source_sha();
            if (ssha) printf(",\"shader_source_sha256\":\"%s\"", ssha);
            size_t mws = 0;
            if (gpu_max_working_set(&mws))
                printf(",\"max_working_set_bytes\":%llu",
                       (unsigned long long)mws);
            printf("}");
        } else
            printf("null");
        // the preset table is a property of the build, not of any one model,
        // so --caps can publish it without loading weights
        printf(",\"sampling_presets\":[");
        for (int i = 0; ; i++) {
            const sampler_preset *sp = sampler_preset_at(i);
            if (!sp) break;
            char esc_src[512];
            json_escape(sp->source, strlen(sp->source), esc_src, sizeof(esc_src));
            printf("%s{\"name\":\"%s\",\"temperature\":%.2f,\"top_p\":%.2f,"
                   "\"top_k\":%d,\"min_p\":%.2f,\"repeat_penalty\":%.2f,"
                   "\"source\":\"%s\"}",
                   i ? "," : "", sp->name, (double)sp->temp, (double)sp->top_p,
                   sp->top_k, (double)sp->min_p, (double)sp->repeat_penalty,
                   esc_src);
        }
        printf("]");
        // KV cache formats this build can store. Both are available on every
        // backend (CPU and CUDA); q8 needs head_dim % 32 == 0, which is a
        // per-model property and so is reported at load, not here. f16 is the
        // default because q8 is lossy — it does not reproduce f16 output.
        printf(",\"kv_types\":[\"f16\",\"q8\"],\"kv_type_default\":\"f16\"");
        // Placement modes controllers may safely request. cpu_moe means the
        // CUDA backend can keep dense/attention tensors resident while sparse
        // expert FFNs execute from system RAM.
        // cpu_moe_partial advertises the optional layer count / auto fit, so a
        // controller can size the expert split instead of guessing all-or-none.
        printf(",\"tensor_placement\":{\"cpu_moe\":true,"
               "\"cpu_moe_partial\":true}");
        // the fixed -m swap-registry capacity, so a controller bounds the set of
        // models it launches instead of overflowing it (see RUNNER_MAX_MODELS)
        printf(",\"max_models\":%d", RUNNER_MAX_MODELS);
        // One table, two consumers: `quants` is every format the engine can
        // read, and `gpu_quants` is that same table filtered by the active
        // backend's own admission test (gpu_quant_ok -> gpu_type_ok). These
        // were two hand-kept literals under an #ifdef, each carrying a comment
        // asking a human to keep it in step with a switch in another file.
        // They drifted, and --caps advertised Metal kernels that did not
        // exist. Sourcing the answer removes the class of bug, not just the
        // instance: a new format needs one edit, in the backend, and both
        // lists follow.
        static const struct { const char *name; int type; } QUANTS[] = {
            { "F32",    T_F32    }, { "F16",    T_F16    },
            { "BF16",   T_BF16   }, { "Q8_0",   T_Q8_0   },
            { "Q4_0",   T_Q4_0   }, { "Q4_1",   T_Q4_1   },
            { "Q5_0",   T_Q5_0   }, { "Q5_1",   T_Q5_1   },
            { "Q2_K",   T_Q2_K   }, { "Q3_K",   T_Q3_K   },
            { "Q4_K",   T_Q4_K   }, { "Q5_K",   T_Q5_K   },
            { "Q6_K",   T_Q6_K   }, { "IQ4_NL", T_IQ4_NL },
            { "IQ4_XS", T_IQ4_XS }, { "MXFP4",  T_MXFP4  },
            { "IQ1_S",  T_IQ1_S  }, { "IQ1_M",  T_IQ1_M  },
            { "IQ2_XXS", T_IQ2_XXS }, { "IQ2_XS", T_IQ2_XS },
            { "IQ2_S",  T_IQ2_S  },
            { "IQ3_XXS", T_IQ3_XXS }, { "IQ3_S", T_IQ3_S },
        };
        const size_t n_quants = sizeof QUANTS / sizeof *QUANTS;
        printf(",\"quants\":[");
        for (size_t qi = 0; qi < n_quants; qi++)
            printf("%s\"%s\"", qi ? "," : "", QUANTS[qi].name);
        // gpu_init() still refuses unsupported architectures and layouts; this
        // is the tensor-format half only, which is what an advisor filters on.
        printf("],\"gpu_quants\":[");
        for (size_t qi = 0, shown_q = 0; qi < n_quants; qi++)
            if (gpu_quant_ok(QUANTS[qi].type))
                printf("%s\"%s\"", shown_q++ ? "," : "", QUANTS[qi].name);
        printf("]");
        // the architectures model_load will admit, so a catalog/advisor can
        // filter unrunnable models by arch without shipping its own copy of the
        // allowlist (single source of truth: model_supported_archs)
        printf(",\"architectures\":[");
        size_t n_arch;
        const char *const *arches = model_supported_archs(&n_arch);
        for (size_t i = 0; i < n_arch; i++)
            printf("%s\"%s\"", i ? "," : "", arches[i]);
        printf("]}\n");
        return 0;
    }
    if (!model_path) { usage(argv[0]); return 1; }
    if (prompt_file) {
        size_t flen = 0;
        char *fdata = read_file(prompt_file, &flen);
        if (!fdata) { fprintf(stderr, "error: cannot read %s\n", prompt_file); return 1; }
        size_t plen = prompt ? strlen(prompt) : 0;
        if (flen > SIZE_MAX - plen - 1) {   // checked concat length
            fprintf(stderr, "error: prompt is too large\n");
            free(fdata);
            return 1;
        }
        char *fbuf = malloc(plen + flen + 1);
        if (!fbuf) { free(fdata); fprintf(stderr, "error: out of memory\n"); return 1; }
        if (prompt) memcpy(fbuf, prompt, plen);
        memcpy(fbuf + plen, fdata, flen);
        fbuf[plen + flen] = 0;
        free(fdata);
        prompt = owned_prompt = fbuf;
    }
    if (!prompt && !interactive && !serve && !quant_out && !bench_json &&
        !tool_info) {
        fprintf(stderr, "error: need -p PROMPT, -i, or --serve\n");
        usage(argv[0]);
        return 1;
    }
    // The tray follows a session a person will sit with — a server or an
    // interactive chat — and is left running afterwards so the next model can
    // be loaded from it. One-shot -p runs and tooling modes (--caps,
    // --quantize, --bench-json) deliberately raise nothing: a two-second
    // process should not leave a menu-bar icon behind it.
    //
    // A terminal on EITHER stdin or stdout counts as "a person launched this".
    // Requiring both, the way the bare-invocation guard above does, would drop
    // the most ordinary case there is — `runner --serve > server.log` — while
    // CI and pipes typically have neither and stay unaffected. --no-tray is the
    // explicit escape for a wrapper that wants the guarantee in writing.
#if defined(__APPLE__) || defined(_WIN32)
    if (!no_tray && (serve || interactive) &&
        (RUNNER_TTY(stdin) || RUNNER_TTY(stdout)))
        tray_ensure_running();
#else
    (void)no_tray;
#endif

    // The sampler's xorshift64 has a fixed point at state 0: it stays 0 and
    // every draw comes back 0.0. So 0 is not a usable seed — but the old
    // `if (rng == 0) rng = time(...)` said that by silently substituting the
    // clock, which turned `-s 0`, a request for a REPRODUCIBLE run, into a
    // random one with nothing on stderr. Only an absent -s takes the clock
    // now; an explicit 0 is refused, because the one thing that must not
    // happen is running it as if it had been honoured.
    if (seed_given && smp.rng == 0) {
        fprintf(stderr, "error: -s 0 is not a usable seed — the sampler's RNG "
                        "state 0 is a fixed point. Use any other value.\n");
        return 1;
    }
    if (!seed_given) smp.rng = (uint64_t)time(NULL) ^ 0x9E3779B97F4A7C15ull;

    if (n_threads <= 0) {
        // Default to a physical-core proxy (nc/2 under the near-universal 2-way
        // SMT of the many-core x86 boxes this targets), capped so a very large
        // box does not over-spawn. SMT siblings add nothing to a compute-bound
        // decode — measured: throughput plateaus at physical cores and SMT gives
        // no gain. The old min(8,nc) default left big boxes badly underused
        // (measured 32.2s -> 9.9s, a 3.2x speedup, at -t 64 vs -t 8 on a 3B Q4,
        // 64-core box; token-identical across thread counts). Scope a shared box
        // with --reserve-cpu, or pin exactly with -t. On Apple Silicon the
        // platform layer returns the performance-core count instead of the
        // logical/2 proxy, because asymmetric cores are not SMT siblings.
        n_threads = plat_default_thread_count();
        // If the model turns out to be CPU-forced (recurrent qwen3.5 path),
        // model_load may raise a defaulted count to this cap; a pinned -t
        // or CPU reservation leaves it 0 and is never overridden.
        int cap = plat_default_thread_count();
        mp.cpu_fallback_threads = cap > n_threads ? cap : n_threads;
    }
    mp.verbose = verbose;
    mp.n_threads = serve ? 1 : n_threads; // server slots create their own pools

    f16_init();

    // `-m name=path` with --parallel > 1 is one named model, not a swap set.
    // Swap mode is genuinely single-slot (ensure_resident loads slots[0]
    // alone), but with one entry there is nothing to swap to, so the caller
    // keeps their slots and gives up /unload and --ttl instead. server_run
    // still receives the name=path form and keeps the name for /v1/models;
    // what has to happen HERE is the preload, which needs the bare path.
    const char *eq_one = strchr(model_path, '=');
    bool one_named = serve && parallel > 1 && eq_one && eq_one != model_path &&
                     eq_one[1] && !strchr(model_path, ',');
    const char *load_path = one_named ? eq_one + 1 : model_path;
    bool registry = serve && eq_one != NULL && !one_named;

    // --chat-template resolves HERE, not down in the interactive-chat block.
    // It used to resolve only there, and server mode returns above it, so a
    // documented public flag was accepted on the command line and silently
    // discarded on the path most callers take: forcing a template under
    // --serve produced a byte-identical prompt to auto-detection.
    int tmpl_override = -1;
    if (tmpl_arg) {
        tmpl_override = template_from_name(tmpl_arg);
        if (tmpl_override < 0) {
            // previously this fell back to auto-detection without a word,
            // which is the same discard in a smaller place
            fprintf(stderr, "error: unknown --chat-template '%s'; see --help "
                            "for the accepted names\n", tmpl_arg);
            return 1;
        }
        // A multi-model swap set cannot honour one global template; server_run
        // refuses that combination, where the parsed entry count is known.
    }

    if (quant_out) {
        int tt;
        if (quant_type) {
            tt = !strcmp(quant_type, "q8_0") ? T_Q8_0 :
                 !strcmp(quant_type, "q4_0") ? T_Q4_0 :
                 !strcmp(quant_type, "f16")  ? T_F16 : INT_MIN;
            if (tt == INT_MIN) {
                fprintf(stderr, "error: --quant must be q8_0, q4_0, or f16\n");
                return 1;
            }
        } else {
            // No --quant given: --prune-experts alone means "just prune,
            // don't touch anyone's precision" (T_KEEP); with neither flag
            // this is plain --quantize, whose long-standing default is q4_0.
            tt = prune_experts ? T_KEEP : T_Q4_0;
        }
        return quantize_gguf_plan(model_path, quant_out, tt, prune_experts,
                                  type_plan);
    }
    // Every flag that only has meaning while rewriting a file. Reaching here
    // means --quantize was not given, and the block above is their only
    // reader, so any of them is an instruction that would otherwise be
    // silently dropped — the same discard --chat-template was fixed for, in a
    // place where the user believes they asked for a different file on disk.
    const char *orphan = prune_experts ? "--prune-experts"
                       : type_plan     ? "--type-plan"
                       : quant_type    ? "--quant"
                       : NULL;
    if (orphan) {
        fprintf(stderr, "error: %s requires --quantize OUT\n", orphan);
        return 1;
    }

    model_t m;
    tokenizer tok;
    // A reservation is a budget for the whole server, so the -c 0 auto-fit has
    // to know how many slots will divide it. Set before the FIRST load, not
    // just for the slots server_run creates: slot 0 is this model, and a slot 0
    // that sized itself alone would both over-commit and — because the CUDA
    // shared-weight registry keys on context — disagree with its peers and
    // force a second upload of the same weights.
    if (serve) mp.n_seq = parallel;
    if (!registry) {
        double t1 = now_s();
        if (!model_load(&m, load_path, &mp)) return 1;
        if (lora_path && !model_lora_load(&m, lora_path, lora_scale))
            return 1;
        if (!tokenizer_init(&tok, &m.gf)) return 1;
        fprintf(stderr, "loaded %s | %s | %d layers | ctx %d | %d threads | %.2fs\n",
                load_path, m.arch, m.n_layer, m.n_ctx, tpool_size(m.tp),
                now_s() - t1);
        // --tool-info: one JSON line naming this model's native tool-call
        // protocol, resolved from its chat template exactly as the chat surface
        // would (honouring --chat-template). stdout is JSON-only.
        if (tool_info) {
            int ti_tmpl = tmpl_override >= 0 ? tmpl_override
                        : template_detect(gguf_get_str(&m.gf,
                                          "tokenizer.chat_template", NULL), &tok);
            bool native = false;
            const char *fam = tool_family_for(ti_tmpl, &native);
            printf("{\"tool_family\":\"%s\",\"native_tool_protocol\":%s}\n",
                   fam, native ? "true" : "false");
            cli_cleanup(NULL, NULL, &tok, &m);
            return 0;
        }
        // Measured-envelope sidecar, if one sits next to the model: the
        // load-time three-state gate. Certified/experimental print a banner and
        // load; an OUTSIDE-envelope verdict REFUSES the load with the measured
        // reason unless --force-uncertified is given. Silent when there is no
        // manifest.
        {
            // Resolve against the kernels this model actually loaded. A GPU
            // may be available while --gpu off (or a model-specific fallback)
            // leaves m.gpu NULL; classifying that CPU run as Metal/CUDA would
            // apply a verdict measured for a backend that did no work here.
#ifdef __APPLE__
            const char *env_backend = m.gpu ? "metal" : "cpu";
#else
            const char *env_backend = m.gpu ? "cuda" : "cpu";
#endif
            char env_line[256];
            bool ok = envelope_gate(load_path, RUNNER_VERSION, env_backend,
                                    force_uncertified, env_line,
                                    (int)sizeof env_line, NULL);
            if (env_line[0]) fprintf(stderr, "%s\n", env_line);
            if (!ok) {
                cli_cleanup(NULL, NULL, &tok, &m);
                return 1;
            }
        }
        // Discovery registry: run modes announce themselves so the tray (or
        // any controller) can list every live runner. Best-effort — failure
        // never affects the run. Utility modes (--quantize/--caps/--bench)
        // stay unregistered.
        if (!bench_json && !quant_out) {
            const char *bn = strrchr(load_path, '/');
#ifdef _WIN32
            const char *bn2 = strrchr(load_path, '\\');
            if (bn2 && (!bn || bn2 > bn)) bn = bn2;
#endif
            bn = bn ? bn + 1 : load_path;
            const char *const names[] = { bn };
            const char *const paths[] = { load_path };
            instances_register(serve ? "serve" : "cli", serve ? port : 0,
                               names, paths, 1);
            atexit(instances_unregister);
        }
        if (m.mtp_layers)
            fprintf(stderr, "mtp: %d predictor block(s) declared and excluded "
                    "from the backbone (training-only; not consumed)\n",
                    m.mtp_layers);
        if (m.agent_profile)
            fprintf(stderr, "agent-profile: protocol=%u tokenizer=%u schema=%s digest=%s features=%llu\n",
                    m.agent_protocol_version, m.agent_tokenizer_version,
                    m.agent_schema_id, m.agent_schema_digest,
                    (unsigned long long)m.n_agent_required_features);
        char ident[256];
        sampler_ident(gguf_get_str(&m.gf, "general.name", NULL), m.path,
                      ident, sizeof(ident));
        int preset_tmpl = tmpl_override >= 0 ? tmpl_override
                        : template_detect(gguf_get_str(&m.gf,
                                          "tokenizer.chat_template", NULL), NULL);
        const sampler_preset *sp = sampler_resolve(&smp, m.arch, ident,
                                                   preset_tmpl, &ov);
        char sdesc[256];
        sampler_describe(&smp, sp, sdesc, sizeof(sdesc));
        fprintf(stderr, "sampling: %s\n", sdesc);
    } else {
        // swap mode: no model yet, so the preset is chosen per model at load
        sampler_resolve(&smp, NULL, NULL, -1, &ov);
    }

    if (serve) {
        if (registry) {
            // swap-mode server: models come and go at runtime; the record
            // carries none and readers ask /v1/models live
            instances_register("serve", port, NULL, NULL, 0);
            atexit(instances_unregister);
        }
        int rc = server_run(registry ? NULL : &m, registry ? NULL : &tok,
                            model_path, &mp, smp, &ov, port, parallel, n_threads,
                            ttl, draft_path, draft_k, ignore_eos, tmpl_override,
                            force_uncertified);
        free(owned_prompt);
        return rc;
    }

    engine e = {0};
    int32_t *toks = NULL;
    // Every failure from here on frees what the success paths free. Written
    // once so a new error return cannot forget it, and so `make debug` is the
    // leak gate the comment on cli_cleanup claims it is: measured under
    // LeakSanitizer on Linux before this, an unreadable --json-schema leaked
    // 237,668 bytes in 19 allocations and an over-long prompt 30,186 in 21,
    // while the successful run was clean.
#define CLI_FAIL do { cli_cleanup(&e, toks, &tok, &m); \
                      free(owned_prompt); return 1; } while (0)
    if (!engine_init(&e, &m, &tok, &smp)) { // e zero-initialized at declaration
        fprintf(stderr, "error: out of memory initializing engine\n");
        CLI_FAIL;
    }
    e.ignore_eos = ignore_eos;
    e.json_mode = json_mode;
    e.progress = true;
    if (draft_path) {
        e.dm = spec_draft_load(draft_path, &m, &mp);
        if (e.dm) {
            fprintf(stderr, "draft: %s (%d tokens per round)\n", draft_path, draft_k);
            e.draft_k = draft_k;
        }
    }
    if (schema_file) {
        size_t ssz = 0;
        char *sbuf = read_file(schema_file, &ssz);
        if (!sbuf) { fprintf(stderr, "error: cannot read %s\n", schema_file); CLI_FAIL; }
        struct jv *sj = json_parse(sbuf, ssz);
        free(sbuf);
        if (!sj) { fprintf(stderr, "error: %s is not valid JSON\n", schema_file); CLI_FAIL; }
        char serr[128];
        e.schema = schema_compile(sj, serr, sizeof(serr));
        jv_free(sj);
        if (!e.schema) { fprintf(stderr, "error: unsupported schema: %s\n", serr); CLI_FAIL; }
        sval_init(&e.sv, e.schema);
    }

    size_t tok_cap = (prompt ? strlen(prompt) : 0) + (size_t)m.n_ctx + 32;
    // tok_encode narrows the capacity to int, so keep it representable
    if (tok_cap > INT_MAX) tok_cap = INT_MAX;
    toks = malloc(sizeof(int32_t) * tok_cap);
    if (!toks) { fprintf(stderr, "error: out of memory\n"); CLI_FAIL; }
    double ptime, gtime, t0;
    int n_prompt, n_gen;

    if (score) {
        // Teacher-forced scoring (adaptation D1): per-position
        // log P(token | prefix) over the RAW prompt bytes — no template, no
        // sampling, no gradients. DEFAULT IS THE SOLO PATH: one forward per
        // position, the exact numerics the sampler sees at decode time —
        // measured 2026-08-21, the CPU batched forward is NOT bit-identical
        // to solo (max |dlp| ~1e-6 on the fixtures, diverging from batch row
        // 2), so a chunked scorer would quietly reintroduce the train/infer
        // mismatch this mode exists to abolish. RUNNER_SCORE_CHUNKED=1 opts
        // into the ~10x-faster batched path with that measured envelope;
        // tests/test_score.py pins the delta and the solo determinism.
        if (!prompt || !*prompt) {
            fprintf(stderr, "error: --score needs a prompt (-p or -f)\n");
            CLI_FAIL;
        }
        char *p = unescape(prompt);
        n_prompt = tok_encode(&tok, p, toks, (int)tok_cap, !no_bos, true);
        free(p);
        if (n_prompt < 0) {
            fprintf(stderr, "error: out of memory tokenizing prompt\n");
            CLI_FAIL;
        }
        if (n_prompt < 2) {
            fprintf(stderr, "error: --score needs at least 2 tokens to score "
                    "a transition (got %d)\n", n_prompt);
            CLI_FAIL;
        }
        if (n_prompt > m.n_ctx) {
            fprintf(stderr, "error: --score prompt is %d tokens, context is "
                    "%d\n", n_prompt, m.n_ctx);
            CLI_FAIL;
        }
        const char *ch_env = getenv("RUNNER_SCORE_CHUNKED");
        bool solo = !(ch_env && *ch_env && strcmp(ch_env, "0") != 0);
        printf("{\"schema_version\":\"xyntetik.runner.score.v1\","
               "\"n_tokens\":%d,\"n_scored\":%d,\"tokens\":[", n_prompt,
               n_prompt - 1);
        for (int i = 0; i < n_prompt; i++)
            printf("%s%d", i ? "," : "", toks[i]);
        printf("],\"logprobs\":[");
        double total = 0;
        int pos = 0, emitted = 0, fail = 0;
        while (pos < n_prompt - 1 && !fail) {
            int k = n_prompt - 1 - pos;
            int cap = m.spec_batch < m.n_batch ? m.spec_batch : m.n_batch;
            if (k > cap) k = cap;
            if (!solo && k > 1 &&
                model_forward_batch_keep(&m, toks + pos, k, pos)) {
                for (int b = 0; b < k; b++) {
                    float *lg = model_spec_row_logits(&m, b);
                    // raw-logit log-softmax, the lp_capture_pre arithmetic:
                    // float max, double sum of expf, float result
                    float mx = lg[0];
                    for (int i = 1; i < m.n_vocab; i++)
                        if (lg[i] > mx) mx = lg[i];
                    double sum = 0;
                    for (int i = 0; i < m.n_vocab; i++)
                        sum += expf(lg[i] - mx);
                    float lp = lg[toks[pos + b + 1]] - (mx + logf((float)sum));
                    printf("%s%.9g", emitted++ ? "," : "", (double)lp);
                    total += lp;
                }
                pos += k;
            } else {
                float *lg = model_forward(&m, toks[pos], pos);
                if (!lg) { fail = 1; break; }
                float mx = lg[0];
                for (int i = 1; i < m.n_vocab; i++)
                    if (lg[i] > mx) mx = lg[i];
                double sum = 0;
                for (int i = 0; i < m.n_vocab; i++)
                    sum += expf(lg[i] - mx);
                float lp = lg[toks[pos + 1]] - (mx + logf((float)sum));
                printf("%s%.9g", emitted++ ? "," : "", (double)lp);
                total += lp;
                pos += 1;
            }
        }
        if (fail) {
            printf("]}\n");
            fprintf(stderr, "error: forward pass failed at position %d\n", pos);
            CLI_FAIL;
        }
        double nll = -total, mean = nll / (n_prompt - 1);
        printf("],\"nll_total\":%.9g,\"nll_mean\":%.9g,\"ppl\":%.9g}\n",
               nll, mean, exp(mean));
        cli_cleanup(&e, toks, &tok, &m);
        free(owned_prompt);
        return 0;
    }

    if (bench_json) {
        // A bench that stops at EOS measures the model's chattiness, not the
        // engine: instruct models that answer a bare prompt with instant EOS
        // reported "0.0 tok/s". Decode every requested token.
        e.ignore_eos = true;
        // A ten-token default measured nothing that matters: on the first
        // outside install this bench looked healthy while a real 2,100-token
        // prompt took 89 s to reach its first word, because prefill — not
        // decode — was the wall. Absent an explicit -p/-f, benchmark a
        // realistic prefill instead, clamped to whatever context exists.
        enum { BENCH_PROMPT_TOKENS = 512 };
        char *p = NULL;
        if (prompt) {
            p = unescape(prompt);
        } else {
            static const char para[] =
                "The lighthouse keeper recorded the weather every morning: "
                "wind direction, cloud cover, the height of the swell against "
                "the north rocks, and the hour the fog lifted. ";
            size_t one = sizeof(para) - 1;
            size_t reps = (size_t)BENCH_PROMPT_TOKENS / 8 + 2;  // >= target tok
            p = malloc(one * reps + 1);
            if (p) {
                p[0] = 0;
                for (size_t r = 0; r < reps; r++) memcpy(p + r * one, para, one);
                p[one * reps] = 0;
            }
        }
        if (!p) {
            fprintf(stderr, "error: out of memory building benchmark prompt\n");
            CLI_FAIL;
        }
        n_prompt = tok_encode(&tok, p, toks, (int)tok_cap, !no_bos, true);
        if (n_prompt < 0) {
            fprintf(stderr, "error: out of memory tokenizing benchmark prompt\n");
            free(p);
            CLI_FAIL;
        }
        if (!prompt) {
            // Truncating the synthesized prompt is safe (it is filler) and is
            // what keeps the default runnable on a small context.
            int want = BENCH_PROMPT_TOKENS;
            // leave the decode phase its tokens, or the rate it reports is
            // measured over a couple of steps on a small-context model
            int room = m.n_ctx - (n_predict > 0 ? n_predict : 0) - 1;
            if (room < 1) room = m.n_ctx - 1;
            if (want > room) want = room;
            if (n_prompt > want) n_prompt = want;
        }
        if (n_prompt == 0 || n_prompt >= m.n_ctx) {
            fprintf(stderr, "error: benchmark prompt does not fit context\n");
            free(p);
            CLI_FAIL;
        }
        t0 = now_s();
        float *logits = engine_feed(&e, toks, n_prompt);
        ptime = now_s() - t0;
        if (!logits) {
            fprintf(stderr, "error: benchmark prompt exceeds context\n");
            free(p);
            CLI_FAIL;
        }
        n_gen = engine_generate(&e, logits, n_predict, discard_cb, NULL, &gtime);
        char esc_model[1024];
        json_escape(model_path, strlen(model_path), esc_model, sizeof(esc_model));
        // prompt_s is time-to-first-token for this prompt length: the number
        // an editor-assistant user actually waits on, and the one a rate alone
        // hides. Both phases report seconds beside their rates.
        printf("{\"model\":\"%s\",\"arch\":\"%s\",\"context\":%d,"
               "\"gpu_layers\":%d,\"layers\":%d,\"prompt_tokens\":%d,"
               "\"generated_tokens\":%d,\"prompt_tok_s\":%.3f,"
               "\"gen_tok_s\":%.3f,\"prompt_s\":%.3f,\"gen_s\":%.3f}\n",
               esc_model, m.arch, m.n_ctx, m.gpu_layers, m.n_layer,
               n_prompt, n_gen,
               n_prompt / (ptime > 0 ? ptime : 1e-9),
               n_gen / (gtime > 0 ? gtime : 1e-9), ptime, gtime);
        free(p);
        cli_cleanup(&e, toks, &tok, &m);
        free(owned_prompt);
        return 0;
    }

    if (!interactive) {
        // one-shot completion
        char *p = unescape(prompt);
        n_prompt = tok_encode(&tok, p, toks, (int)tok_cap, !no_bos, true);
        if (n_prompt < 0) { fprintf(stderr, "error: out of memory tokenizing prompt\n"); free(p); CLI_FAIL; }
        if (n_prompt == 0) { fprintf(stderr, "error: empty prompt\n"); free(p); CLI_FAIL; }
        if (n_prompt >= m.n_ctx) {
            fprintf(stderr, "error: prompt is %d tokens but context is %d — "
                    "rerun with -c %d or larger\n", n_prompt, m.n_ctx,
                    (n_prompt + n_predict + 1023) / 1024 * 1024);
            free(p);
            CLI_FAIL;
        }
        if (verbose) {
            fprintf(stderr, "prompt tokens (%d):", n_prompt);
            for (int i = 0; i < n_prompt && i < 64; i++) fprintf(stderr, " %d", toks[i]);
            fprintf(stderr, "\n");
        }
        t0 = now_s();
        float *logits = engine_feed(&e, toks, n_prompt);
        ptime = now_s() - t0;
        if (!logits) { fprintf(stderr, "error: prompt exceeds context\n"); free(p); CLI_FAIL; }

        if (!prompt_file && !json_mode && !schema_file)
            printf("%s", p); // don't echo file/json/schema prompts
        free(p);
        fflush(stdout);
        n_gen = engine_generate(&e, logits, n_predict, stdout_cb, NULL, &gtime);
        printf("\n");
        if (e.hit_stop && !json_mode) fprintf(stderr, "[end of text]\n");
        fprintf(stderr, "\nprompt: %d tok, %.2f tok/s | gen: %d tok, %.2f tok/s\n",
                n_prompt, n_prompt / (ptime > 0 ? ptime : 1e-9),
                n_gen, n_gen / (gtime > 0 ? gtime : 1e-9));
        cli_cleanup(&e, toks, &tok, &m);
        free(owned_prompt);
        return 0;
    }

    // interactive chat
    // resolved and validated up with the serve checks, so it is a TMPL_* here
    if (tmpl_override >= 0) tmpl = tmpl_override;
    if (tmpl < 0)
        tmpl = template_detect(gguf_get_str(&m.gf, "tokenizer.chat_template", NULL), &tok);
    if (!system_prompt) system_prompt = "You are a helpful assistant.";
    fprintf(stderr, "chat mode (template: %s) — Ctrl-D or /exit to quit\n\n",
            template_name(tmpl));

    bool first_turn = true;
    for (;;) {
        fprintf(stderr, "> ");
        bool line_oom = false;
        char *line = read_line(stdin, &line_oom);
        if (!line) {
            if (line_oom) fprintf(stderr, "[out of memory reading input]\n");
            break;
        }
        if (!strcmp(line, "/exit") || !strcmp(line, "/quit")) { free(line); break; }
        if (!line[0]) { free(line); continue; }

        chat_msg msgs[2];
        int n_msgs = 0;
        if (first_turn && system_prompt[0])
            msgs[n_msgs++] = (chat_msg){
                .role = "system", .content = system_prompt,
            };
        msgs[n_msgs++] = (chat_msg){ .role = "user", .content = line };
        // Measured, never estimated — the same rule the server holds itself
        // to. render_messages truncates the TAIL, so a fixed buffer that a
        // long --system overruns does not fail the turn, it deletes the
        // question and the generation header and asks something else.
        char *rendered = render_prompt_alloc(tmpl, msgs, n_msgs, true, thinking,
                                             NULL, strlen(line) +
                                             strlen(system_prompt) + 1024);
        free(line);   // the render is the only reader; every path below is safe
        if (!rendered) {
            fprintf(stderr, "[out of memory rendering the prompt — turn skipped]\n");
            continue;
        }
        n_prompt = tok_encode(&tok, rendered, toks, (int)tok_cap,
                              first_turn && !no_bos, true);
        free(rendered);
        first_turn = false;

        if (n_prompt < 0) {
            fprintf(stderr, "[out of memory tokenizing input — turn skipped]\n");
            continue;
        }
        if (e.pos + n_prompt >= m.n_ctx) {
            fprintf(stderr, "[context full — restart to continue]\n");
            break;
        }
        float *logits = engine_feed(&e, toks, n_prompt);
        if (!logits) { fprintf(stderr, "[context full]\n"); break; }

        jsonv_init(&e.jv); // fresh JSON constraint per turn
        chat_out co = { .in_think = false };
        // Harmony primes the analysis channel in the generation prompt, so the
        // stream starts INSIDE reasoning; a content-first split would never
        // see the open marker and would print the model's analysis (and the
        // "assistantfinal" handoff) as the answer. Same seam the server uses
        // for this template — see harmony_primed_think in completion.c.
        if (tmpl == TMPL_HARMONY)
            think_init_reasoning(&co.ts, m.think_open, m.think_close);
        else
            think_init(&co.ts, m.think_open, m.think_close);
        n_gen = engine_generate(&e, logits, n_predict, chat_cb, &co, &gtime);
        think_finish(&co.ts, chat_emit, &co);
        if (co.in_think) fputs("\n[/thinking]", stdout);
        think_free(&co.ts);
        printf("\n");
        fprintf(stderr, "[%d tok, %.1f tok/s]\n",
                n_gen, n_gen / (gtime > 0 ? gtime : 1e-9));
    }
    cli_cleanup(&e, toks, &tok, &m);
    free(owned_prompt);
    return 0;
#undef CLI_FAIL
}
