// runner — CLI: one-shot completion, interactive chat, and server launcher.
#include "runner.h"

#include "compat.h"
#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>

int server_run(model_t *base, tokenizer *tok, const char *model_path,
               const model_params *mp, sampler defaults,
               const sampler_override *ov, int port, int parallel,
               int n_threads, int ttl, const char *draft_path, int draft_k,
               bool ignore_eos);
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
        "  --port N       server port (default 8080)\n"
        "  --parallel N   parallel inference slots in server mode (default 1)\n"
        "                 -m \"name=path,name2=path2\" serves multiple models,\n"
        "                 loading the requested one on demand (swap mode)\n"
        "  --ttl N        swap mode: unload an idle model after N seconds\n"
        "                 (default 300, 0 = never)\n"
        "  --json         constrain output to a single valid JSON object\n"
        "  --json-schema F constrain output to the JSON Schema in file F\n"
        "  --quantize OUT rewrite the model to OUT.gguf (see --quant) and exit\n"
        "  --quant T      quantize target: q8_0 | q4_0 | f16 (default q4_0)\n"
        "  -n N           max tokens to generate (default 256, -1 = until EOS)\n"
        "  -c N           context length (default: min(model max, 4096));\n"
        "                 beyond the training context, YaRN rope scaling is\n"
        "                 applied automatically\n"
        "  -b N           prompt batch size (default 64)\n"
        "  -t N           threads (default: physical cores, capped at 64)\n"
        "  -s N           RNG seed (default: time)\n"
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
        "  --chat-template chatml|llama2|llama3|mistral|zephyr|phi3|gemma|gemma4|apertus|ornith|raw\n"
        "                 (default: auto)\n"
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
        "  --draft PATH   small same-vocab GGUF for speculative decoding\n"
        "                 (one-shot, chat, and single-model --serve)\n"
        "  --draft-k N    draft tokens per round (default 4)\n"
        "  --bench-json   run a small decode benchmark and print JSON metrics\n"
        "  --caps         print machine capabilities as JSON and exit\n"
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

int main(int argc, char **argv) {
    const char *model_path = NULL, *prompt = NULL, *system_prompt = NULL;
    char *owned_prompt = NULL;
    const char *tmpl_arg = NULL, *prompt_file = NULL, *schema_file = NULL;
    const char *quant_out = NULL, *quant_type = "q4_0";
    int n_predict = 256, n_threads = 0, tmpl = -1, reserve_cpu_pct = 0;
    int port = 8080, parallel = 1, ttl = -1; // -1: 300 for swap mode, never for single
    long parent_pid = 0;
    const char *draft_path = NULL;
    int draft_k = 4;
    bool interactive = false, verbose = false, no_bos = false;
    bool ignore_eos = false, json_mode = false, serve = false, caps = false;
    bool bench_json = false;
    model_params mp = {0};
    // The four sampling knobs are filled in from the model's family preset
    // once the model is known; anything named on the command line is recorded
    // here and wins over the preset.
    sampler smp = { .rng = 0 };
    sampler_override ov = {0};

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
        else if (!strcmp(a, "-s")) smp.rng = u64_arg(a, NEXT);
        else if (!strcmp(a, "-i")) interactive = true;
        else if (!strcmp(a, "-v")) verbose = true;
        else if (!strcmp(a, "--serve")) serve = true;
        else if (!strcmp(a, "--port")) port = (int)int_arg(a, NEXT, 1, 65535);
        else if (!strcmp(a, "--parallel")) parallel = (int)int_arg(a, NEXT, 1, 16);
        else if (!strcmp(a, "--ttl")) ttl = (int)int_arg(a, NEXT, -1, INT_MAX);
        else if (!strcmp(a, "--json")) json_mode = true;
        else if (!strcmp(a, "--json-schema")) schema_file = NEXT;
        else if (!strcmp(a, "--quantize")) quant_out = NEXT;
        else if (!strcmp(a, "--quant")) quant_type = NEXT;
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
        else if (!strcmp(a, "--draft"))   draft_path = NEXT;
        else if (!strcmp(a, "--draft-k")) draft_k = (int)int_arg(a, NEXT, 1, 15);
        else if (!strcmp(a, "--bench-json")) bench_json = true;
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
        else if (!strcmp(a, "--parent-pid")) parent_pid = (long)int_arg(a, NEXT, 1, LONG_MAX);
        else if (!strcmp(a, "--caps")) caps = true;
        else if (!strcmp(a, "--version")) { printf("runner %s\n", RUNNER_VERSION); return 0; }
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown option %s\n", a); usage(argv[0]); return 1; }
    }
    plat_parent_watch(parent_pid);
    if (n_threads <= 0 && reserve_cpu_pct > 0) {
        n_threads = plat_cpu_count() * reserve_cpu_pct / 100;
        if (n_threads < 1) n_threads = 1;
    }
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
            // Read this before believing gpu_quants. MXFP4 and the Q*_K family
            // are listed there on every backend, but Metal refuses a model
            // with experts before it looks at the quant at all — so "MXFP4 is
            // a GPU quant" and "gpt-oss runs on the GPU" are different claims,
            // and only this one answers the second.
            printf(",\"moe\":%s", gpu_moe_ok() ? "true" : "false");
            printf(",\"kv_q8\":%s", gpu_kv_q8_ok() ? "true" : "false");
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
        printf(",\"quants\":[\"F32\",\"F16\",\"BF16\",\"Q8_0\",\"Q4_0\",\"Q4_1\","
               "\"Q5_0\",\"Q5_1\",\"Q2_K\",\"Q3_K\",\"Q4_K\",\"Q5_K\",\"Q6_K\","
               "\"IQ4_NL\",\"IQ4_XS\",\"MXFP4\"],"
               // Q3_K has a CUDA kernel (added with GPU MoE) and MXFP4 has one
               // as of the gpt-oss CUDA work; keep this in sync with
               // gpu_type_ok() in cuda.c. Q2_K remains CPU-only.
               "\"gpu_quants\":[\"F32\",\"F16\",\"Q8_0\",\"Q4_0\",\"Q4_1\",\"Q5_0\",\"Q5_1\","
               "\"Q3_K\",\"Q4_K\",\"Q5_K\",\"Q6_K\",\"IQ4_NL\",\"IQ4_XS\",\"MXFP4\"]");
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
    if (!prompt && !interactive && !serve && !quant_out && !bench_json) {
        fprintf(stderr, "error: need -p PROMPT, -i, or --serve\n");
        usage(argv[0]);
        return 1;
    }
    if (smp.rng == 0) smp.rng = (uint64_t)time(NULL) ^ 0x9E3779B97F4A7C15ull;

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

    if (quant_out) {
        int tt = !strcmp(quant_type, "q8_0") ? T_Q8_0 :
                 !strcmp(quant_type, "q4_0") ? T_Q4_0 :
                 !strcmp(quant_type, "f16")  ? T_F16 : -1;
        if (tt < 0) {
            fprintf(stderr, "error: --quant must be q8_0, q4_0, or f16\n");
            return 1;
        }
        return quantize_gguf(model_path, quant_out, tt);
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
        if (!tokenizer_init(&tok, &m.gf)) return 1;
        fprintf(stderr, "loaded %s | %s | %d layers | ctx %d | %d threads | %.2fs\n",
                load_path, m.arch, m.n_layer, m.n_ctx, tpool_size(m.tp),
                now_s() - t1);
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
        const sampler_preset *sp = sampler_resolve(&smp, m.arch, ident, &ov);
        char sdesc[256];
        sampler_describe(&smp, sp, sdesc, sizeof(sdesc));
        fprintf(stderr, "sampling: %s\n", sdesc);
    } else {
        // swap mode: no model yet, so the preset is chosen per model at load
        sampler_resolve(&smp, NULL, NULL, &ov);
    }

    if (serve) {
        int rc = server_run(registry ? NULL : &m, registry ? NULL : &tok,
                            model_path, &mp, smp, &ov, port, parallel, n_threads,
                            ttl, draft_path, draft_k, ignore_eos);
        free(owned_prompt);
        return rc;
    }

    engine e = {0};
    if (!engine_init(&e, &m, &tok, &smp)) { // e zero-initialized at declaration
        fprintf(stderr, "error: out of memory initializing engine\n");
        return 1;
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
        if (!sbuf) { fprintf(stderr, "error: cannot read %s\n", schema_file); return 1; }
        struct jv *sj = json_parse(sbuf, ssz);
        free(sbuf);
        if (!sj) { fprintf(stderr, "error: %s is not valid JSON\n", schema_file); return 1; }
        char serr[128];
        e.schema = schema_compile(sj, serr, sizeof(serr));
        jv_free(sj);
        if (!e.schema) { fprintf(stderr, "error: unsupported schema: %s\n", serr); return 1; }
        sval_init(&e.sv, e.schema);
    }

    size_t tok_cap = (prompt ? strlen(prompt) : 0) + (size_t)m.n_ctx + 32;
    // tok_encode narrows the capacity to int, so keep it representable
    if (tok_cap > INT_MAX) tok_cap = INT_MAX;
    int32_t *toks = malloc(sizeof(int32_t) * tok_cap);
    if (!toks) { fprintf(stderr, "error: out of memory\n"); return 1; }
    double ptime, gtime, t0;
    int n_prompt, n_gen;

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
            return 1;
        }
        n_prompt = tok_encode(&tok, p, toks, (int)tok_cap, !no_bos, true);
        if (n_prompt < 0) {
            fprintf(stderr, "error: out of memory tokenizing benchmark prompt\n");
            free(p);
            return 1;
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
            return 1;
        }
        t0 = now_s();
        float *logits = engine_feed(&e, toks, n_prompt);
        ptime = now_s() - t0;
        if (!logits) {
            fprintf(stderr, "error: benchmark prompt exceeds context\n");
            free(p);
            return 1;
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
        if (n_prompt < 0) { fprintf(stderr, "error: out of memory tokenizing prompt\n"); return 1; }
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
        float *logits = engine_feed(&e, toks, n_prompt);
        ptime = now_s() - t0;
        if (!logits) { fprintf(stderr, "error: prompt exceeds context\n"); return 1; }

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
    if (tmpl_arg) tmpl = template_from_name(tmpl_arg);
    if (tmpl < 0)
        tmpl = template_detect(gguf_get_str(&m.gf, "tokenizer.chat_template", NULL), &tok);
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

        chat_msg msgs[2];
        int n_msgs = 0;
        if (first_turn && system_prompt[0])
            msgs[n_msgs++] = (chat_msg){ "system", system_prompt };
        msgs[n_msgs++] = (chat_msg){ "user", line };
        render_messages(tmpl, msgs, n_msgs, true, rendered, sizeof(rendered));
        n_prompt = tok_encode(&tok, rendered, toks, (int)tok_cap,
                              first_turn && !no_bos, true);
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
}
