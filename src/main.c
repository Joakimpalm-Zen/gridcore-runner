// runner — CLI: one-shot completion, interactive chat, and server launcher.
#include "runner.h"

#include "compat.h"
#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int server_run(model_t *base, tokenizer *tok, const char *model_path,
               const model_params *mp, sampler defaults, int port, int parallel,
               int n_threads, int ttl);
int quantize_gguf(const char *in_path, const char *out_path, int target);

// ---------------------------------------------------------------- misc

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
        "  -t N           threads (default: min(8, cpus))\n"
        "  -s N           RNG seed (default: time)\n"
        "  --temp F       temperature (default 0.8, 0 = greedy)\n"
        "  --top-k N      top-k (default 40)\n"
        "  --top-p F      top-p (default 0.95)\n"
        "  --repeat-penalty F  (default 1.1)\n"
        "  --rope-scale F force linear rope position scaling by F\n"
        "  --rope-base F  override rope frequency base\n"
        "  --system TEXT  system prompt for chat mode\n"
        "  --chat-template chatml|llama2|llama3|zephyr|raw  (default: auto)\n"
        "  --no-bos       do not add BOS token\n"
        "  --ignore-eos   keep generating past end-of-text tokens\n"
        "  --gpu auto|off GPU offload if a backend is available (default auto)\n"
        "  --caps         print machine capabilities as JSON and exit\n"
        "  -v             verbose model info\n",
        prog);
}

static int stdout_cb(void *ud, const char *bytes, int n) {
    (void)ud;
    fwrite(bytes, 1, n, stdout);
    fflush(stdout);
    return 0;
}

int main(int argc, char **argv) {
    const char *model_path = NULL, *prompt = NULL, *system_prompt = NULL;
    const char *tmpl_arg = NULL, *prompt_file = NULL, *schema_file = NULL;
    const char *quant_out = NULL, *quant_type = "q4_0";
    int n_predict = 256, n_threads = 0, tmpl = -1, reserve_cpu_pct = 0;
    int port = 8080, parallel = 1, ttl = -1; // -1: 300 for swap mode, never for single
    bool interactive = false, verbose = false, no_bos = false;
    bool ignore_eos = false, json_mode = false, serve = false, caps = false;
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
        else if (!strcmp(a, "--serve")) serve = true;
        else if (!strcmp(a, "--port")) port = atoi(NEXT);
        else if (!strcmp(a, "--parallel")) parallel = atoi(NEXT);
        else if (!strcmp(a, "--ttl")) ttl = atoi(NEXT);
        else if (!strcmp(a, "--json")) json_mode = true;
        else if (!strcmp(a, "--json-schema")) schema_file = NEXT;
        else if (!strcmp(a, "--quantize")) quant_out = NEXT;
        else if (!strcmp(a, "--quant")) quant_type = NEXT;
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
        else if (!strcmp(a, "--gpu")) mp.gpu_mode = strcmp(NEXT, "off") ? GPU_AUTO : GPU_OFF;
        else if (!strcmp(a, "--reserve")) mp.reserve_vram_pct = mp.reserve_ram_pct = atoi(NEXT);
        else if (!strcmp(a, "--reserve-vram")) mp.reserve_vram_pct = atoi(NEXT);
        else if (!strcmp(a, "--reserve-ram")) mp.reserve_ram_pct = atoi(NEXT);
        else if (!strcmp(a, "--reserve-cpu")) reserve_cpu_pct = atoi(NEXT);
        else if (!strcmp(a, "--caps")) caps = true;
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown option %s\n", a); usage(argv[0]); return 1; }
    }
    if (n_threads <= 0 && reserve_cpu_pct > 0) {
        n_threads = plat_cpu_count() * reserve_cpu_pct / 100;
        if (n_threads < 1) n_threads = 1;
    }
    if (caps) {
        char gname[128];
        bool has_gpu = gpu_available(gname, sizeof(gname));
        printf("{\"os\":\"%s\",\"arch\":\"%s\",\"cpu_cores\":%d,"
               "\"ram_bytes\":%llu,\"gpu\":",
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
               plat_cpu_count(), (unsigned long long)plat_ram_bytes());
        if (has_gpu) {
            size_t vfree = 0, vtotal = 0;
            bool vm = gpu_mem_info(&vfree, &vtotal);
#ifdef __APPLE__
            printf("{\"backend\":\"metal\",\"name\":\"%s\",\"unified_memory\":true",
                   gname);
#else
            printf("{\"backend\":\"cuda\",\"name\":\"%s\",\"unified_memory\":false",
                   gname);
#endif
            if (vm)
                printf(",\"vram_bytes\":%llu,\"vram_free_bytes\":%llu",
                       (unsigned long long)vtotal, (unsigned long long)vfree);
            printf("}");
        } else
            printf("null");
        printf(",\"quants\":[\"F32\",\"F16\",\"BF16\",\"Q8_0\",\"Q4_0\",\"Q4_1\","
               "\"Q5_0\",\"Q5_1\",\"Q2_K\",\"Q3_K\",\"Q4_K\",\"Q5_K\",\"Q6_K\","
               "\"IQ4_NL\",\"IQ4_XS\"],"
               "\"gpu_quants\":[\"F32\",\"F16\",\"Q8_0\",\"Q4_0\",\"Q4_1\",\"Q5_0\",\"Q5_1\",\"Q4_K\",\"Q5_K\",\"Q6_K\"]}\n");
        return 0;
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
    if (!prompt && !interactive && !serve && !quant_out) {
        fprintf(stderr, "error: need -p PROMPT, -i, or --serve\n");
        usage(argv[0]);
        return 1;
    }
    if (smp.rng == 0) smp.rng = (uint64_t)time(NULL) ^ 0x9E3779B97F4A7C15ull;

    if (n_threads <= 0) {
        int nc = plat_cpu_count();
        n_threads = nc > 8 ? 8 : nc;
    }
    mp.verbose = verbose;
    mp.n_threads = serve ? 1 : n_threads; // server slots create their own pools

    f16_init();

    bool registry = serve && strchr(model_path, '=') != NULL;

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
    if (!registry) {
        double t1 = now_s();
        if (!model_load(&m, model_path, &mp)) return 1;
        if (!tokenizer_init(&tok, &m.gf)) return 1;
        fprintf(stderr, "loaded %s | %s | %d layers | ctx %d | %d threads | %.2fs\n",
                model_path, m.arch, m.n_layer, m.n_ctx, n_threads, now_s() - t1);
    }

    if (serve)
        return server_run(registry ? NULL : &m, registry ? NULL : &tok,
                          model_path, &mp, smp, port, parallel, n_threads, ttl);

    engine e;
    engine_init(&e, &m, &tok, &smp);
    e.ignore_eos = ignore_eos;
    e.json_mode = json_mode;
    e.progress = true;
    if (schema_file) {
        FILE *sf = fopen(schema_file, "rb");
        if (!sf) { fprintf(stderr, "error: cannot open %s\n", schema_file); return 1; }
        fseek(sf, 0, SEEK_END);
        long ssz = ftell(sf);
        fseek(sf, 0, SEEK_SET);
        char *sbuf = malloc(ssz + 1);
        ssz = (long)fread(sbuf, 1, ssz, sf);
        fclose(sf);
        struct jv *sj = json_parse(sbuf, ssz);
        free(sbuf);
        if (!sj) { fprintf(stderr, "error: %s is not valid JSON\n", schema_file); return 1; }
        char serr[128];
        e.schema = schema_compile(sj, serr, sizeof(serr));
        jv_free(sj);
        if (!e.schema) { fprintf(stderr, "error: unsupported schema: %s\n", serr); return 1; }
        sval_init(&e.sv, e.schema);
    }

    size_t tok_cap = (prompt ? strlen(prompt) : 0) + m.n_ctx + 32;
    int32_t *toks = malloc(sizeof(int32_t) * tok_cap);
    double ptime, gtime, t0;
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

        if (e.pos + n_prompt >= m.n_ctx) {
            fprintf(stderr, "[context full — restart to continue]\n");
            break;
        }
        float *logits = engine_feed(&e, toks, n_prompt);
        if (!logits) { fprintf(stderr, "[context full]\n"); break; }

        jsonv_init(&e.jv); // fresh JSON constraint per turn
        n_gen = engine_generate(&e, logits, n_predict, stdout_cb, NULL, &gtime);
        printf("\n");
        fprintf(stderr, "[%d tok, %.1f tok/s]\n",
                n_gen, n_gen / (gtime > 0 ? gtime : 1e-9));
    }
    return 0;
}
