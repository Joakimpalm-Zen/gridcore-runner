// HTTP server: OpenAI-compatible API with N parallel inference slots.
//
//   POST /v1/chat/completions   messages, sampling params, stream (SSE),
//                               response_format {"type":"json_object"}
//   POST /v1/completions        raw prompt completion
//   POST /v1/embeddings         mean-pooled L2-normed embeddings
//   GET  /v1/models             the loaded model
//   GET  /v1/capabilities       registry + feature discovery
//   GET  /health                liveness
//
// Swap-mode request bodies may carry "keep_alive" (seconds of idle before
// the model unloads; 0 = unload now, negative = keep forever).
//
// Each slot owns a full inference context (KV cache + thread pool); model
// weights are shared between slots through the page cache (mmap).
#include "runner.h"
#include "json.h"
#include "compat.h"
#include "http.h"
#include "server_int.h"
#include "scheduler.h"
#include "completion.h"
#include "api.h"
#include "server.h"

#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <stdatomic.h>




// ---------------------------------------------------------------- routes

// Declared in api.h, where the reason it exists is written down.
char *render_prompt_alloc(int tmpl, const chat_msg *msgs, int n_msgs,
                          bool add_assistant, int thinking, const jv *tools,
                          size_t hint) {
    size_t cap = hint < 512 ? 512 : hint;
    for (;;) {
        char *p = malloc(cap);
        if (!p) return NULL;
        size_t n = render_messages_with_tools(tmpl, msgs, n_msgs, add_assistant,
                                              thinking, tools, p, cap);
        // Complete iff the render did not fill the buffer. Two independent
        // signals, because one of them lives in a module this call does not
        // own: emit() reports an offset at or past `cap` when snprintf
        // truncated, AND a truncating snprintf always fills to cap-1. A
        // prompt that happens to end one byte short of `cap` is grown once
        // unnecessarily, which costs a render; the reverse mistake costs the
        // user's question.
        if (n < cap && strlen(p) + 1 < cap) return p;
        free(p);
        if (cap > SIZE_MAX / 2) return NULL;  // no size can hold it
        size_t want = (n > cap ? n : cap) + 1;
        cap = want > cap * 2 ? want : cap * 2;
    }
}

// flatten one OpenAI message to plain text: string content passes through;
// the AI-SDK part-array form (Cline et al.) concatenates its text parts;
// assistant tool_calls render in runner's own call syntax so replayed
// history reads like what the model actually emitted. Returns a heap
// string, or NULL when the message carries nothing usable.
static char *message_text(jv *msg, int tmpl) {
    jv *content = jv_get(msg, "content");
    sbuf b = {0};
    const char *role = jv_str(jv_get(msg, "role"), "user");
    const char *reason = jv_str(jv_get(msg, "reasoning_content"), NULL);
    if (tmpl == TMPL_ORNITH && !strcmp(role, "assistant")) {
        sb_lit(&b, "<think>\n");
        if (reason) sb_put(&b, reason, strlen(reason));
        sb_lit(&b, "\n</think>\n\n");
    }
    // gemma4 renders an assistant turn's CALLS before its visible text --
    //   <|tool_call>...<tool_call|><|tool_response>...<tool_response|>SURE
    // -- so the flattened turn has to carry them in that order for the
    // renderer to place the results between the two. Every other family
    // appends the calls after the content, which is what they have always
    // done and what their own references expect.
    jv *calls = jv_get(msg, "tool_calls");
    if (tmpl == TMPL_GEMMA4) tool_history_render_for(tmpl, calls, &b);
    // Muse's reference emits NO visible text in a turn that carries calls:
    // the turn is addressed `to=NAME` and holds only the
    // <atem:function_calls> block, so text before it is a shape the model
    // never saw at that position. Runner rendered it anyway until 2026-08-14,
    // which the matrix could not see because it had no case with text AND
    // calls in one turn.
    //
    // This DISCARDS something the client sent, and that is the deliberate
    // choice rather than an oversight: a recipient turn is the model's own
    // protocol for "I am calling this function", and prepending prose to it
    // teaches a turn shape its training never contained. The text is not lost
    // to the conversation -- a client that wants the model to say something
    // before calling sends it as its own assistant turn, which renders as an
    // ordinary `to=user` message. Ruled 2026-08-14; the alternatives
    // considered were allowlisting the divergence and synthesising a
    // preceding `to=user` turn, which no reference writes.
    bool muse_calls = tmpl == TMPL_MUSE && calls && calls->type == J_ARR &&
                      calls->n > 0;
    if (muse_calls) content = NULL;
    if (content && content->type == J_STR) {
        sb_put(&b, content->str, strlen(content->str));
    } else if (content && content->type == J_ARR) {
        // `parts` counts TEXT PARTS, not bytes already in the buffer: the
        // separator belongs between two text parts. Testing b.n would put one
        // in front of the first part whenever something was written ahead of
        // the content -- ornith's <think> block, or gemma4's calls above.
        int parts = 0;
        for (int i = 0; i < content->n; i++) {
            const char *type = jv_str(jv_get(content->items[i], "type"), "");
            const char *text = jv_str(jv_get(content->items[i], "text"), NULL);
            if (strcmp(type, "text") != 0 || !text) continue; // images etc.
            if (parts++) sb_lit(&b, "\n");
            sb_put(&b, text, strlen(text));
        }
    }
    if (tmpl != TMPL_GEMMA4) tool_history_render_for(tmpl, calls, &b);
    if (tmpl == TMPL_ORNITH && !strcmp(role, "tool")) {
        sbuf wrapped = {0};
        sb_lit(&wrapped, "<tool_response>\n");
        if (b.s) sb_put(&wrapped, b.s, b.n);
        sb_lit(&wrapped, "\n</tool_response>");
        free(b.s);
        b = wrapped;
    }
    return b.s;
}

// The function a tool turn is reporting for, resolved for HARMONY.
//
// template.c's tool_result_name() has a last resort this surface cannot use:
// with no `name` and an unmatched `tool_call_id` it hands back the ID, and
// Harmony authors the turn by that string -- `<|start|>functions.call_1`,
// a function name invented from an identifier and declared nowhere. So the
// lookup is done here without that step, and an unresolved result is treated
// as unresolved rather than named after its id.
static const char *harmony_result_name(const jv *msgs, int message_index) {
    jv *msg = msgs->items[message_index];
    const char *name = jv_str(jv_get(msg, "name"), NULL);
    if (name && name[0]) return name;
    const char *id = jv_str(jv_get(msg, "tool_call_id"), NULL);
    if (!id) return NULL;
    for (int i = 0; i < message_index; i++) {
        jv *calls = jv_get(msgs->items[i], "tool_calls");
        if (!calls || calls->type != J_ARR) continue;
        for (int k = 0; k < calls->n; k++) {
            const char *candidate = jv_str(jv_get(calls->items[k], "id"), NULL);
            if (!candidate || strcmp(candidate, id)) continue;
            jv *fn = jv_get(calls->items[k], "function");
            const char *fname = jv_str(jv_get(fn, "name"), NULL);
            if (fname && fname[0]) return fname;
        }
    }
    return NULL;
}

// Declared in api.h, where the reason it exists is written down. It lived as
// an identical private copy in this file, api_responses.c and api_anthropic.c
// -- one definition each of the same five lines, because the commit that
// introduced it could not add to api.h.
const char *sole_tool_name(const jv *tools) {
    if (!tools || tools->type != J_ARR || tools->n != 1) return NULL;
    jv *fn = jv_get(tools->items[0], "function");
    const char *name = jv_str(jv_get(fn, "name"), NULL);
    return name && name[0] ? name : NULL;
}

static void handle_chat(slot_t *s, sock_t fd, jv *req) {
    jv *msgs = jv_get(req, "messages");
    if (!msgs || msgs->type != J_ARR || msgs->n == 0) {
        send_error(fd, 400, "missing messages");
        return;
    }
    // OpenAI "tools" become a leading system turn (template.c owns the syntax).
    //
    // Strict mode compiles them into a discriminated union that constrains
    // sampling, so the model cannot name an undeclared tool or malform its
    // arguments. It applies to streamed requests too: the envelope is
    // demultiplexed as it is generated (tool_stream) rather than parsed
    // afterward, so both paths reach the same call from the same guarantee.
    jv *tools = jv_get(req, "tools");
    tool_envelope env = {0};
    // parallel_tool_calls is read BEFORE the envelope is built, because it
    // changes the envelope's shape. Silently ignoring a request for several
    // calls would leave the caller expecting calls it never gets.
    bool parallel = false;
    if (!request_bool(req, "parallel_tool_calls", false, &parallel)) {
        send_error(fd, 400, "parallel_tool_calls must be a boolean");
        return;
    }
    bool atem_tool_calling = true;
    if (!request_bool(req, "atem_tool_calling", true, &atem_tool_calling)) {
        send_error(fd, 400, "atem_tool_calling must be a boolean");
        return;
    }
    // parallel_tool_calls:true + stream:true used to be refused here: the
    // demultiplexer tracked one call per turn, and downgrading silently
    // would leave the caller expecting calls it never got. tool_stream now
    // loops the {"calls":[...]} array the same way ts_atem already looped
    // native <atem:invoke> blocks -- announcing each call on its own index,
    // closing it, and moving to the next -- so both modes stream several
    // calls in one turn and the refusal is gone.
    char terr[224];
    int rc = tool_envelope_build_ex(tools, jv_get(req, "tool_choice"),
                                    request_schema(req), parallel, &env,
                                    terr, sizeof(terr));
    if (rc < 0) { send_error(fd, 400, terr); return; }
    // Ornith is specifically trained on qwen3_xml. Keep its native protocol
    // instead of forcing the model into runner's generic JSON envelope.
    bool strict = rc == 1 && s->tmpl != TMPL_ORNITH;
    if (strict && s->tmpl == TMPL_MUSE) {
        env.atem = atem_tool_calling;
        env.muse_user_header = !atem_tool_calling;
        env.tools = tools;
    }
    if (strict && s->tmpl == TMPL_HARMONY) {
        env.harmony = true;
        env.tools = tools;
    }
    if (strict && s->tmpl == TMPL_GEMMA4) {
        env.gemma4 = true;
        env.tools = tools;
    }
    // When the strict envelope does not apply — no tools declared, or the
    // ornith template's native protocol — the flag is vacuous and stays
    // TOLERATED, exactly as before: ordinary OpenAI-shaped traffic sends
    // parallel_tool_calls alongside requests that will never call anything,
    // and rejecting those would break it.
    // Families with a native declaration syntax render it themselves, from
    // the structured `tools` passed to the renderer below. Everyone else gets
    // the strict envelope's teaching turn, or the generic declaration block
    // when the envelope does not apply.
    //
    // gemma4 joined that list on 2026-08-14. Unlike muse and harmony it is
    // listed here UNCONDITIONALLY rather than only on the strict path: its
    // reference emits `<|tool>declaration:...<tool|>` whenever `tools` are
    // present, `tool_choice` being a concept the template does not have, and a
    // caller who sends tool_choice:"none" to gemma4 was previously handed the
    // generic block -- a second, untrained tool protocol -- for a turn that
    // must not call anything anyway.
    bool native_decl = s->tmpl == TMPL_GEMMA4 ||
                       (s->tmpl == TMPL_MUSE && env.atem) ||
                       s->tmpl == TMPL_HARMONY;
    sbuf ts = {0};
    if (strict && !native_decl)
        sb_put(&ts, env.system_turn, strlen(env.system_turn));
    else if (!native_decl && s->tmpl != TMPL_MUSE)
        tools_render_for(s->tmpl, tools, &ts);
    bool ornith_merged_system = false;
    if (s->tmpl == TMPL_ORNITH && ts.n && msgs->n > 0 &&
        !strcmp(jv_str(jv_get(msgs->items[0], "role"), ""), "system")) {
        char *system = message_text(msgs->items[0], s->tmpl);
        if (system && system[0]) {
            sb_lit(&ts, "\n\n");
            sb_put(&ts, system, strlen(system));
        }
        free(system);
        ornith_merged_system = true;
    }
    size_t cm_cap = (size_t)msgs->n + 1;
    if (s->tmpl == TMPL_HARMONY) {
        for (int i = 0; i < msgs->n; i++) {
            jv *calls = jv_get(msgs->items[i], "tool_calls");
            if (calls && calls->type == J_ARR) cm_cap += (size_t)calls->n;
            if (jv_str(jv_get(msgs->items[i], "reasoning_content"), NULL))
                cm_cap++;
        }
    }
    chat_msg *cm = malloc(sizeof(chat_msg) * cm_cap);
    char **owned = malloc(sizeof(char *) * msgs->n);
    // client-controlled size (a 32MB body of tiny messages): a NULL here would
    // be indexed below. Fail the request cleanly instead of crashing.
    if (!cm || !owned) {
        free(cm); free(owned); free(ts.s);
        tool_envelope_free(&env);
        send_error(fd, 500, "out of memory building chat prompt");
        return;
    }
    size_t total = ts.n + 64;
    int n_cm = 0, n_own = 0;
    if (ts.n)
        cm[n_cm++] = (chat_msg){ .role = "system", .content = ts.s };
    for (int i = 0; i < msgs->n; i++) {
        if (i == 0 && ornith_merged_system) continue;
        const char *role = jv_str(jv_get(msgs->items[i], "role"), "user");
        const char *turn_name = NULL;
        // gemma4 names the function in its <|tool_response> block
        // (`response:NAME{...}`) exactly as muse names its tool turn, so the
        // result needs the same resolution. Without it every replayed result
        // reported for the template's own `'unknown'` fallback.
        if ((s->tmpl == TMPL_MUSE || s->tmpl == TMPL_GEMMA4) &&
            !strcmp(role, "tool"))
            turn_name = tool_result_name(msgs, i);
        if (s->tmpl == TMPL_HARMONY && !strcmp(role, "tool")) {
            turn_name = harmony_result_name(msgs, i);
            if (!turn_name) turn_name = sole_tool_name(tools);
            // Harmony authors a tool turn BY the function that ran, so a
            // result nothing can name has no on-protocol rendering: it would
            // go out either as a turn shape the model has never seen or under
            // an invented function name, and either way the caller is told
            // 200. Refused instead, naming the field that would fix it.
            if (!turn_name) {
                const char *id = jv_str(jv_get(msgs->items[i], "tool_call_id"),
                                        NULL);
                for (int k = 0; k < n_own; k++) free(owned[k]);
                free(owned); free(cm); free(ts.s);
                tool_envelope_free(&env);
                char e[288];
                snprintf(e, sizeof(e),
                         "messages[%d] is a tool result that cannot be "
                         "attributed to a tool: %s%.40s%s. Give it a `name`, "
                         "or a `tool_call_id` matching an earlier assistant "
                         "tool_calls[].id.", i,
                         id ? "no earlier assistant tool_calls[] carries id \""
                            : "it carries neither", id ? id : "",
                         id ? "\"" : " `name` nor `tool_call_id`");
                send_error(fd, 400, e);
                return;
            }
        }
        if (s->tmpl == TMPL_MUSE && !strcmp(role, "assistant")) {
            jv *calls = jv_get(msgs->items[i], "tool_calls");
            if (calls && calls->type == J_ARR && calls->n) {
                jv *fn = jv_get(calls->items[0], "function");
                turn_name = jv_str(jv_get(fn, "name"), NULL);
            }
        }
        if (s->tmpl == TMPL_HARMONY && !strcmp(role, "assistant")) {
            const char *reason = jv_str(
                jv_get(msgs->items[i], "reasoning_content"), NULL);
            if (reason && reason[0]) {
                cm[n_cm++] = (chat_msg){ .role = "assistant",
                                        .content = reason,
                                        .channel = "analysis" };
                total += strlen(reason) + 64;
            }
            char *visible = message_text(msgs->items[i], s->tmpl);
            jv *calls = jv_get(msgs->items[i], "tool_calls");
            bool have_calls = calls && calls->type == J_ARR && calls->n;
            if (visible && visible[0]) {
                owned[n_own++] = visible;
                cm[n_cm++] = (chat_msg){ .role = "assistant",
                                        .content = visible,
                                        .channel = have_calls
                                                   ? "commentary" : NULL };
                total += strlen(visible) + 64;
            } else {
                free(visible);
            }
            if (have_calls) for (int k = 0; k < calls->n; k++) {
                jv *fn = jv_get(calls->items[k], "function");
                const char *name = jv_str(jv_get(fn, "name"), NULL);
                const char *args = jv_str(jv_get(fn, "arguments"), "{}");
                if (!name) continue;
                cm[n_cm++] = (chat_msg){ .role = "assistant",
                                        .content = args, .name = name };
                total += strlen(name) + strlen(args) + 96;
            }
            continue;
        }
        char *content = message_text(msgs->items[i], s->tmpl);
        if (!content) continue;
        owned[n_own++] = content;
        if (s->tmpl == TMPL_ORNITH && !strcmp(role, "tool")) role = "user";
        cm[n_cm++] = (chat_msg){
            .role = role, .content = content, .name = turn_name,
        };
        total += strlen(role) + strlen(content) + 64;
    }
    if (n_cm == 0) {
        free(owned);
        free(cm);
        free(ts.s);
        tool_envelope_free(&env);
        send_error(fd, 400, "no message content");
        return;
    }
    // Native templates render the declarations themselves -- muse's JSON
    // block, Harmony's TypeScript namespace -- so their bytes belong in the
    // opening guess. It is only a guess: render_prompt_alloc measures the
    // real size and grows, so an under-count here costs a render pass, not a
    // truncated prompt.
    const jv *native_tools = (s->tmpl == TMPL_MUSE && env.atem) ||
                             (s->tmpl == TMPL_HARMONY && env.harmony) ||
                             s->tmpl == TMPL_GEMMA4
                                 ? tools : NULL;
    sbuf tool_bytes = {0};
    if (native_tools) jv_dump(tools, &tool_bytes);
    total += tool_bytes.n + 4096;
    free(tool_bytes.s);
    char *prompt = render_prompt_alloc(s->tmpl, cm, n_cm, true,
                                       req_thinking_mode(req), native_tools,
                                       total + 256);
    if (!prompt) {
        for (int i = 0; i < n_own; i++) free(owned[i]);
        free(owned); free(cm); free(ts.s);
        tool_envelope_free(&env);
        send_error(fd, 500, "out of memory building chat prompt");
        return;
    }
    run_completion(s, fd, prompt, API_CHAT, req, strict ? &env : NULL);
    free(prompt);
    for (int i = 0; i < n_own; i++) free(owned[i]);
    free(owned);
    free(cm);
    free(ts.s);
    tool_envelope_free(&env);
}

static void handle_completion(slot_t *s, sock_t fd, jv *req) {
    const char *prompt = jv_str(jv_get(req, "prompt"), NULL);
    if (!prompt) { send_error(fd, 400, "missing prompt"); return; }
    run_completion(s, fd, prompt, API_TEXT, req, NULL);
}

// One byte of the vector as the wire sees it: little-endian float32, spelled
// out rather than memcpy'd wholesale so a big-endian host emits the same bytes.
static unsigned char emb_byte(const float *v, size_t i) {
    uint32_t bits;
    memcpy(&bits, &v[i >> 2], sizeof bits);
    return (unsigned char)(bits >> (8 * (i & 3)));
}

// base64 of that byte stream. Not an optimisation we chose: the OpenAI SDKs
// send `encoding_format: "base64"` by DEFAULT and decode it client-side, so a
// server that only speaks "float" cannot be called by an official client at
// all. Refusing it was a 400 on every SDK embeddings call.
static void sb_emb_b64(sbuf *r, const float *v, int n) {
    static const char T[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t nb = (size_t)n * sizeof(float);
    for (size_t i = 0; i < nb; i += 3) {
        unsigned char b0 = emb_byte(v, i);
        unsigned char b1 = i + 1 < nb ? emb_byte(v, i + 1) : 0;
        unsigned char b2 = i + 2 < nb ? emb_byte(v, i + 2) : 0;
        char q[4];
        q[0] = T[b0 >> 2];
        q[1] = T[((b0 & 0x03) << 4) | (b1 >> 4)];
        q[2] = i + 1 < nb ? T[((b1 & 0x0F) << 2) | (b2 >> 6)] : '=';
        q[3] = i + 2 < nb ? T[b2 & 0x3F] : '=';
        sb_put(r, q, 4);
    }
}

static void handle_embeddings(slot_t *s, sock_t fd, jv *req) {
    jv *input = jv_get(req, "input");
    const char *one = jv_str(input, NULL);
    int n_in = one ? 1 : (input && input->type == J_ARR ? input->n : 0);
    if (n_in == 0) { send_error(fd, 400, "missing input"); return; }

    model_t *m = s->m;
    jv *encoding = jv_get(req, "encoding_format");
    bool b64 = false;
    if (!absent(encoding)) {
        if (encoding->type != J_STR) {
            send_error(fd, 400, "encoding_format must be float or base64");
            return;
        }
        if (strcmp(encoding->str, "base64") == 0) b64 = true;
        else if (strcmp(encoding->str, "float") != 0) {
            send_error(fd, 400, "encoding_format must be float or base64");
            return;
        }
    }
    jv *dimensions = jv_get(req, "dimensions");
    if (!absent(dimensions) &&
        (dimensions->type != J_NUM || !isfinite(dimensions->num) ||
         dimensions->num != m->n_embd)) {
        send_error(fd, 400, "dimensions must equal the model embedding size");
        return;
    }
    float *emb = malloc(sizeof(float) * m->n_embd);
    if (!emb) { send_error(fd, 500, "out of memory"); return; }  // RNS-3
    sbuf r = {0};
    sb_lit(&r, "{\"object\":\"list\",\"data\":[");
    int total = 0;
    bool ok = true;
    // RNS-1: an error must be reported to the client only AFTER the device turn
    // is released. send_error() does a blocking socket write bounded by the 30s
    // SO_SNDTIMEO; doing it under dev_mu would stall decode_worker — and thereby
    // every other in-flight generation — behind one slow/dead embeddings client.
    // So the loop records (err_code, err_msg) and breaks; the send happens below,
    // off-lock, mirroring run_completion.
    int err_code = 0;
    const char *err_msg = NULL;
    // model_embed launches kernels and engine_reset touches this slot's KV —
    // the same device work every other handler serializes under dev_mu. Take the
    // device turn so an embeddings request cannot launch while decode_worker is
    // capturing a CUDA graph (hazard at dev_mu, ~line 596). Every exit from the
    // loop below is a break, so the single sched_prefill_end() after the loop
    // always runs — the lock is never left held.
    sched_prefill_begin();
    for (int k = 0; k < n_in && ok; k++) {
        const char *txt = one ? one : jv_str(input->items[k], NULL);
        if (!txt) { err_code = 400; err_msg = "input must be strings"; ok = false; break; }
        size_t cap = strlen(txt) + 16;
        int32_t *toks = malloc(sizeof(int32_t) * cap);
        int n = toks ? tok_encode(s->tok, txt, toks, (int)cap, true, true) : -1;
        if (n < 0) {
            free(toks);
            err_code = 500; err_msg = "out of memory tokenizing input";
            ok = false;
            break;
        }
        if (n == 0 || !model_embed(m, toks, n, emb)) {
            free(toks);
            err_code = 400;
            err_msg = n == 0 ? "empty input" : "input exceeds context window";
            ok = false;
            break;
        }
        free(toks);
        total += n;
        sb_fmt(&r, "%s{\"object\":\"embedding\",\"index\":%d,\"embedding\":",
               k ? "," : "", k);
        if (b64) {
            sb_lit(&r, "\"");
            sb_emb_b64(&r, emb, m->n_embd);
            sb_lit(&r, "\"");
        } else {
            sb_lit(&r, "[");
            for (int j = 0; j < m->n_embd; j++)
                sb_fmt(&r, "%s%.7g", j ? "," : "", emb[j]);
            sb_lit(&r, "]");
        }
        sb_lit(&r, "}");
    }
    // model_embed overwrote this slot's KV cache — invalidate the prefix cache
    engine_reset(&s->e);
    s->e.pos = 0;
    sched_prefill_end();
    if (!ok) {
        // deferred from the loop — sent here, off the device lock (RNS-1)
        send_error(fd, err_code, err_msg);
    } else {
        sb_lit(&r, "],\"model\":\"");
        sb_esc(&r, SV.model_name, strlen(SV.model_name));
        sb_fmt(&r, "\",\"usage\":{\"prompt_tokens\":%d,\"total_tokens\":%d}}",
               total, total);
        send_built(fd, &r);
        fprintf(stderr, "[slot %d] embeddings: %d input(s), %d tok\n",
                s->id, n_in, total);
    }
    free(r.s);
    free(emb);
}

// ---------------------------------------------------------------- http

// /health and /v1/models read only startup-immutable strings plus an atomic
// resident snapshot, so they are safe to answer from the accept thread with no lock
static void send_health(sock_t fd) {
    char b[640];
    int n, res = resident_load();
    // Inference requests in flight. "A model is loaded" and "the model is
    // working" look identical from outside the process, and the tray needs to
    // tell them apart to show the right glyph. This is the count the server
    // already keeps for swap and unload decisions, so reading it costs an
    // atomic load and adds nothing to the request path. /health is not itself
    // counted (the increment at dispatch covers the inference routes only), so
    // a poller never sees its own request here.
    int active = atomic_load(&SV.active_requests);
    // Process cost and cumulative work, for a supervisor budgeting several
    // runners. RSS is the process total -- weights, KV cache, activations and
    // allocator overhead -- which is the number a machine is sized against and
    // which no mapping-level measure accounts for. The token and second totals
    // are monotonic so a dashboard can difference them over its own window.
    unsigned long long wp, wg; double ws;
    server_work_totals(&wp, &wg, &ws);
    char m[256];
    snprintf(m, sizeof(m),
             ",\"rss_bytes\":%llu,\"peak_rss_bytes\":%llu,"
             "\"tokens_prompt\":%llu,\"tokens_generated\":%llu,"
             "\"generate_seconds\":%.6f",
             (unsigned long long)plat_proc_rss_bytes(),
             (unsigned long long)plat_proc_peak_rss_bytes(),
             wp, wg, ws);

    if (SV.n_reg > 0 && res >= 0) {
        char esc[192];
        json_escape(SV.reg[res].name, strlen(SV.reg[res].name), esc, sizeof(esc));
        n = snprintf(b, sizeof(b),
                     "{\"status\":\"ok\",\"resident\":\"%s\","
                     "\"active_requests\":%d%s}", esc, active, m);
    } else if (SV.n_reg > 0) {
        n = snprintf(b, sizeof(b), "{\"status\":\"ok\",\"resident\":null,"
                                   "\"active_requests\":%d%s}", active, m);
    } else {
        n = snprintf(b, sizeof(b),
                     "{\"status\":\"ok\",\"active_requests\":%d%s}",
                     active, m);
    }
    send_response(fd, 200, "application/json", b, n);
}

static void send_models(sock_t fd) {
    sbuf r = {0};
    sb_lit(&r, "{\"object\":\"list\",\"data\":[");
    if (SV.n_reg > 0) {
        for (int i = 0; i < SV.n_reg; i++) {
            char esc[192];
            json_escape(SV.reg[i].name, strlen(SV.reg[i].name), esc, sizeof(esc));
            sb_fmt(&r, "%s{\"id\":\"%s\",\"object\":\"model\","
                       "\"owned_by\":\"runner\"}", i ? "," : "", esc);
        }
    } else {
        char esc[256];
        json_escape(SV.model_name, strlen(SV.model_name), esc, sizeof(esc));
        sb_fmt(&r, "{\"id\":\"%s\",\"object\":\"model\",\"owned_by\":\"runner\"}", esc);
    }
    sb_lit(&r, "]}");
    send_built(fd, &r);
    free(r.s);
}

// Server-wide prefix-cache telemetry. Per-request telemetry says what one
// request saved; this says whether the cache is earning its memory —
// hit rate, resident bytes against the budget, and the prefill time it has
// avoided so far.
static void send_prefix_cache(sock_t fd) {
    prefix_cache_stats st;
    prefix_cache_stats_get(&st);
    char b[640];
    int n = snprintf(b, sizeof(b),
        "{\"object\":\"runner.prefix_cache\","
        "\"enabled\":%s,\"entries\":%d,\"bytes\":%llu,\"budget_bytes\":%llu,"
        "\"ttl_seconds\":%.1f,\"hits\":%llu,\"misses\":%llu,\"stores\":%llu,"
        "\"evictions\":%llu,\"tokens_reused\":%llu,"
        "\"saved_prefill_seconds\":%.6f,\"prefill_seconds_per_token\":%.9f}",
        st.budget ? "true" : "false", st.entries,
        (unsigned long long)st.bytes, (unsigned long long)st.budget, st.ttl,
        (unsigned long long)st.hits, (unsigned long long)st.misses,
        (unsigned long long)st.stores, (unsigned long long)st.evictions,
        (unsigned long long)st.tokens_reused,
        st.saved_prefill_s, st.cost_per_token_s);
    send_response(fd, 200, "application/json", b, n);
}

static void send_capabilities(sock_t fd) {
    sbuf r = {0};
    int res = resident_load();
    sb_lit(&r, "{\"object\":\"runner.capabilities\",\"swap\":");
    sb_lit(&r, SV.n_reg > 0 && !SV.single ? "true" : "false");
    sb_lit(&r, ",\"resident\":");
    if (SV.n_reg > 0 && res >= 0) {
        sb_lit(&r, "\"");
        sb_esc(&r, SV.reg[res].name, strlen(SV.reg[res].name));
        sb_lit(&r, "\"");
    } else {
        sb_lit(&r, "null");
    }
    sb_fmt(&r, ",\"context\":%d,\"models\":[", context_load());
    if (SV.n_reg > 0) {
        for (int i = 0; i < SV.n_reg; i++) {
            if (i) sb_lit(&r, ",");
            sb_lit(&r, "{\"id\":\"");
            sb_esc(&r, SV.reg[i].name, strlen(SV.reg[i].name));
            sb_fmt(&r, "\",\"resident\":%s}", i == res ? "true" : "false");
        }
    } else {
        sb_lit(&r, "{\"id\":\"");
        sb_esc(&r, SV.model_name, strlen(SV.model_name));
        sb_lit(&r, "\",\"resident\":true}");
    }
    model_t *pm = res >= 0 ? SV.slots[0].m : NULL;
    sb_lit(&r, "],\"agent_profile\":");
    if (!pm || !pm->agent_profile) {
        sb_lit(&r, "null");
    } else {
        sb_fmt(&r, "{\"protocol_version\":%u,\"tokenizer_version\":%u,\"schema_id\":\"",
               pm->agent_protocol_version, pm->agent_tokenizer_version);
        sb_esc(&r, pm->agent_schema_id, strlen(pm->agent_schema_id));
        sb_lit(&r, "\",\"schema_digest\":\"");
        sb_esc(&r, pm->agent_schema_digest, strlen(pm->agent_schema_digest));
        sb_lit(&r, "\",\"required_features\":[");
        for (uint64_t i = 0; i < pm->n_agent_required_features; i++) {
            if (i) sb_lit(&r, ",");
            sb_lit(&r, "\"");
            sb_esc(&r, pm->agent_required_features[i].s,
                   pm->agent_required_features[i].n);
            sb_lit(&r, "\"");
        }
        sb_lit(&r, "]}");
    }
    // Admitted-but-unconsumed MTP heads: a controller can see that the export
    // carries predictor blocks and that this build excluded them, rather than
    // inferring it from a layer count that silently differs from the card.
    sb_fmt(&r, ",\"mtp\":{\"declared_layers\":%d,\"consumed\":false}",
           pm ? pm->mtp_layers : 0);
    sb_lit(&r, ",\"sampling\":{\"preset\":");
    if (SV.preset_name) {
        sb_lit(&r, "\"");
        sb_esc(&r, SV.preset_name, strlen(SV.preset_name));
        sb_lit(&r, "\"");
    } else {
        sb_lit(&r, "null");  // swap mode before the first model is resident
    }
    {
        const sampler *d = &SV.slots[0].smp_base;
        sb_fmt(&r, ",\"temperature\":%.2f,\"top_p\":%.2f,\"top_k\":%d,"
                   "\"min_p\":%.2f,\"repeat_penalty\":%.2f}",
               (double)d->temp, (double)d->top_p, d->top_k,
               (double)d->min_p, (double)d->repeat_penalty);
    }
    sb_lit(&r, ",\"features\":{"
               "\"responses_api\":true,"
               "\"messages_api\":true,"
               "\"json_object\":true,"
               "\"json_schema\":true,"
               "\"stop_sequences\":true,"
               "\"schema_conditionals\":true,"
               "\"schema_string_bounds\":true,"
               "\"schema_integer_bounds\":true,"
               "\"request_telemetry\":true,"
               "\"prefix_cache\":true,"
               "\"prefix_cache_controls\":true,"
               "\"shared_prefix_cache\":true,"
               "\"forkable_prefixes\":true,"
               "\"repeat_penalty\":true,"
               "\"family_sampling_presets\":true}}");
    send_built(fd, &r);
    free(r.s);
}

static void handle_conn(slot_t *s, sock_t fd) {
    // a stalled or dead client must not pin an inference slot: the whole
    // request (header + body) has to arrive within this budget. Generation
    // time stays unbounded — the deadline only covers reading the request.
    //
    // The write side needs its own bound. A client that sends a valid request
    // and then stops reading fills the socket buffer, and an unbounded
    // blocking write parks the slot forever: the read deadline is already
    // satisfied and never fires again, so the slot is never returned.
    sock_send_timeout(fd, 30.0);
    double deadline = now_s() + 10.0;
    char hdr[16384];
    size_t got = 0;
    char *body_start = NULL;
    while (got < sizeof(hdr) - 1) {
        double remaining = deadline - now_s();
        if (remaining <= 0) { send_error(fd, 408, "request read timed out"); return; }
        sock_recv_timeout(fd, remaining);
        int r = sock_recv(fd, hdr + got, sizeof(hdr) - 1 - got);
        if (r <= 0) {
            // r == 0: orderly close, client is gone. r < 0: timeout (or a
            // socket error, where the 408 write fails harmlessly).
            if (r < 0) send_error(fd, 408, "request read timed out");
            return;
        }
        got += (size_t)r;
        hdr[got] = 0;
        if ((body_start = strstr(hdr, "\r\n\r\n")) != NULL) break;
    }
    if (!body_start) { send_error(fd, 400, "bad request"); return; }
    char *header_end = body_start;
    body_start += 4;

    char method[8] = {0}, path[256] = {0};
    char *first_header = NULL;
    if (!parse_request_line(hdr, method, path, &first_header)) {
        send_error(fd, 400, "malformed request line");
        return;
    }
    if (!validate_request_authority(first_header, header_end)) {
        send_error(fd, 403, "Host and Origin must be loopback");
        return;
    }
    // Route on the path component. SDKs use query parameters for protocol
    // feature selection — Claude Code, for example, sends
    // `/v1/messages?beta=true`. The query does not rename the resource and is
    // deliberately not interpreted by this stateless inference boundary.
    char *query = strchr(path, '?');
    if (query) *query = 0;

    size_t content_length = 0;
    if (!parse_request_framing(first_header, header_end, &content_length)) {
        send_error(fd, 400, "invalid request framing");
        return;
    }
    if (content_length > 32u * 1024 * 1024) {
        send_error(fd, 400, "body too large");
        return;
    }

    char *body = NULL;
    if (content_length > 0) {
        body = malloc(content_length + 1);
        if (!body) { send_error(fd, 500, "cannot allocate request body"); return; }
        size_t have = got - (size_t)(body_start - hdr);
        if (have > content_length) have = content_length;
        memcpy(body, body_start, have);
        while (have < content_length) {
            double remaining = deadline - now_s();
            if (remaining <= 0) {
                free(body);
                send_error(fd, 408, "request read timed out");
                return;
            }
            sock_recv_timeout(fd, remaining);
            int r = sock_recv(fd, body + have, content_length - have);
            if (r <= 0) {
                free(body);
                if (r < 0) send_error(fd, 408, "request read timed out");
                return;
            }
            have += (size_t)r;
        }
        body[content_length] = 0;
    }

    if (!strcmp(method, "POST") && !strcmp(path, "/unload")) {
        // Free the resident model's memory. Normally answered straight from
        // the accept loop (see accept_fastpath); this path still serves a
        // request that slipped past it.
        handle_unload(fd);
    } else if (!strcmp(method, "GET") && !strcmp(path, "/unload")) {
        // POST-only since 0.1.5-alpha, and the refusal is explicit rather
        // than a 404 so an operator with the old call sees what changed.
        //
        // It was a GET, which made it reachable from any web page the user
        // happened to be visiting: <img src="http://127.0.0.1:PORT/unload">
        // frees the model with no preflight, no CORS, and no rebinding needed,
        // because loopback binding does not stop a browser. A POST is not a
        // CORS simple request unless its Content-Type says so, so requiring
        // one restores the preflight that stands between a drive-by page and
        // a freed model. Host/Origin validation is the other half and is
        // tracked separately.
        send_error(fd, 405, "unload is POST-only: GET was reachable from any "
                            "web page via <img src>. Use POST /unload");
    } else if (!strcmp(method, "GET") &&
               !strcmp(path, "/v1/runner/prefix-cache")) {
        send_prefix_cache(fd);
    } else if (!strcmp(method, "POST") &&
               !strcmp(path, "/v1/runner/prefix-cache/clear")) {
        // Explicit release, for benchmarks that need a cold cache and for
        // operators reclaiming the memory without unloading the model.
        prefix_cache_clear();
        send_prefix_cache(fd);
    } else if (!strcmp(method, "GET") && !strcmp(path, "/health")) {
        send_health(fd);
    } else if (!strcmp(method, "GET") && !strcmp(path, "/v1/models")) {
        send_models(fd);
    } else if (!strcmp(method, "GET") && !strcmp(path, "/v1/capabilities")) {
        send_capabilities(fd);
    } else if (!strcmp(method, "POST") &&
               (!strcmp(path, "/v1/chat/completions") ||
                !strcmp(path, "/v1/responses") ||
                !strcmp(path, "/v1/messages") ||
                !strcmp(path, "/v1/messages/count_tokens") ||
                !strcmp(path, "/v1/completions") ||
                !strcmp(path, "/v1/embeddings"))) {
        jv *req = body ? json_parse(body, content_length) : NULL;
        if (!req) {
            send_error(fd, 400, "invalid JSON body");
        } else {
            bool has_keep_alive = false;
            int keep_alive = 0;
            if (!request_keep_alive(req, &has_keep_alive, &keep_alive)) {
                send_error(fd, 400, "keep_alive out of range");
                jv_free(req);
                free(body);
                return;
            }
            atomic_fetch_add(&SV.active_requests, 1);
            bool ok = true;
            if (SV.n_reg > 0) {
                int sw = swap_to(jv_str(jv_get(req, "model"), NULL));
                if (sw == SWAP_LOAD_FAILED) {
                    send_error(fd, 500,
                               "model failed to load (registered but broken; see server log)");
                    ok = false;
                } else if (sw == SWAP_ABORTED) {
                    send_error(fd, 503,
                               "model load abandoned (unload or shutdown requested; retry)");
                    ok = false;
                } else if (sw < 0) {
                    send_error_detail(fd, 404,
                                      "unknown model (see /v1/models)",
                                      "model", "model_not_found");
                    ok = false;
                }
            } else if (!validate_single_model_request(fd, req)) {
                ok = false;
            }
            if (ok) {
                if (strcmp(path, "/v1/chat/completions") == 0) handle_chat(s, fd, req);
                else if (strcmp(path, "/v1/responses") == 0) handle_responses(s, fd, req);
                else if (strcmp(path, "/v1/messages") == 0) handle_messages(s, fd, req);
                else if (strcmp(path, "/v1/messages/count_tokens") == 0)
                    handle_count_tokens(s, fd, req);
                else if (strcmp(path, "/v1/embeddings") == 0) handle_embeddings(s, fd, req);
                else handle_completion(s, fd, req);
                // Ollama-style keep_alive: seconds of idle before the model
                // unloads (swap mode) — 0 unloads now, negative pins forever
                if (has_keep_alive && SV.n_reg > 0) {
                    pthread_mutex_lock(&SV.swap_mu);
                    if (keep_alive == 0) unload_resident();
                    else SV.ttl = keep_alive < 0 ? 0 : keep_alive;
                    pthread_mutex_unlock(&SV.swap_mu);
                }
            }
            // Drop the request from the count BEFORE the bookkeeping lock:
            // an /unload that saw this request active left pending_unload for
            // us, and the count must already be zero when we honour it.
            atomic_fetch_sub(&SV.active_requests, 1);
            if (SV.n_reg > 0) {
                bool unloaded = false;
                pthread_mutex_lock(&SV.swap_mu);
                SV.last_used = now_s();
                if (SV.pending_unload && !SV.loading &&
                    !atomic_load(&SV.active_requests)) {
                    unload_draft();
                    unload_resident();
                    SV.pending_unload = false;
                    unloaded = true;
                }
                pthread_mutex_unlock(&SV.swap_mu);
                if (unloaded) prefix_cache_clear();
            }
            jv_free(req);
        }
    } else {
        send_error(fd, 404, "not found");
    }
    free(body);
}

static void *slot_worker(void *arg) {
    slot_t *s = arg;
    for (;;) {
        sock_t fd = q_pop();
        if (fd == SOCK_INVALID) return NULL;
        handle_conn(s, fd);
        sock_close(fd);
    }
}

// answer tiny requests from the accept loop: single-slot serving means one
// long generation used to block /health until the xyntetik watchdog declared a
// live runner "unhealthy: timed out". POST /unload is answered here too — it
// never frees anything a slot is using (handle_unload defers under an active
// load or generation), and an operator reclaiming memory must not queue behind
// the very work that holds it. Every other POST is handed to a slot untouched.
static bool accept_fastpath(sock_t fd) {
#ifndef _WIN32
    // POSIX fd_set is a fixed-size bitmask indexed by fd value; FD_SET on an
    // fd >= FD_SETSIZE is undefined behavior (out-of-bounds write). Windows
    // fd_set is a count-based array instead, so it isn't affected.
    if (fd >= FD_SETSIZE) return false;
#endif
    fd_set rs;
    struct timeval tv = { 0, 250000 }; // loopback data lands in <1ms
    FD_ZERO(&rs);
    FD_SET(fd, &rs);
    if (select(fd + 1, &rs, NULL, NULL, &tv) != 1) return false;
    char hdr[2048];
    int n = sock_peek(fd, hdr, 64);
    if (n <= 0) { sock_close(fd); return true; } // died before speaking
    hdr[n] = 0;
    // match the path plus the space HTTP/1.x always puts before the version,
    // so "GET /healthzzz" falls through to the slot path instead of being
    // misrouted here. Bare HTTP/0.9 "GET /health\r\n" (no version) won't
    // match, but neither curl nor the xyntetik watchdog send that, so it's
    // not worth the extra branch.
    bool health = !strncmp(hdr, "GET /health ", 12);
    bool models = !strncmp(hdr, "GET /v1/models ", 15);
    bool caps = !strncmp(hdr, "GET /v1/capabilities ", 21);
    bool unload = !strncmp(hdr, "POST /unload ", 13);
    // The old spelling still has to reach a handler, or an operator's script
    // gets a 404 that says nothing. It is not answered here — it falls through
    // to the slot path, which replies 405 with the reason.
    if (!strncmp(hdr, "GET /unload ", 12)) return false;
    if (!health && !models && !caps && !unload) return false;
    // Drain the request before replying: closing with unread bytes can RST
    // the connection and discard our response. But the accept thread must
    // never block indefinitely on a single connection — it's the only thing
    // calling accept(), so a stalled client here would queue every later
    // connection behind it, reproducing the watchdog-timeout bug this
    // fastpath exists to fix. Re-select before each recv with a short
    // timeout and cap the total drain time; if a client dribbles bytes too
    // slowly to finish the header in the budget, answer anyway (these GETs
    // are tiny and read-only, so a reply is always correct) and move on.
    double deadline = now_s() + 0.5;
    size_t got = 0;
    while (got < sizeof(hdr) - 1 && !strstr(hdr, "\r\n\r\n")) {
        double remaining = deadline - now_s();
        if (remaining <= 0) break;
        struct timeval dtv;
        dtv.tv_sec = 0;
        dtv.tv_usec = (long)(remaining * 1e6);
        if (dtv.tv_usec > 100000) dtv.tv_usec = 100000; // poll in <=100ms steps
        FD_ZERO(&rs);
        FD_SET(fd, &rs);
        if (select(fd + 1, &rs, NULL, NULL, &dtv) != 1) break;
        int r = sock_recv(fd, hdr + got, sizeof(hdr) - 1 - got);
        if (r <= 0) break;
        got += (size_t)r;
        hdr[got] = 0;
    }
    char *header_end = strstr(hdr, "\r\n\r\n");
    if (!header_end) {
        send_error(fd, 400, "bad request");
        sock_close(fd);
        return true;
    }
    char method[8] = {0}, path[256] = {0};
    char *first_header = NULL;
    size_t content_length = 0;
    if (!parse_request_line(hdr, method, path, &first_header) ||
        !parse_request_framing(first_header, header_end, &content_length) ||
        content_length > 32u * 1024 * 1024) {
        send_error(fd, 400, "invalid request framing");
        sock_close(fd);
        return true;
    }
    if (!validate_request_authority(first_header, header_end)) {
        send_error(fd, 403, "Host and Origin must be loopback");
        sock_close(fd);
        return true;
    }
    if (unload && content_length) {
        // The accept loop must not sit reading a body: it is the only thread
        // calling accept(), which is the whole reason this fastpath exists.
        // /unload takes no body, so a request with one is refused rather than
        // drained.
        send_error(fd, 400, "unload takes no request body");
        sock_close(fd);
        return true;
    }
    if (health) send_health(fd);
    else if (models) send_models(fd);
    else if (unload) handle_unload(fd);
    else        send_capabilities(fd);
    sock_close(fd);
    return true;
}

// ---------------------------------------------------------------- entry

#ifndef _WIN32
static volatile sig_atomic_t stop_requested;
static volatile sig_atomic_t listener_fd = -1;

void server_request_stop(void) {
    // A second signal is the operator overruling the drain: exit now. _exit is
    // async-signal-safe (128+SIGINT — the shell's convention for a Ctrl-C kill),
    // where the alternative on a pinned drain was reaching for SIGKILL.
    if (stop_requested) _exit(130);
    stop_requested = 1;
    int fd = (int)listener_fd;
    if (fd >= 0) {
        listener_fd = -1;
        // shutdown() BEFORE close(), and it is not belt-and-braces.
        //
        // A blocked accept() is woken by the signal only in the thread the
        // signal was delivered to, and a process may deliver SIGTERM to any
        // thread that has it unblocked — a slot worker, the decode thread, the
        // TTL reaper. Closing the descriptor from one of those does NOT wake a
        // thread already parked in accept() on Linux; it stays there until a
        // connection happens to arrive. Observed directly: with the accept loop
        // on a non-main thread, /proc showed it in inet_csk_accept long after
        // the handler had run and closed the fd.
        //
        // shutdown() does wake it, and both calls are async-signal-safe.
        shutdown(fd, SHUT_RDWR);
        close(fd);
    }
}

static void stop_handler(int sig) {
    (void)sig;
    server_request_stop();
}

static void install_stop_handlers(void) {
    struct sigaction sa = {0};
    sa.sa_handler = stop_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
}

// Between the handlers installing and the listener publishing there is nothing
// a signal can close, so startup polls this at its long stops (model loads)
// and abandons the launch instead of serving a request nobody wants anymore.
static bool stop_was_requested(void) { return stop_requested != 0; }
#else
// Windows port of the same design: stop flag + listener close + drain +
// second-signal escalation, via SetConsoleCtrlHandler. The handler runs on
// its own thread (not an async-signal context), so plain volatile stores and
// closesocket() are safe here.
static volatile LONG win_stop_requested;
static volatile SOCKET win_listener_socket = INVALID_SOCKET;

void server_request_stop(void) {
    if (InterlockedExchange(&win_stop_requested, 1)) _exit(130);
    SOCKET s = win_listener_socket;
    if (s != INVALID_SOCKET) {
        win_listener_socket = INVALID_SOCKET;
        closesocket(s); // wakes accept()
    }
}

static BOOL WINAPI win_stop_handler(DWORD ctrl_type) {
    switch (ctrl_type) {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
    case CTRL_CLOSE_EVENT:
    case CTRL_SHUTDOWN_EVENT: {
        // A second Ctrl-C is the operator overruling the drain: exit now,
        // with the shell's 128+SIGINT convention for parity with POSIX.
        server_request_stop();
        if (ctrl_type == CTRL_CLOSE_EVENT || ctrl_type == CTRL_SHUTDOWN_EVENT) {
            // Returning from these lets Windows terminate the process
            // immediately; hold the handler thread briefly so the drain in
            // main gets its window (Windows grants ~5s on close).
            Sleep(4000);
        }
        return TRUE; // handled: keep running so the drain can finish
    }
    default:
        return FALSE;
    }
}

static void install_stop_handlers(void) {
    win_stop_requested = 0;
    win_listener_socket = INVALID_SOCKET;
    SetConsoleCtrlHandler(win_stop_handler, TRUE);
}

static bool stop_was_requested(void) { return win_stop_requested != 0; }
#endif

int server_run(model_t *base, tokenizer *tok, const char *model_path,
               const model_params *mp, sampler defaults,
               const sampler_override *ov, int port, int parallel,
               int n_threads, int ttl, const char *draft_path, int draft_k,
               bool ignore_eos, int tmpl_override) {
    sock_init();
    // The shared server state gets a lifetime, and it is this call. Everything
    // below sets fields on SV and the teardown at the bottom releases them, but
    // until now nothing reset the ones that are only ever *set* — and there are
    // several: `q.shutdown` and `shutdown` are raised during teardown and never
    // lowered, `load_cancel` is left at 1, and `reaper_started` stays true
    // alongside a `reaper_th` whose thread has already been joined.
    //
    // A second server_run therefore did not work. Its slot workers saw a
    // shut-down queue and exited immediately, so the server listened, accepted
    // connections and answered nothing but `/health` — which is served on the
    // accept path and needs no worker, and is exactly why a casual check
    // looked fine. Teardown then joined an already-joined thread handle.
    //
    // This is what the RNR-019 finding meant by global state making
    // initialization and teardown hard: nothing ever asked the state to come
    // back, so the asymmetry could not be observed. tests/test_server_restart.c
    // now asks, twice. Note this makes the state's lifetime explicit, NOT
    // per-instance — two servers in one process would still share it. That
    // half is filed; nothing needs it today, and it costs threading a context
    // through six translation units.
    memset(&SV, 0, sizeof(SV));
#ifndef _WIN32
    stop_requested = 0;
    listener_fd = -1;
#endif
    install_stop_handlers(); // resets the stop flag + listener on both platforms
    SV.ignore_eos = ignore_eos;
    // Set before anything can load a model: registry.c reads it on every swap
    // and reload, so a forced template survives /unload and --ttl instead of
    // being detected away at the next request.
    SV.tmpl_override = tmpl_override;
    if (ov) SV.ov = *ov;
    // `defaults` arrives already resolved against the preloaded model; in swap
    // mode there is no model yet and swap_to resolves per load
    if (base) {
        char ident[256];
        sampler_ident(gguf_get_str(&base->gf, "general.name", NULL),
                      base->path, ident, sizeof(ident));
        SV.preset_name = sampler_preset_for(base->arch, ident)->name;
    } else {
        SV.preset_name = NULL;
    }
    if (parallel < 1) parallel = 1;
    if (parallel > 16) parallel = 16;
    // One named model is not a swap set, and asking for a name should not
    // silently cost the caller their slots.
    //
    // `-m llama=path` is the documented way to pin the /v1/models id for a
    // client config, and a Continue user did exactly that and lost
    // --parallel 2 without ever asking for swapping. Registry mode genuinely
    // is single-slot — ensure_resident() loads into slots[0] alone — but with
    // one entry there is nothing to swap *to*, so the honest trade is to give
    // up the registry rather than the slots. Plain `-m path --parallel 2`
    // already forgoes the registry for the same reason, so this lands in an
    // existing configuration rather than a new one.
    //
    // Done HERE, ahead of swap_mode, so everything downstream that keys off it
    // — the --ttl default, the --draft refusal, the registry setup — sees the
    // truth rather than being unwound afterwards.
    char single_path[sizeof(SV.reg[0].path)];
    const char *forced_name = NULL;
    const char *eq1 = strchr(model_path, '=');
    if (eq1 && parallel > 1 && !strchr(model_path, ',') &&
        eq1 != model_path && eq1[1] &&
        (size_t)(eq1 - model_path) < sizeof(SV.reg[0].name) &&
        strlen(eq1 + 1) < sizeof(single_path)) {
        static char single_name[sizeof(SV.reg[0].name)];
        snprintf(single_name, sizeof(single_name), "%.*s",
                 (int)(eq1 - model_path), model_path);
        snprintf(single_path, sizeof(single_path), "%s", eq1 + 1);
        forced_name = single_name;
        model_path  = single_path;
        fprintf(stderr,
                "note: one model named '%s' with --parallel %d — serving it on"
                " %d slots. /unload and --ttl are swap-mode features and need"
                " more than one model.\n",
                forced_name, parallel, parallel);
    }

    bool swap_mode = strchr(model_path, '=') != NULL;
    if (ttl < 0) ttl = swap_mode ? 300 : 0; // single-model default: never unload
    if (draft_path && swap_mode) {
        fprintf(stderr, "note: --draft needs a single served model — "
                "ignoring it in swap mode\n");
        draft_path = NULL;
    }

    // "name=path,name2=path2" enables swap mode: one resident model,
    // loaded per request's "model" field, unloaded after ttl idle seconds
    if (strchr(model_path, '=')) {
        // Validate every limit and reject the whole spec with an exact reason
        // rather than silently truncating a name/path (which can collide or
        // select the wrong file) or dropping entries past the cap (RNR-014).
        const int max_reg = (int)(sizeof(SV.reg) / sizeof(SV.reg[0]));
        char tmp[4096];
        if (strlen(model_path) >= sizeof(tmp)) {
            fprintf(stderr, "error: -m registry spec is too long (max %zu bytes)\n",
                    sizeof(tmp) - 1);
            return 1;
        }
        snprintf(tmp, sizeof(tmp), "%s", model_path);
        for (char *tk = strtok(tmp, ","); tk; tk = strtok(NULL, ",")) {
            if (SV.n_reg >= max_reg) {
                fprintf(stderr, "error: too many models in -m (max %d)\n", max_reg);
                return 1;
            }
            char *eq = strchr(tk, '=');
            if (!eq) { fprintf(stderr, "error: bad registry entry '%s' (want name=path)\n", tk); return 1; }
            *eq = 0;
            const char *nm = tk, *pth = eq + 1;
            if (!*nm || !*pth) {
                fprintf(stderr, "error: registry entry '%s=%s' has an empty name or path\n", nm, pth);
                return 1;
            }
            if (strlen(nm) >= sizeof(SV.reg[0].name)) {
                fprintf(stderr, "error: model name '%s' is too long (max %zu chars)\n",
                        nm, sizeof(SV.reg[0].name) - 1);
                return 1;
            }
            if (strlen(pth) >= sizeof(SV.reg[0].path)) {
                fprintf(stderr, "error: model path for '%s' is too long (max %zu chars)\n",
                        nm, sizeof(SV.reg[0].path) - 1);
                return 1;
            }
            for (int j = 0; j < SV.n_reg; j++)
                if (!strcmp(SV.reg[j].name, nm)) {
                    fprintf(stderr, "error: duplicate model name '%s' in -m\n", nm);
                    return 1;
                }
            snprintf(SV.reg[SV.n_reg].name, sizeof(SV.reg[0].name), "%s", nm);
            snprintf(SV.reg[SV.n_reg].path, sizeof(SV.reg[0].path), "%s", pth);
            if (!plat_file_readable(SV.reg[SV.n_reg].path)) {
                fprintf(stderr, "error: cannot read %s\n", SV.reg[SV.n_reg].path);
                return 1;
            }
            SV.n_reg++;
        }
        // A real swap set is several models, each detecting its own template.
        // --chat-template names ONE and has no per-model spelling, so applying
        // it across the set would be right for at most one member and would
        // silently mis-render the rest. Refused rather than applied to all or
        // dropped for all: the caller asked for something this shape of
        // invocation cannot mean. Checked HERE and not in main.c because this
        // is where the entry count is actually known -- `-m name=path` on its
        // own is a one-entry registry (it pins the /v1/models id) and takes
        // the override perfectly well.
        if (SV.n_reg > 1 && tmpl_override >= 0) {
            fprintf(stderr,
                    "error: --chat-template cannot be used with a swap set "
                    "(-m \"name=path,name2=path2\"): it names one template and "
                    "the set holds %d models, each detecting its own. Serve "
                    "that model on its own instance to force its template.\n",
                    SV.n_reg);
            return 1;
        }
        if (parallel > 1) {
            fprintf(stderr, "note: model swapping uses a single inference slot; "
                    "ignoring --parallel %d\n", parallel);
        }
        parallel = SV.n_reg > 0 ? 1 : parallel;
        resident_store(-1);
    }

    int threads_per_slot = n_threads / parallel;
    if (threads_per_slot < 1) threads_per_slot = 1;

    const char *name = strrchr(model_path, '/');
    const char *bsname = strrchr(model_path, '\\'); // Windows path separator
    if (bsname && (!name || bsname > name)) name = bsname;
    SV.model_name = SV.n_reg > 0 ? SV.reg[0].name
                  : forced_name  ? forced_name
                                 : (name ? name + 1 : model_path);
    SV.n_predict_cap = 1024;
    SV.q.limit = (int)(sizeof(SV.q.fds) / sizeof(sock_t));
    // a queue bound may only lower the fixed fd capacity, never raise it
    SV.q.limit = (int)env_i64("RUNNER_MAX_QUEUE", 1, SV.q.limit, SV.q.limit);
    // a bad timeout must not silently become 0 (which disables the deadline)
    SV.req_timeout = env_f64("RUNNER_REQUEST_TIMEOUT", 0.0, 1e9, SV.req_timeout);
    // Shared, forkable prompt prefixes. Sized in host RAM rather than as a
    // fraction of anything, because it is the one cache whose useful size is
    // set by the *traffic* (how many distinct system/tool/schema blocks the
    // agents on this box use) and not by the model.
    {
        uint64_t mb  = env_u64("RUNNER_PREFIX_CACHE_MB", 0, 1u << 20, 512);
        double   ttl = env_f64("RUNNER_PREFIX_CACHE_TTL", 0.0, 1e9, 600.0);
        prefix_cache_configure((size_t)mb * 1024 * 1024, ttl);
    }
    context_store(base ? base->n_ctx : mp->n_ctx);
    SV.n_slots = parallel;
    SV.slots = calloc(parallel, sizeof(slot_t));
    if (!SV.slots) {
        fprintf(stderr, "error: cannot allocate server slots\n");
        return 1;
    }
    if (pthread_mutex_init(&SV.q.mu, NULL) != 0) {
        fprintf(stderr, "error: cannot initialize server queue mutex\n");
        return 1;
    }
    if (pthread_cond_init(&SV.q.cv, NULL) != 0) {
        fprintf(stderr, "error: cannot initialize server queue condition\n");
        pthread_mutex_destroy(&SV.q.mu);
        return 1;
    }

    if (SV.n_reg > 0) {
        // swap mode: models are loaded on demand
        slot_t *s = &SV.slots[0];
        s->id = 0;
        s->smp = defaults;
        s->smp_base = defaults;
        if (!init_swap_runtime(mp, n_threads, ttl)) return 1;
    } else {
        int tmpl = SV.tmpl_override >= 0
                 ? SV.tmpl_override
                 : template_detect(gguf_get_str(&base->gf,
                                                "tokenizer.chat_template", NULL),
                                   tok);
        // Announced, because a forced template changes every prompt this
        // server renders and the operator asked for it explicitly. Detection
        // stays quiet: it is the default and nobody chose it.
        if (SV.tmpl_override >= 0)
            fprintf(stderr, "chat template: %s (forced by --chat-template)\n",
                    template_name(tmpl));
        model_params slot_mp = *mp;
        slot_mp.verbose = false;
        slot_mp.n_threads = threads_per_slot;
        // a reservation is a budget for the server, and every slot pays its
        // own KV cache out of it -- see the auto-fit in model_alloc_runtime
        slot_mp.n_seq = parallel;
        // each slot gets its share of the CPU-forced fallback cap too
        slot_mp.cpu_fallback_threads = mp->cpu_fallback_threads / parallel;

        for (int i = 0; i < parallel; i++) {
            // a Ctrl-C during a multi-slot load means "don't start": honour it
            // between loads rather than serving with the flag silently set
            if (stop_was_requested()) {
                fprintf(stderr, "shutdown requested during startup — exiting\n");
                return 0;
            }
            slot_t *s = &SV.slots[i];
            s->id = i;
            s->tok = tok;
            s->tmpl = tmpl;
            s->smp = defaults;
            s->smp.rng = defaults.rng ^ (0x9E3779B97F4A7C15ull * (unsigned)(i + 1));
            s->smp_base = s->smp;
            if (i == 0) {
                s->m = base;
                // mirror model_load's CPU-forced bump: this replacement pool
                // would otherwise silently undo it for slot 0
                int slot_threads = threads_per_slot;
                if (base->qwen35 &&
                    slot_mp.cpu_fallback_threads > slot_threads)
                    slot_threads = slot_mp.cpu_fallback_threads;
                tpool *replacement = tpool_create(slot_threads);
                if (!replacement) {
                    fprintf(stderr, "error: cannot create pool for slot 0\n");
                    return 1;
                }
                tpool_destroy(base->tp); // replace the single-thread load pool
                base->tp = replacement;
            } else {
                s->m = calloc(1, sizeof(model_t));
                if (!s->m) {
                    fprintf(stderr, "error: cannot allocate slot %d model\n", i);
                    return 1;
                }
                if (!model_load(s->m, model_path, &slot_mp)) {
                    fprintf(stderr, "error: failed to load slot %d\n", i);
                    return 1;
                }
            }
            if (!engine_init(&s->e, s->m, s->tok, &s->smp)) {
                fprintf(stderr, "error: out of memory initializing slot %d engine\n", i);
                return 1;
            }
            // CLI generation applies this after engine_init; server slots are
            // separate engines and must receive the same process-level flag.
            s->e.ignore_eos = ignore_eos;
            if (draft_path) {
                // per-slot draft context: each slot owns a full draft KV;
                // weights dedupe through the page cache like slot models
                s->e.dm = spec_draft_load(draft_path, s->m, &slot_mp);
                if (s->e.dm) s->e.draft_k = draft_k;
            }
        }

        if (parallel == 1) {
            // join the registry machinery so POST /unload frees the resident
            // model (the next request lazily reloads it) and --ttl works.
            // slot 0's containers are the caller's; borrowed avoids freeing
            // them on the first unload
            snprintf(SV.reg[0].name, sizeof(SV.reg[0].name), "%s", SV.model_name);
            snprintf(SV.reg[0].path, sizeof(SV.reg[0].path), "%s", model_path);
            SV.reg[0].tmpl = tmpl;
            SV.model_name = SV.reg[0].name;
            SV.n_reg = 1;
            SV.single = true;
            SV.borrowed = true;
            resident_store(0);
            SV.last_used = now_s();
            SV.draft = SV.slots[0].e.dm;
            SV.draft_k = draft_k;
            // a draft the gates rejected at startup stays rejected: only a
            // draft that actually served is worth reloading after /unload
            if (SV.draft) SV.draft_path = draft_path;
            if (!init_swap_runtime(mp, threads_per_slot, ttl)) return 1;
        }
    }

    // last long stop is behind us; a signal from here on is either caught
    // right now or by the published listener below
    if (stop_was_requested()) {
        fprintf(stderr, "shutdown requested during startup — exiting\n");
        return 0;
    }

    sock_t lfd = socket(AF_INET, SOCK_STREAM, 0);
    if (lfd == SOCK_INVALID) {
        fprintf(stderr, "error: cannot create server socket\n");
        return 1;
    }
    int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, (const char *)&one, sizeof(one));
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((uint16_t)port);
    if (bind(lfd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        fprintf(stderr, "error: cannot bind 127.0.0.1:%d (%s)\n", port, sock_errstr());
        sock_close(lfd);
        return 1;
    }
    if (listen(lfd, 64) != 0) {
        fprintf(stderr, "error: cannot listen on 127.0.0.1:%d (%s)\n", port,
                sock_errstr());
        sock_close(lfd);
        return 1;
    }
#ifndef _WIN32
    listener_fd = lfd;
#else
    win_listener_socket = lfd;
#endif

    // Every slot's model exists now, which is the only precondition the batch
    // has. Declining is not an error: sched_generate then runs the untouched
    // solo path, which is what a single slot, swap mode, or a backend without
    // batched kernels all get.
    bool batched = sched_start();

    for (int i = 0; i < parallel; i++) {
        if (pthread_create(&SV.slots[i].th, NULL, slot_worker, &SV.slots[i]) != 0) {
            fprintf(stderr, "error: cannot start server slot %d\n", i);
            sock_close(lfd);
            return 1;
        }
    }

    if (SV.n_reg > 0 && !SV.single)
        fprintf(stderr,
                "server listening on http://127.0.0.1:%d — %d models, swap on demand"
                " (ttl %ds)\n"
                "  POST /v1/chat/completions | POST /v1/responses | POST /v1/completions\n"
                "  GET /v1/models | GET /v1/capabilities | GET /health\n",
                port, SV.n_reg, SV.ttl);
    else
        fprintf(stderr,
                "server listening on http://127.0.0.1:%d — %d slot%s x %d threads%s\n"
                "  POST /v1/chat/completions | POST /v1/responses | POST /v1/completions\n"
                "  GET /v1/models | GET /v1/capabilities | GET /health\n",
                port, parallel, parallel > 1 ? "s" : "", threads_per_slot,
                batched ? ", continuous batching" : "");

    // Say it in the banner, not only at init.
    //
    // A backend that cannot run this model's exact expert variant prints its
    // reason from gpu_init(), which scrolls past above the load line and is
    // easy to miss. The banner is the last thing on screen when a server comes
    // up, so the CPU-expert placement belongs here too.
    if (SV.n_slots > 0 && SV.slots[0].m && SV.slots[0].m->n_expert > 0 &&
        !SV.slots[0].m->gpu)
        fprintf(stderr,
                "  note: this is a sparse-MoE model and its experts are running"
                " on the CPU — expect a fraction of dense throughput\n");

    for (;;) {
        // covers the race where the signal landed after the socket-creation
        // check but before listener_fd published: the handler had no fd to
        // close, so accept() would block forever with the flag already set.
        // A signal landing after this check finds listener_fd published and
        // closes it, so accept() fails; no window remains.
        if (stop_was_requested()) break;
        sock_t cfd = accept(lfd, NULL, NULL);
        if (cfd == SOCK_INVALID) {
#ifndef _WIN32
            if (stop_requested) break;
#else
            if (win_stop_requested) break;
#endif
            if (errno == EINTR) continue;
            break;
        }
        if (!accept_fastpath(cfd)) q_push(cfd);
    }
    // Stop admission first, fail work that has not started, then allow active
    // requests to finish before dismantling the scheduler and model state.
#ifndef _WIN32
    if (listener_fd >= 0) {
        listener_fd = -1;
        sock_close(lfd);
    }
#else
    if (win_listener_socket != INVALID_SOCKET) {
        win_listener_socket = INVALID_SOCKET;
        sock_close(lfd);
    }
#endif
    queue_shutdown();
    // A worker parked in a --wait-for-vram queue would pin the joins below
    // for up to the full wait; the load gives up at its next poll instead.
    atomic_store(&SV.load_cancel, 1);
    for (int i = 0; i < parallel; i++) pthread_join(SV.slots[i].th, NULL);
    sched_shutdown();
    atomic_store(&SV.shutdown, true);
    if (SV.reaper_started) pthread_join(SV.reaper_th, NULL);

    for (int i = 0; i < parallel; i++) free(SV.slots[i].e.hist);
    if (SV.n_reg > 0) {
        pthread_mutex_lock(&SV.swap_mu);
        unload_draft();
        unload_resident();
        pthread_mutex_unlock(&SV.swap_mu);
        pthread_mutex_destroy(&SV.swap_mu);
    } else {
        tokenizer_free(tok);
        for (int i = 0; i < parallel; i++) {
            model_t *draft = SV.slots[i].e.dm;
            if (draft) { model_free(draft); free(draft); }
            model_free(SV.slots[i].m);
            if (i > 0) free(SV.slots[i].m);
        }
    }
    prefix_cache_clear();
    pthread_cond_destroy(&SV.q.cv);
    pthread_mutex_destroy(&SV.q.mu);
    free(SV.slots);
    return 0;
}
