# Server hardening + CUDA tuning — implementation plan (2026-07-14)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute this plan task-by-task, with a fresh subagent per task and verification (build + smoke) before every commit.

**Goal**: Four independently-landable improvements to the runner: (1) `/health` and `/v1/models` answered from the accept loop so a busy inference slot never makes the gridcore watchdog call a live runner "unhealthy: timed out"; (2) a `--parent-pid N` flag that ties the runner's lifetime to its supervisor so SIGKILLed gridcore-clu processes stop leaking orphaned runners; (3) server-side speculative decoding via per-slot draft contexts, for unconstrained requests only (the existing schema/JSON self-disable in `engine_generate` is preserved untouched; constraint-state rewind is explicitly out of scope); (4) measurement-driven CUDA kernel tuning — CUDA graphs for the decode loop and vectorized loads in the hottest kernels, each kept only if a reproducible benchmark improves and temp-0 output stays byte-identical.

**Architecture**: Plain-C GGUF inference engine (~8k lines, single `make` compile of all `src/*.c`). `src/server.c` runs an accept loop (`server_run`, src/server.c:699-852) pushing accepted fds onto a mutex/condvar `fdqueue`; N `slot_worker` threads pop fds and run `handle_conn` → `run_completion` → `engine_generate` (src/engine.c:245). Speculative decoding lives in `engine_generate_spec` (src/engine.c:143-243) driven by `engine.dm/dpos/draft_k` (src/runner.h:472-474), with the target's batched verify in `model_forward_batch_keep`/`model_spec_row_logits` (src/model.c:777-803). The CUDA backend (src/cuda.c) uses the driver API loaded from nvcuda.dll, JIT-compiling PTX embedded in `src/kernels_ptx.h` (generated from `src/kernels.cu` by `make ptx` + `scripts/embed-ptx.py`, which pins `.version 7.8`).

**Tech Stack**: C (gnu11), gcc via MSYS2 UCRT64 on Windows (also Linux/macOS in CI), pthreads (winpthreads statically linked), winsock2/BSD sockets, CUDA driver API + PTX (no toolkit needed at runtime), Python 3 for CI helper scripts, GitHub Actions CI (`.github/workflows/ci.yml`) with end-to-end smoke tests only (no unit test suite).

## Global Constraints

- **Build**: from an MSYS2 UCRT64 shell (`C:\msys64\ucrt64.exe`, or Git Bash with `export PATH=/c/msys64/ucrt64/bin:$PATH`), run `make` in `/c/ProjectGrid/Runner`. Produces `runner.exe` at repo root. `make debug` builds the ASan/UBSan binary (Linux CI only; ASan is unavailable under MinGW — do not gate local work on it).
- **Test model**: `test.gguf` exists at repo root; regenerate anytime with `python scripts/make-test-model.py test.gguf`.
- **Smoke tests locally**: there is no test runner; the CI steps in `.github/workflows/ci.yml` ARE the test suite. After every task run at minimum, in MSYS2 bash at the repo root:
  ```sh
  ./runner.exe -m test.gguf -p "hello" -n 8 --temp 0
  ./runner.exe -m test.gguf -p "hi" -n 24 --temp 0 --json 2>/dev/null | python -c "import json,sys; json.load(sys.stdin); print('valid json')"
  ./runner.exe -m test.gguf --serve --port 8123 --parallel 2 &
  sleep 2
  curl -sf http://127.0.0.1:8123/health && curl -sf http://127.0.0.1:8123/v1/models
  curl -sf -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0}' http://127.0.0.1:8123/v1/chat/completions | python -c "import json,sys; json.load(sys.stdin); print('server ok')"
  kill %1
  ```
  plus the task-specific checks defined below.
- **Style** (gleaned from the codebase — match it exactly): 4-space indent, ~90-col lines, `snake_case`, file-local helpers are `static`, section banners `// ---------------------------------------------------------------- name`, comments explain *why* (often referencing the failure they prevent), errors go to `fprintf(stderr, ...)` followed by graceful fallback rather than abort, platform splits via `#ifdef _WIN32` with small same-signature wrappers (see src/server.c:29-50, src/compat.c). No new dependencies.
- **Commits**: message style from `git log` is `area: lowercase summary` (`server:`, `engine:`, `gpu:`, `compat:`, `scripts:`, `docs:`, `fix:`), a body explaining why (wrapped ~72 cols), and the trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. The repo is on `main`; before the first task run `git checkout -b server-hardening-and-cuda`. One commit per task (Task 4: one commit per kept experiment).
- **Read-only invariants**: never edit `src/kernels_ptx.h` by hand (generated); keep `MVB` in cuda.c and kernels.cu in sync; the PTX `.version` pin in `scripts/embed-ptx.py` must survive regeneration.

---

### Task 1: /health and /v1/models bypass the slot queue

A long in-flight generation occupies the only slot worker; `GET /health` sits in the `fdqueue` behind it and the gridcore watchdog times out. Answer `/health` and `/v1/models` inline in the accept loop (both are read-only over `SV`), leaving `/unload` on the slot path on purpose — it frees the resident model and must stay serialized with generation.

**Files**
- Modify `C:\ProjectGrid\Runner\src\server.c` — add `sock_peek` next to the `sock_recv` wrappers (Windows branch after line 38, POSIX branch after line 47, plus `#include <sys/select.h>` in the POSIX block near line 45); extract `send_health`/`send_models` from `handle_conn` (currently inline at lines 616-629 and 630-647); add `accept_fastpath` above `server_run` (~line 697); call it in the accept loop (lines 843-850).
- Modify `C:\ProjectGrid\Runner\.github\workflows\ci.yml` — new smoke step in the `unix` job after "smoke test (server)" (after line 55).
- Modify `C:\ProjectGrid\Runner\docs\ROADMAP.md` — remove the "/health should bypass the slot queue" bullet (lines under `## Server`).

**Interfaces**
- `static int sock_peek(int fd, char *buf, size_t n)` — `recv(..., MSG_PEEK)`, both platforms.
- `static void send_health(int fd)` / `static void send_models(int fd)` — exact bodies moved from `handle_conn`; called from both `handle_conn` (fallback when the fast path misses) and `accept_fastpath`.
- `static bool accept_fastpath(int fd)` — returns true when the connection was fully handled (responded or found dead) and closed; false means "hand it to the slot queue untouched".
- Concurrency contract: `SV.reg[*].name` and `SV.model_name` are written only before slot threads start (src/server.c:718, 746, 796-799); `SV.resident` is a plain aligned int snapshot — the accept-thread read is a benign race, documented in a comment.

**Steps**

- [ ] Write the failing repro. In MSYS2 bash at `/c/ProjectGrid/Runner`:
  ```sh
  ./runner.exe -m test.gguf --serve --port 8126 --parallel 1 &
  sleep 2
  python -c "import socket,time; s=socket.create_connection(('127.0.0.1',8126)); s.sendall(b'POST /v1/chat/completions HTTP/1.1\r\n'); time.sleep(30)" &
  sleep 1
  curl -sf -m 3 http://127.0.0.1:8126/health && echo HEALTH-OK || echo HEALTH-BLOCKED
  kill %1 %2
  ```
  Expected now: `HEALTH-BLOCKED` (the stalled POST pins slot 0; curl times out).
- [ ] Add the same check to CI so it stays fixed — in `.github/workflows/ci.yml`, `unix` job, after the "smoke test (server)" step:
  ```yaml
        - name: smoke test (health bypasses a busy slot)
          # a stalled request pins the only slot; /health and /v1/models must
          # still answer from the accept loop (gridcore watchdog regression)
          run: |
            ./runner -m test.gguf --serve --port 8126 --parallel 1 &
            sleep 2
            python3 -c "import socket,time; s=socket.create_connection(('127.0.0.1',8126)); s.sendall(b'POST /v1/chat/completions HTTP/1.1\r\n'); time.sleep(20)" &
            sleep 1
            curl -sf -m 3 http://127.0.0.1:8126/health
            curl -sf -m 3 http://127.0.0.1:8126/v1/models
            kill %1 %2 2>/dev/null || true
            echo "health bypass ok"
  ```
- [ ] Add `sock_peek` to both platform blocks in src/server.c:
  ```c
  static int  sock_peek(int fd, char *buf, size_t n) { return recv(fd, buf, (int)n, MSG_PEEK); }
  ```
  (identical line in both branches except the POSIX one casts `(int)recv(fd, buf, n, MSG_PEEK)`), and add `#include <sys/select.h>` to the POSIX include block (after `#include <arpa/inet.h>`, line 43).
- [ ] Extract the two handlers. Cut the bodies out of `handle_conn` (lines 617-629 and 631-647) into file-scope functions placed just above `handle_conn`:
  ```c
  // /health and /v1/models read only startup-immutable strings plus the
  // SV.resident int (snapshot; a torn read is impossible for an aligned int),
  // so they are safe to answer from the accept thread with no lock
  static void send_health(int fd) {
      char b[384];
      int n, res = SV.resident;
      if (SV.n_reg > 0 && res >= 0) {
          char esc[192];
          json_escape(SV.reg[res].name, strlen(SV.reg[res].name), esc, sizeof(esc));
          n = snprintf(b, sizeof(b), "{\"status\":\"ok\",\"resident\":\"%s\"}", esc);
      } else if (SV.n_reg > 0) {
          n = snprintf(b, sizeof(b), "{\"status\":\"ok\",\"resident\":null}");
      } else {
          n = snprintf(b, sizeof(b), "{\"status\":\"ok\"}");
      }
      send_response(fd, 200, "application/json", b, n);
  }

  static void send_models(int fd) {
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
      send_response(fd, 200, "application/json", r.s, r.n);
      free(r.s);
  }
  ```
  In `handle_conn`, replace the two blocks with `send_health(fd);` and `send_models(fd);` (keeping the `else if` routing intact — the slot path remains the fallback for requests whose header arrives late).
- [ ] Add the fast path above `server_run`:
  ```c
  // answer tiny read-only GETs from the accept loop: single-slot serving means
  // one long generation used to block /health until the gridcore watchdog
  // declared a live runner "unhealthy: timed out". POSTs (and /unload, which
  // frees the resident model and must stay serialized with generation) are
  // handed to a slot untouched.
  static bool accept_fastpath(int fd) {
      fd_set rs;
      struct timeval tv = { 0, 250000 }; // loopback data lands in <1ms
      FD_ZERO(&rs);
      FD_SET(fd, &rs);
      if (select(fd + 1, &rs, NULL, NULL, &tv) != 1) return false;
      char hdr[2048];
      int n = sock_peek(fd, hdr, 16);
      if (n <= 0) { sock_close(fd); return true; } // died before speaking
      hdr[n] = 0;
      bool health = !strncmp(hdr, "GET /health", 11);
      bool models = !strncmp(hdr, "GET /v1/models", 14);
      if (!health && !models) return false;
      // drain the request before replying: closing with unread bytes can RST
      // the connection and discard our response
      size_t got = 0;
      while (got < sizeof(hdr) - 1 && !strstr(hdr, "\r\n\r\n")) {
          int r = sock_recv(fd, hdr + got, sizeof(hdr) - 1 - got);
          if (r <= 0) break;
          got += (size_t)r;
          hdr[got] = 0;
      }
      if (health) send_health(fd);
      else        send_models(fd);
      sock_close(fd);
      return true;
  }
  ```
- [ ] Wire it into the accept loop (src/server.c:843-850):
  ```c
      for (;;) {
          int cfd = (int)accept(lfd, NULL, NULL);
          if (cfd < 0) {
              if (errno == EINTR) continue;
              break;
          }
          if (!accept_fastpath(cfd)) q_push(cfd);
      }
  ```
- [ ] Rebuild: `make`. Fix warnings (`-Wall -Wextra` must stay clean — that is the codebase baseline).
- [ ] Re-run the repro from step 1: expect `HEALTH-OK`, and `curl -sf -m 3 http://127.0.0.1:8126/v1/models` also answers while the slot is pinned. Then run the Global Constraints smoke block (all endpoints must behave identically on an idle server, including the swap-mode `/health` `resident` field: `./runner.exe -m "a=test.gguf" --serve --port 8124 &` then `curl -sf http://127.0.0.1:8124/health` shows `"resident":null` before any request).
- [ ] Update `docs/ROADMAP.md`: delete the `- **/health should bypass the slot queue.** ...` bullet from `## Server`.
- [ ] Commit:
  ```sh
  git add src/server.c .github/workflows/ci.yml docs/ROADMAP.md
  git commit -m "server: answer /health and /v1/models from the accept loop

  Single-slot serving meant one long generation blocked /health until
  gridcore's watchdog called a busy runner \"unhealthy: timed out\". The
  accept loop now peeks each new connection (250ms select guard) and
  answers the two read-only GETs inline; everything else — including
  /unload, which frees the resident model and must stay serialized with
  generation — is queued to a slot untouched. Both handlers read only
  startup-immutable registry strings plus a snapshot of the resident
  index, so no lock is taken on the accept thread.

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 2: --parent-pid ties the runner's lifetime to its supervisor

When gridcore-clu is SIGKILLed its spawned runner is orphaned and keeps the model (and VRAM) resident. A child cannot attach itself to a parent's job object after the parent is gone, so the supervisor passes its own pid and the runner watches it: on Windows, `OpenProcess(SYNCHRONIZE)` + `WaitForSingleObject` in a thread (event-driven, zero polling); on Linux, `prctl(PR_SET_PDEATHSIG, SIGTERM)` as the instant path plus a 2s `kill(pid, 0)` poll as the general fallback; on macOS the poll alone suffices. This is the simplest design that matches the codebase's compat.c pattern (same-signature per-platform wrappers, no new dependencies).

**Files**
- Modify `C:\ProjectGrid\Runner\src\compat.h` — declare `plat_parent_watch` after `plat_now` (line 18).
- Modify `C:\ProjectGrid\Runner\src\compat.c` — Windows implementation in the `#ifdef _WIN32` block (after `plat_now`, line 54), POSIX implementation in the `#else` block (after `plat_now`, line 102); add `#include <stdio.h>` at the top (line 3 area).
- Modify `C:\ProjectGrid\Runner\src\main.c` — flag parse (insert after `--reserve-cpu`, line 165), usage text (after the `--caps` line, ~line 79), activation right after the argv loop (~line 170).
- Modify `C:\ProjectGrid\Runner\.github\workflows\ci.yml` — one smoke step in each of the `unix` and `windows` jobs.
- Modify `C:\ProjectGrid\Runner\docs\ROADMAP.md` — remove the "Job-object / process-group cleanup story" bullet.

**Interfaces**
- `void plat_parent_watch(long pid);` in compat.h — no-op for `pid <= 0`; otherwise arranges for the process to `_exit(0)` when `pid` dies. If the pid cannot be observed at all (Windows `OpenProcess` failure), the parent is treated as already dead and the runner exits immediately — an unobservable supervisor is exactly the orphan case this exists to prevent.
- CLI: `--parent-pid N` (supervisors pass their native pid: `GetCurrentProcessId()` on Windows, `getpid()` on POSIX).

**Steps**

- [ ] Write the failing check. Add to `.github/workflows/ci.yml`, `unix` job (after the health-bypass step from Task 1):
  ```yaml
        - name: smoke test (parent-pid watch)
          # the runner must exit when the process named by --parent-pid dies
          run: |
            python3 - <<'PYEOF'
            import os, subprocess, sys, time
            exe = "./runner.exe" if os.name == "nt" else "./runner"
            p = subprocess.Popen([exe, "-m", "test.gguf", "--serve", "--port", "8127",
                                  "--parent-pid", str(os.getpid())])
            time.sleep(2)
            if p.poll() is not None: sys.exit("runner died early")
            os._exit(0)  # abrupt parent death, no cleanup
            PYEOF
            sleep 6
            if curl -s -m 2 http://127.0.0.1:8127/health > /dev/null; then
              echo "runner survived parent death"; exit 1
            fi
            echo "parent-pid ok"
  ```
  and the identical step (it is exe-agnostic via `os.name`) to the `windows` job after "smoke test (server, backslash model path)".
- [ ] Run the script body locally in MSYS2 bash. Expected now: `unknown option --parent-pid` — the runner exits 1 immediately, port 8127 never opens, and the step "passes" vacuously; the real failing observation is the `unknown option` error plus `runner died early` from the wrapper. Confirm that output.
- [ ] Declare in `src/compat.h` after `double plat_now(void);`:
  ```c
  // exit(0) this process when pid dies — supervisors pass their own pid so a
  // SIGKILLed gridcore-clu cannot leave an orphaned runner. pid <= 0 = no-op.
  void        plat_parent_watch(long pid);
  ```
- [ ] Implement in `src/compat.c`. Add `#include <stdio.h>` next to the existing `#include <stdlib.h>`. Windows block (after `plat_now`):
  ```c
  static DWORD WINAPI parent_wait(LPVOID h) {
      WaitForSingleObject((HANDLE)h, INFINITE);
      fprintf(stderr, "parent process exited — shutting down\n");
      _exit(0);
  }

  void plat_parent_watch(long pid) {
      if (pid <= 0) return;
      HANDLE h = OpenProcess(SYNCHRONIZE, FALSE, (DWORD)pid);
      if (!h) {
          // unobservable parent == already-dead parent: refusing to run
          // unwatched is the whole point of the flag
          fprintf(stderr, "error: --parent-pid %ld is not observable — exiting\n", pid);
          _exit(0);
      }
      HANDLE th = CreateThread(NULL, 0, parent_wait, h, 0, NULL);
      if (th) CloseHandle(th);
  }
  ```
  POSIX block (after its `plat_now`), with `#include <errno.h>`, `#include <pthread.h>`, `#include <signal.h>` added to that block's includes and `#include <sys/prctl.h>` under `#ifdef __linux__`:
  ```c
  static void *parent_poll(void *arg) {
      long pid = (long)(intptr_t)arg;
      for (;;) {
          struct timespec ts = { 2, 0 };
          nanosleep(&ts, NULL);
          if (kill((pid_t)pid, 0) != 0 && errno == ESRCH) {
              fprintf(stderr, "parent %ld exited — shutting down\n", pid);
              _exit(0);
          }
      }
      return NULL;
  }

  void plat_parent_watch(long pid) {
      if (pid <= 0) return;
  #ifdef __linux__
      // instant path when the watched pid is the direct parent; the poll
      // below still covers grandparent supervisors and the pre-prctl race
      prctl(PR_SET_PDEATHSIG, SIGTERM);
  #endif
      pthread_t th;
      pthread_attr_t at;
      pthread_attr_init(&at);
      pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
      pthread_create(&th, &at, parent_poll, (void *)(intptr_t)pid);
      pthread_attr_destroy(&at);
  }
  ```
- [ ] Wire the flag in `src/main.c`: add `long parent_pid = 0;` next to the other option locals (~line 117), parse it after the `--reserve-cpu` line:
  ```c
      else if (!strcmp(a, "--parent-pid")) parent_pid = strtol(NEXT, NULL, 10);
  ```
  add the usage line after `--caps`:
  ```c
      "  --parent-pid N exit when process N dies (supervisor cleanup)\n"
  ```
  and activate it immediately after the argv loop (before model load, so a dead supervisor never pays for a load):
  ```c
      plat_parent_watch(parent_pid);
  ```
- [ ] Rebuild (`make`) and re-run the CI-step script body locally: the runner must serve `/health` while the python parent lives, and within ~6s of `os._exit` the health curl must fail. Verify Task Manager shows no leftover `runner.exe`.
- [ ] Run the Global Constraints smoke block (no regression without the flag).
- [ ] Update `docs/ROADMAP.md`: delete the `- **Job-object / process-group cleanup story** ...` bullet.
- [ ] Commit:
  ```sh
  git add src/compat.h src/compat.c src/main.c .github/workflows/ci.yml docs/ROADMAP.md
  git commit -m "compat: --parent-pid N ties the runner's lifetime to its supervisor

  A SIGKILLed gridcore-clu left its spawned runner orphaned with the model
  (and VRAM) resident. A child cannot join a job object after its parent
  is gone, so the supervisor passes its own pid and the runner watches it:
  Windows waits on an OpenProcess(SYNCHRONIZE) handle in a thread (no
  polling), Linux adds prctl(PR_SET_PDEATHSIG) for the direct-parent case,
  and a 2s kill(pid, 0) poll covers macOS and grandparent supervisors.
  Only ESRCH counts as death — EPERM after pid reuse must not kill a
  healthy runner.

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 3: server-side speculative decoding (unconstrained requests only)

`--draft` is CLI-only because server slots would share one draft KV cache. Give each slot its own draft `model_t` (weights dedupe through the page cache mmap, same as slot target models — see the comment at src/server.c:13-14). Scope decision already made: speculation stays self-disabled for schema/JSON-mode requests exactly as today via the existing gate `if (e->dm && !e->schema && !e->json_mode && e->lp_cap == 0)` at src/engine.c:249 — this task must not touch that line, and constraint-state rewind is out of scope. Two correctness fixes are required for the server path: `engine_rewind` must clamp `e->dpos` to the kept prefix (otherwise the draft KV silently holds tokens from the previous request), and single-model serve's `/unload`→lazy-reload path (`swap_to`, which re-runs `engine_init` and memsets the engine) must re-attach the draft. Multi-model swap mode is excluded (vocab compatibility cannot be guaranteed across a registry).

**Files**
- Modify `C:\ProjectGrid\Runner\src\engine.c` — new `spec_draft_load` helper (place after `engine_rewind`, ~line 57); clamp `dpos` in `engine_rewind` (line 45 area) and reset it in `engine_reset` (line 36 area).
- Modify `C:\ProjectGrid\Runner\src\runner.h` — declare `spec_draft_load` in the engine section (after `engine_generate`, line 486).
- Modify `C:\ProjectGrid\Runner\src\main.c` — `server_run` declaration (lines 12-14), the serve call (lines 271-273), replace the CLI draft block (lines 280-303) with the helper, usage text for `--draft` (line 77).
- Modify `C:\ProjectGrid\Runner\src\server.c` — `server_run` signature (lines 699-701), swap-mode guard (~line 706), per-slot draft load in the non-registry branch (after `engine_init` at line 788), draft stash in the `single_setup` block (lines 791-812), re-attach in `swap_to` (after `engine_init` at line 193), `SV` fields (~line 141), `[spec]` marker in the slot log line (line 457-461).
- Modify `C:\ProjectGrid\Runner\.github\workflows\ci.yml` — new smoke step in the `unix` job.
- Modify `C:\ProjectGrid\Runner\docs\ROADMAP.md` — rewrite the "Server-side speculation" bullet to note it landed for unconstrained requests, constraint rewind still open.

**Interfaces**
- `model_t *spec_draft_load(const char *path, const model_t *target, const model_params *mp);` — heap-allocates and loads a draft with the CLI's exact gates (target must keep a CPU verify path, i.e. not fully GPU-offloaded; `|dm->n_vocab - target->n_vocab| <= 512`); forces `n_ctx = target->n_ctx`, `kv_q8 = false`, `gpu_mode = GPU_AUTO`; returns NULL with a stderr note when speculation cannot be enabled. Deliberate behavior change: a failed draft load now degrades to plain decoding instead of the CLI's old exit(1).
- `int server_run(model_t *base, tokenizer *tok, const char *model_path, const model_params *mp, sampler defaults, int port, int parallel, int n_threads, int ttl, const char *draft_path, int draft_k);` — two new trailing params.
- `SV` gains `model_t *draft; int draft_k;` (single-model serve only; used by `swap_to` to re-attach after reload).
- Engine invariants: `engine_rewind` guarantees `e->dpos <= e->pos` on return; `engine_reset` zeroes `dpos`. The spec catch-up loop (src/engine.c:165-172) then re-feeds `hist[dpos..pos)` into the draft, which is exactly the prefix-reuse contract the server needs.

**Steps**

- [ ] Write the failing check — CI step in the `unix` job (after "smoke test (model swap)"):
  ```yaml
        - name: smoke test (server speculative decoding)
          # a self-draft server at temp 0 must match the plain server's output
          # byte-for-byte, and a json-constrained request must still succeed
          # (speculation self-disables under constraints, exactly as the CLI)
          run: |
            REQ='{"messages":[{"role":"user","content":"hi"}],"max_tokens":12,"temperature":0}'
            ./runner -m test.gguf --serve --port 8128 --gpu off &
            sleep 2
            P=$(curl -sf -d "$REQ" http://127.0.0.1:8128/v1/chat/completions | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])")
            kill %1; wait %1 2>/dev/null || true
            ./runner -m test.gguf --draft test.gguf --serve --port 8128 --gpu off &
            sleep 2
            S=$(curl -sf -d "$REQ" http://127.0.0.1:8128/v1/chat/completions | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])")
            [ -n "$P" ] || exit 1
            [ "$P" = "$S" ] || { echo "spec output diverged: '$P' vs '$S'"; exit 1; }
            curl -sf -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":24,"temperature":0,"response_format":{"type":"json_object"}}' \
              http://127.0.0.1:8128/v1/chat/completions | python3 -c "import json,sys; json.loads(json.load(sys.stdin)['choices'][0]['message']['content'])"
            kill %1
            echo "server spec ok"
  ```
- [ ] Run the step body locally (with `./runner.exe`). Expected now: the second server start prints nothing about drafts (`--draft` is accepted by the parser but ignored under `--serve` since `main.c` only wires it on the non-serve path) — `S` equals plain output trivially, but the *engine telemetry proves the gap*: no `spec: N rounds ...` line appears on the draft server's stderr. Record that as the failing observation (grep the server stderr for `spec:` — absent).
- [ ] Engine fixes in `src/engine.c`. In `engine_reset` (after `e->hit_stop = false;`):
  ```c
      e->dpos = 0;
  ```
  In `engine_rewind` (after `e->pos = keep;`):
  ```c
      // the draft's KV beyond the kept prefix was computed from the previous
      // request's tokens; the catch-up loop re-feeds hist[dpos..pos)
      if (e->dpos > keep) e->dpos = keep;
  ```
- [ ] Add the shared loader to `src/engine.c` (after `engine_rewind`), and its declaration to `src/runner.h` after `engine_generate`:
  ```c
  // load a draft model for speculative decoding, with the same gates in CLI
  // and server mode: the target must keep a CPU verify path, and the vocabs
  // must match modulo family padding. NULL (with a stderr note) = run plain.
  model_t *spec_draft_load(const char *path, const model_t *target,
                           const model_params *mp) {
      if (target->gpu && target->gpu_layers >= target->n_layer) {
          fprintf(stderr, "draft: target is fully GPU-offloaded — speculative "
                  "decoding needs the CPU verify path, ignoring --draft\n");
          return NULL;
      }
      model_params dmp = *mp;
      dmp.n_ctx = target->n_ctx; // draft must cover the target's positions
      dmp.kv_q8 = false;
      dmp.gpu_mode = GPU_AUTO;   // a small draft usually fits VRAM whole
      model_t *dm = malloc(sizeof(model_t));
      if (!model_load(dm, path, &dmp)) {
          free(dm);
          return NULL;
      }
      if (abs(dm->n_vocab - target->n_vocab) > 512) {
          // model families pad the vocab differently per size; small
          // differences are padding ids the draft never emits
          fprintf(stderr, "draft: vocab mismatch (%d vs %d) — ignoring --draft\n",
                  dm->n_vocab, target->n_vocab);
          model_free(dm);
          free(dm);
          return NULL;
      }
      return dm;
  }
  ```
  Declaration: `model_t *spec_draft_load(const char *path, const model_t *target, const model_params *mp);`
- [ ] Rewire the CLI in `src/main.c`: delete the `static model_t dmodel;` block (lines 280-303) and replace with:
  ```c
      if (draft_path) {
          e.dm = spec_draft_load(draft_path, &m, &mp);
          if (e.dm) {
              fprintf(stderr, "draft: %s (%d tokens per round)\n", draft_path, draft_k);
              e.draft_k = draft_k;
          }
      }
  ```
  Keep this block where it is (CLI-only path); extend the serve call at lines 271-273:
  ```c
      if (serve)
          return server_run(registry ? NULL : &m, registry ? NULL : &tok,
                            model_path, &mp, smp, port, parallel, n_threads, ttl,
                            draft_path, draft_k);
  ```
  and the declaration at lines 12-14 to match the new signature. Update the usage line 77 to:
  ```c
      "  --draft PATH   small same-vocab GGUF for speculative decoding\n"
      "                 (one-shot, chat, and single-model --serve)\n"
  ```
- [ ] Wire the server in `src/server.c`. Signature (lines 699-701) gains `, const char *draft_path, int draft_k`. Add to `SV` (after `model_params mp;`, line 141):
  ```c
      model_t    *draft;        // per-slot draft would be plural; single-model
      int         draft_k;      // serve has exactly one slot (see swap_to)
  ```
  Swap-mode guard right after `bool swap_mode = ...` (line 705):
  ```c
      if (draft_path && swap_mode) {
          fprintf(stderr, "note: --draft needs a single served model — "
                  "ignoring it in swap mode\n");
          draft_path = NULL;
      }
  ```
  Per-slot attach in the non-registry branch, inside the slot loop directly after `engine_init(&s->e, s->m, s->tok, &s->smp);` (line 788):
  ```c
              if (draft_path) {
                  // per-slot draft context: each slot owns a full draft KV;
                  // weights dedupe through the page cache like slot models
                  s->e.dm = spec_draft_load(draft_path, s->m, &slot_mp);
                  if (s->e.dm) s->e.draft_k = draft_k;
              }
  ```
  Stash for reloads in the `single_setup` block (after `SV.mp.n_threads = threads_per_slot;`, line 808):
  ```c
              SV.draft = SV.slots[0].e.dm;
              SV.draft_k = draft_k;
  ```
  Re-attach in `swap_to` after `engine_init(&s->e, s->m, s->tok, &s->smp);` (line 193):
  ```c
          if (SV.single && SV.draft) {
              // engine_init memsets the engine; the draft (own KV, own pool)
              // survives target unload/reload and is re-attached here
              s->e.dm = SV.draft;
              s->e.draft_k = SV.draft_k;
          }
  ```
  Telemetry: in the slot log fprintf (lines 457-461) extend the marker expression to
  ```c
              schema ? " [schema]" : e->json_mode ? " [json]" : e->dm ? " [spec]" : "",
  ```
- [ ] Rebuild: `make`. Re-run the CI step body locally: draft server stderr now shows `spec: N rounds, N drafted, N accepted` per request, `P == S`, and the json_object request returns valid JSON with NO `spec:` line for that request (self-disable preserved). Also exercise the reload path: `curl -sf http://127.0.0.1:8128/unload` then repeat `$REQ` — output must still equal `$P` and still print a `spec:` line (proves the `swap_to` re-attach).
- [ ] Run the full Global Constraints smoke block plus the existing CLI spec smoke (`A=$(./runner.exe -m test.gguf -p "hello" -n 12 --temp 0 --gpu off 2>/dev/null); B=$(./runner.exe -m test.gguf --draft test.gguf -p "hello" -n 12 --temp 0 --gpu off 2>/dev/null); [ "$A" = "$B" ] && echo ok`).
- [ ] Update `docs/ROADMAP.md`: replace the `- **Server-side speculation** ...` bullet with a note that per-slot draft contexts landed for unconstrained requests; schema/JSON-constrained requests still run plain (constraint-state rewind per rejected draft remains open).
- [ ] Commit:
  ```sh
  git add src/engine.c src/runner.h src/main.c src/server.c .github/workflows/ci.yml docs/ROADMAP.md
  git commit -m "server: per-slot draft contexts — --draft now works under --serve

  Speculation was CLI-only because slots would have shared one draft KV
  cache. Each slot now loads its own draft model_t (weights dedupe via the
  page-cache mmap, same as slot targets), attached to the slot engine. The
  existing constraint gate in engine_generate is untouched: schema/JSON
  requests keep decoding plain, exactly as before.

  Two latent engine fixes the server path exposed: engine_rewind clamps
  dpos to the kept prefix (the draft KV beyond it belongs to the previous
  request), and engine_reset zeroes dpos. Single-model serve re-attaches
  the draft after swap_to's engine_init, so /unload + lazy reload keeps
  speculating. Multi-model swap mode refuses --draft (registry entries
  cannot guarantee a shared vocab). Draft-load failure now degrades to
  plain decoding instead of exit(1) — the loader is shared with the CLI.

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 4: CUDA kernel tuning — measurement-driven

Kernels are correctness-first PTX; CUDA graphs and coalesced loads were flagged as the big headroom when the backend landed. **Toolchain status (verified on this machine)**: `nvcc` v13.3 exists at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe` and MSVC 14.44.35207 BuildTools are installed, so `make ptx` regeneration is fully possible locally; `scripts/regen-ptx-win.sh` has a stale `cd /c/Grid/runner` that must be fixed first. Every experiment is gated: keep only if median decode tok/s improves AND temp-0 output is byte-identical to the pre-change baseline. Bench models (verified present): `C:\ProjectGrid\models\Qwen_Qwen3-4B-Q8_0.gguf` (4.3 GB, fully offloads on the 3070 — cleanest signal, exercises `k_mv_q8_0`) and `C:\ProjectGrid\models\Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf` (5.7 GB, exercises `k_mv_q5_K`/`k_mv_q6_K`); Q4_K coverage via `C:\ProjectGrid\models\gemma-4-12B-it-Q4_K_M.gguf` under partial offload.

**Files**
- Create `C:\ProjectGrid\Runner\scripts\bench.sh` — reproducible decode benchmark.
- Modify `C:\ProjectGrid\Runner\scripts\regen-ptx-win.sh` — fix the stale path (line 4).
- Modify `C:\ProjectGrid\Runner\src\cuda.c` — typedefs (line 41-47), `cu` struct + `cu_load` (lines 48-110), `gpu_t` (lines 120-139), `launch()` (lines 409-412) and all `enc_*` call sites, `gpu_init` allocations (~line 332-342), `gpu_free` (lines 470-496), `fwd_tile` (lines 532-627), `gpu_forward_batch` (lines 629-677), arg structs (lines 145-147).
- Modify `C:\ProjectGrid\Runner\src\kernels.cu` — `rope_args`/`k_rope` (lines 625-643), `k_store_kv` (lines 648-658), `attn_args`/`k_attn` (lines 664-728), experiment B: `k_mv_q4_K`/`k_mv_q4_K_b` (lines 347-404), experiment C: `k_attn` score/value loops (lines 687-693, 722-727).
- Regenerated: `C:\ProjectGrid\Runner\src\kernels.ptx`, `C:\ProjectGrid\Runner\src\kernels_ptx.h` (via the ptx command, never by hand).

**Interfaces**
- Benchmark protocol (the contract every experiment is judged by): `sh scripts/bench.sh <model.gguf> [n]` runs 3 identical greedy completions (`-p "Write a short story about a lighthouse keeper." -n 256 --temp 0 -s 1`), prints each run's `gen: N tok, X tok/s` stderr line (emitted by main.c:356-358) and the md5 of each stdout. Decision metric: median of the 3 `gen` tok/s. Correctness gate: all three md5s identical to each other AND to the baseline md5 recorded before the change.
- PTX regeneration (exact command, MSYS2 bash at repo root):
  ```sh
  PATH="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64:$PATH" \
  "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/nvcc.exe" \
      -ptx -arch=compute_75 -O3 -o src/kernels.ptx src/kernels.cu \
  && python scripts/embed-ptx.py
  ```
  (embed-ptx.py re-pins `.version 7.8` — verify the pin survives: after each regen the header must contain `.version 7.8`.)
- Graph experiment internal contract: `pos` moves from kernel arguments into a device int (`g->pos_dev`), making the recorded graph position-invariant; kernels `k_rope`, `k_attn`, `k_store_kv` gain a `const int *posp` parameter and `rope_args`/`attn_args` lose their `pos` field; `launch()` becomes `launch(gpu_t *g, ...)` and submits to `g->stream`; graphs activate only for `n == 1 && want_logits && !partial`, are disabled by `RUNNER_CUDA_GRAPH_OFF=1` (A/B testing without rebuilds, mirroring the existing `RUNNER_DEBUG_TOKENS` getenv pattern), and any capture/launch failure sets `g->graph_bad` and falls back to the plain path forever.

**Steps**

- [ ] Fix the stale script and add the bench harness. `scripts/regen-ptx-win.sh` line 4: replace `cd /c/Grid/runner` with `cd "$(dirname "$0")/.."`. Create `scripts/bench.sh`:
  ```sh
  #!/bin/sh
  # reproducible decode benchmark: 3 greedy runs, per-run gen tok/s + output md5.
  # decision metric is the MEDIAN tok/s; the md5s must all match the baseline.
  M=${1:?usage: bench.sh model.gguf [n_tokens]}
  N=${2:-256}
  EXE=./runner
  [ -x ./runner.exe ] && EXE=./runner.exe
  for i in 1 2 3; do
      "$EXE" -m "$M" -p "Write a short story about a lighthouse keeper." \
          -n "$N" --temp 0 -s 1 > "bench-out.$i.txt" 2> "bench-err.$i.txt"
      grep -o 'gen: [0-9]* tok, [0-9.]* tok/s' "bench-err.$i.txt"
  done
  md5sum bench-out.1.txt bench-out.2.txt bench-out.3.txt
  ```
  `chmod +x scripts/bench.sh`. Leave bench output files untracked; do not commit them.
- [ ] Record baselines (this is the "failing test" of a measurement task — the numbers to beat):
  ```sh
  sh scripts/bench.sh /c/ProjectGrid/models/Qwen_Qwen3-4B-Q8_0.gguf
  sh scripts/bench.sh /c/ProjectGrid/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf
  sh scripts/bench.sh /c/ProjectGrid/models/gemma-4-12B-it-Q4_K_M.gguf
  ```
  Save the three median tok/s figures and the three md5s for the commit messages. Also sanity-check PTX regen round-trips before touching kernels: run the regeneration command from Interfaces, confirm `git diff --stat src/kernels_ptx.h` is empty-or-cosmetic, then `git checkout -- src/kernels.ptx src/kernels_ptx.h`.
- [ ] Commit the harness:
  ```sh
  git add scripts/bench.sh scripts/regen-ptx-win.sh
  git commit -m "scripts: decode benchmark harness + regen-ptx-win path fix

  bench.sh pins prompt/seed/temp and reports per-run gen tok/s plus output
  md5s: kernel experiments are gated on median-tok/s improvement with
  byte-identical temp-0 output. regen-ptx-win.sh cd'd into the repo's old
  /c/Grid/runner location; it now derives the root from its own path.

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

**Experiment A — CUDA graphs for the single-token decode loop** (~19 launches x n_layer per token today; WDDM launch overhead is the known cost, per the comment at src/cuda.c:526-531).

- [ ] Kernels: make position a device read. In `src/kernels.cu` change `rope_args` (line 625) to `{ int head_dim, n_heads, half_dim, neox; float mscale; }` and `k_rope` to:
  ```c
  extern "C" __global__ void k_rope(float *v, const float *fr, rope_args a,
                                    const int *posp, int vs) {
      int j = blockIdx.x * blockDim.x + threadIdx.x;
      int h = blockIdx.y;
      if (j >= a.half_dim || h >= a.n_heads) return;
      int pos = *posp + blockIdx.z;
      ...rest unchanged...
  ```
  `k_store_kv` (line 648):
  ```c
  extern "C" __global__ void k_store_kv(const float *k, const float *v,
                                        __half *kc, __half *vc,
                                        int kv_dim, ulong64 l_off,
                                        const int *posp) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < kv_dim) {
          ulong64 src = (ulong64)blockIdx.y * kv_dim + i;
          ulong64 dst = l_off + (ulong64)(*posp + blockIdx.y) * kv_dim + i;
          kc[dst] = __float2half(k[src]);
          vc[dst] = __float2half(v[src]);
      }
  }
  ```
  `attn_args` (line 664) loses `pos`; `k_attn` gains `const int *posp` after `attn_args a` and computes `int pos = *posp + tk;` (replacing `a.pos + tk` at line 678).
- [ ] cuda.c mirror: update the host-side `rope_args`/`attn_args` structs (lines 146-147) to match; add typedefs `typedef void *CUstream; typedef void *CUgraph; typedef void *CUgraphExec;` (line 47 area); extend the `cu` struct with
  ```c
      CUresult (*StreamCreate)(CUstream *, unsigned);
      CUresult (*StreamDestroy)(CUstream);
      CUresult (*StreamSynchronize)(CUstream);
      CUresult (*StreamBeginCapture)(CUstream, int);
      CUresult (*StreamEndCapture)(CUstream, CUgraph *);
      CUresult (*GraphInstantiateWithFlags)(CUgraphExec *, CUgraph, unsigned long long);
      CUresult (*GraphLaunch)(CUgraphExec, CUstream);
      CUresult (*GraphExecDestroy)(CUgraphExec);
      CUresult (*GraphDestroy)(CUgraph);
  ```
  loaded in `cu_load` via `dl_sym`/`sym2` with names `cuStreamCreate`, `cuStreamDestroy` (`sym2`), `cuStreamSynchronize`, `cuStreamBeginCapture` (`sym2`), `cuStreamEndCapture` (`sym2`), `cuGraphInstantiateWithFlags`, `cuGraphLaunch`, `cuGraphExecDestroy`, `cuGraphDestroy` — these are OPTIONAL: do not add them to the required `&&` chain; add
  ```c
  static bool cu_graphs_ok(void) {
      return cu.StreamCreate && cu.StreamDestroy && cu.StreamSynchronize &&
             cu.StreamBeginCapture && cu.StreamEndCapture &&
             cu.GraphInstantiateWithFlags && cu.GraphLaunch &&
             cu.GraphExecDestroy && cu.GraphDestroy;
  }
  ```
- [ ] `gpu_t` gains `CUstream stream; CUgraph graph; CUgraphExec gexec; CUdeviceptr pos_dev; bool graph_bad;`. In `gpu_init` (next to the `g->dummy` alloc, line 342): `CK(cu.MemAlloc(&g->pos_dev, sizeof(int)));` and `if (cu_graphs_ok()) cu.StreamCreate(&g->stream, 0);` — also set `g->graph_bad = true` when any offloaded layer has `ly->wv == NULL` (the gemma4 V-copy path uses a synchronous `MemsetD8` that cannot be captured). In `gpu_free`: `if (g->gexec) cu.GraphExecDestroy(g->gexec); if (g->graph) cu.GraphDestroy(g->graph); if (g->stream) cu.StreamDestroy(g->stream);` and add `g->pos_dev` to the `bufs[]` free list.
- [ ] Route every launch through the stream: change `launch` (line 409) to
  ```c
  static bool launch(gpu_t *g, CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                     unsigned bx, void **params) {
      return cu.LaunchKernel(f, gx, gy, gz, bx, 1, 1, 0, g->stream, params, NULL) == 0;
  }
  ```
  and update all call sites (`enc_rmsnorm`, `enc_mv`, `enc_qknorm`, `enc_rope`, `enc_scale`, `enc_add`, `enc_actmul`, and the two direct `launch(...)` calls in `fwd_tile` for store/attn) to pass `g`. Update `enc_rope`'s params to `{ &v, &fr, &a, &g->pos_dev, &vs }` (dropping `pos` from the struct init), the store launch params to `{ &g->kt, &g->vt, &g->kc, &g->vc, &kv_dim, &l_off, &g->pos_dev }` where `uint64_t l_off = (uint64_t)m->kv_off[l];`, and the attn params to `{ &g->q, &g->kc, &g->vc, &g->att, &g->xb2, &aa, &g->pos_dev }`.
- [ ] Hoist embedding staging out of `fwd_tile` (lines 541-551) into
  ```c
  // stage the tile's token embeddings into g->x (host dequant + one HtoD);
  // kept outside fwd_tile so graph capture records kernels only
  static bool stage_x(gpu_t *g, model_t *m, const int32_t *tokens, int tn) {
      size_t ers = ggml_row_size(m->tok_embd->type, m->n_embd);
      for (int b = 0; b < tn; b++) {
          float *hx = g->h_x + (size_t)b * m->n_embd;
          dequant_row(m->tok_embd->type,
                      (uint8_t *)m->tok_embd->data + (size_t)tokens[b] * ers,
                      hx, m->n_embd);
          if (m->embd_scale != 1.0f)
              for (int i = 0; i < m->n_embd; i++) hx[i] *= m->embd_scale;
      }
      return cu.MemcpyHtoD(g->x, g->h_x, sizeof(float) * tn * m->n_embd) == 0;
  }
  ```
  `fwd_tile` drops its `l0 == 0` staging block; the tile loop in `gpu_forward_batch` calls, per tile, `int p = pos + i;` then `stage_x(g, m, tokens + i, tn)` and `cu.MemcpyHtoD(g->pos_dev, &p, sizeof(int))` before `fwd_tile(...)`.
- [ ] Graph path in `gpu_forward_batch`, inserted after the KV resync (line 641) and before the tile loop:
  ```c
      bool partial = m->gpu_layers < m->n_layer;
      if (n == 1 && want_logits && !partial && !g->graph_bad && g->stream &&
          !getenv("RUNNER_CUDA_GRAPH_OFF")) {
          if (!stage_x(g, m, tokens, 1) ||
              cu.MemcpyHtoD(g->pos_dev, &pos, sizeof(int)) != 0) return false;
          if (!g->gexec) {
              // capture records the launch sequence without executing it; the
              // recorded graph is position-invariant because pos lives in
              // device memory, so one capture serves every decode token
              if (cu.StreamBeginCapture(g->stream, 0) != 0 ||
                  !fwd_tile(g, m, tokens, 1, pos, true, 0, m->gpu_layers) ||
                  cu.StreamEndCapture(g->stream, &g->graph) != 0 ||
                  cu.GraphInstantiateWithFlags(&g->gexec, g->graph, 0) != 0) {
                  CUgraph junk = NULL;
                  cu.StreamEndCapture(g->stream, &junk); // ensure capture ended
                  if (junk && junk != g->graph) cu.GraphDestroy(junk);
                  fprintf(stderr, "gpu: graph capture failed — plain launches\n");
                  g->graph_bad = true;
              }
          }
          if (g->gexec && cu.GraphLaunch(g->gexec, g->stream) == 0 &&
              cu.StreamSynchronize(g->stream) == 0) {
              if (!kv_copyback(g, m, pos, pos + 1)) return false;
              g->last_pos = pos;
              if (cu.MemcpyDtoH(g->h_logits, g->logits, sizeof(float) * m->n_vocab) != 0)
                  return false;
              if (logits) *logits = g->h_logits;
              return true;
          }
          if (!g->graph_bad) { g->graph_bad = true;
              fprintf(stderr, "gpu: graph launch failed — plain launches\n"); }
      }
  ```
  (The pre-existing tile loop below remains the universal fallback; remove its now-duplicate `bool partial` declaration.)
- [ ] Regenerate PTX (Interfaces command), verify `.version 7.8` survived, `make`, then correctness: `./runner.exe -m /c/ProjectGrid/models/Qwen_Qwen3-4B-Q8_0.gguf -p "hello" -n 32 --temp 0 -s 1` output must equal the same command with `RUNNER_CUDA_GRAPH_OFF=1`, and both must equal the pre-change baseline output. Then measure: `sh scripts/bench.sh /c/ProjectGrid/models/Qwen_Qwen3-4B-Q8_0.gguf` and the Llama-8B run. Also run the full Global Constraints smoke block (the CI test model exercises CPU only, so also re-run the gemma-4-12B bench for the partial-offload + wv==NULL gate).
- [ ] Decision: if median tok/s improves on the fully-offloaded model AND all md5s match baseline, commit; otherwise revert everything from this experiment (`git checkout -- src/cuda.c src/kernels.cu src/kernels.ptx src/kernels_ptx.h`) and record the numbers in the ROADMAP bullet instead.
  ```sh
  git add src/cuda.c src/kernels.cu src/kernels.ptx src/kernels_ptx.h
  git commit -m "gpu: CUDA graphs for the single-token decode path

  ~19 launches per layer per token paid WDDM submission overhead on every
  decode step. pos now lives in a device int, making the launch sequence
  position-invariant: one stream capture at the first decode token yields
  a graph that replays for every subsequent token (embedding staging, pos
  upload, logits DtoH and KV copyback stay outside). Prompt tiles, partial
  offload, gemma4's V-copy memset, and drivers without the graph entry
  points all keep the plain path; RUNNER_CUDA_GRAPH_OFF=1 forces it.

  Qwen3-4B-Q8_0 (full offload, RTX 3070): <BASELINE> -> <NEW> tok/s
  median, temp-0 output byte-identical.

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

**Experiment B — vectorized loads in k_mv_q4_K / k_mv_q4_K_b** (hottest matvec for the fleet's Q4_K_M models; per-byte loads today at kernels.cu:367-370). Alignment is guaranteed: q4_K blocks are 144 bytes (144 % 16 == 0), tensor data is 32-byte aligned in GGUF, and `q` sits at blk+16 — `uint4` loads are legal.

- [ ] Rewrite the inner loop of `k_mv_q4_K` (lines 360-373) to load 16 quant bytes per instruction:
  ```c
          int is = 0;
          for (int j = 0; j < 256; j += 64) {
              uchar s1, m1, s2, m2;
              get_scale_min_k4(is + 0, sc, &s1, &m1);
              get_scale_min_k4(is + 1, sc, &s2, &m2);
              float d1 = d * s1, mm1 = dmin * m1;
              float d2 = d * s2, mm2 = dmin * m2;
              float t1 = 0, t2 = 0, sx1 = 0, sx2 = 0;
              const uint4 *q16 = (const uint4 *)q;   // blk+16 is 16B-aligned
              for (int v = 0; v < 2; v++) {
                  uint4 w = q16[v];
                  uint ws[4] = { w.x, w.y, w.z, w.w };
                  #pragma unroll
                  for (int c = 0; c < 4; c++) {
                      #pragma unroll
                      for (int k = 0; k < 4; k++) {
                          int l = v * 16 + c * 4 + k;
                          uint b8 = (ws[c] >> (8 * k)) & 0xFFu;
                          t1 += (float)(b8 & 0xF) * xp[l];      sx1 += xp[l];
                          t2 += (float)(b8 >> 4)  * xp[l + 32]; sx2 += xp[l + 32];
                      }
                  }
              }
              s += d1 * t1 - mm1 * sx1 + d2 * t2 - mm2 * sx2;
              q += 32; is += 2; xp += 64;
          }
  ```
  Apply the same transformation to the `k_mv_q4_K_b` inner loop (lines 390-401), replacing the two `MV_FMA` lines with the decoded `b8` nibbles: `MV_FMA(d1 * (float)(b8 & 0xF) - mm1, base + j + l); MV_FMA(d2 * (float)(b8 >> 4) - mm2, base + j + l + 32);`.
- [ ] Regenerate PTX, `make`, correctness + measure on the Q4_K model: `sh scripts/bench.sh /c/ProjectGrid/models/gemma-4-12B-it-Q4_K_M.gguf` (md5s must match that model's baseline; gemma-4 runs partial offload on 8 GB, which is exactly the fleet configuration). Also re-run the Qwen3-4B bench to confirm no regression on non-Q4_K models.
- [ ] Keep/revert on the same gate. If kept:
  ```sh
  git add src/kernels.cu src/kernels.ptx src/kernels_ptx.h
  git commit -m "gpu: 16-byte vectorized quant loads in the q4_K matvec

  k_mv_q4_K decoded its 32-byte quant chunks one uchar at a time; q4_K's
  144-byte blocks and GGUF's 32-byte tensor alignment make uint4 loads
  legal at blk+16, cutting load instructions 16x in the hottest kernel
  for the fleet's Q4_K_M models.

  gemma-4-12B Q4_K_M (partial offload, RTX 3070): <BASELINE> -> <NEW>
  tok/s median, temp-0 output byte-identical.

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

**Experiment C — __half2 K/V loads in k_attn** (per-thread serial `__half` reads of whole K rows at kernels.cu:688-690; grows with context). Element offsets `l_off + t*kv_dim + kvh*hd` are always even (kv_dim and hd are even for every supported arch), so `__half2` (4-byte) loads are aligned.

- [ ] Rewrite the score loop:
  ```c
      for (int t = t0 + tid; t <= pos; t += tpg) {
          const __half2 *kt2 = (const __half2 *)(kc + a.l_off +
                               (ulong64)t * kv_dim + kvh * hd);
          float s = 0;
          for (int i = 0; i < hd / 2; i++) {
              float2 kf = __half22float2(kt2[i]);
              s += qh[2 * i] * kf.x + qh[2 * i + 1] * kf.y;
          }
          ah[t] = s * a.scale;
      }
  ```
  and the value loop (lines 722-727) — widen the per-thread read by processing two `i` per thread:
  ```c
      for (int i2 = tid; i2 < hd / 2; i2 += tpg) {
          float o0 = 0, o1 = 0;
          for (int t = t0; t <= pos; t++) {
              const __half2 *vt2 = (const __half2 *)(vc + a.l_off +
                                   (ulong64)t * kv_dim + kvh * hd);
              float2 vf = __half22float2(vt2[i2]);
              o0 += ah[t] * vf.x;
              o1 += ah[t] * vf.y;
          }
          out[(ulong64)tk * a.os + h * hd + 2 * i2]     = o0 / sum;
          out[(ulong64)tk * a.os + h * hd + 2 * i2 + 1] = o1 / sum;
      }
  ```
  (hd is even for all supported archs; the old scalar tail is unnecessary.)
- [ ] Regenerate PTX, `make`, correctness + measure with a longer decode so attention weight grows: `sh scripts/bench.sh /c/ProjectGrid/models/Qwen_Qwen3-4B-Q8_0.gguf 512` against a freshly recorded n=512 baseline (record it before applying this experiment). md5 gate as always.
- [ ] Keep/revert. If kept:
  ```sh
  git add src/kernels.cu src/kernels.ptx src/kernels_ptx.h
  git commit -m "gpu: half2 K/V loads in the attention kernel

  k_attn read K rows one __half at a time per thread and V one lane-column
  at a time; KV row offsets are always even-element so 4-byte __half2
  loads are aligned, halving load instructions in both attention loops.

  Qwen3-4B-Q8_0, n=512 decode (RTX 3070): <BASELINE> -> <NEW> tok/s
  median, temp-0 output byte-identical.

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```
- [ ] Close out: update `docs/ROADMAP.md`'s "CUDA kernel tuning headroom" bullet with what landed and what was measured-and-rejected (numbers included), commit as `docs: record CUDA tuning results` with the standard trailer, and run the complete Global Constraints smoke block one final time on the branch tip.

---

### Critical Files for Implementation
- C:\ProjectGrid\Runner\src\server.c
- C:\ProjectGrid\Runner\src\engine.c
- C:\ProjectGrid\Runner\src\cuda.c
- C:\ProjectGrid\Runner\src\kernels.cu
- C:\ProjectGrid\Runner\.github\workflows\ci.yml
