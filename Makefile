CC      ?= cc
# Probe by RUNNING the interpreter, not by looking it up: on Windows a
# python3 "app execution alias" stub exists on PATH but exits non-zero with
# a Store advert, so `command -v python3` picks an interpreter that cannot run
PYTHON  ?= $(shell python3 -c "" >/dev/null 2>&1 && echo python3 || echo python)
# gnu11 (not c11): strict ISO mode hides M_PI and POSIX symbols on glibc/MinGW
# -march=native unlocks the AVX2/FMA/F16C dot kernels in quants.c on x86;
# other ISAs (ARM macs) compile the scalar fallbacks
CFLAGS  ?= -O3 -ffast-math -std=gnu11 -Wall -Wextra -Wno-unused-parameter -march=native
# Mandatory engine codegen, appended so it survives a hostile *environment*
# CFLAGS. A conda/distro toolchain that exports CFLAGS=-march=nocona -O2
# otherwise silently defeats the `?=` default above: __AVX2__ goes undefined,
# every AVX2/FMA/F16C dot kernel in quants.c is `#if`-compiled out, and the
# runner ships a SCALAR binary on AVX-512 hardware (measured: zero ymm/zmm
# instructions, ~6x slower end-to-end). Plain `+=` (NOT `override`) is exactly
# right here: it appends to an environment-set CFLAGS so the last -O/-march wins
# and the conda clobber is undone, but it is ignored for a *command-line* CFLAGS
# so the release build's portable `make CFLAGS="... -march=x86-64-v3"` pin is
# preserved (a release must not bake in the build host's -march=native).
# Cross-compile a local build with ARCH_FLAGS=-march=<target>. NB: do NOT name
# this RUNNER_ARCH — GitHub Actions sets RUNNER_ARCH=X64/ARM64 in the build
# environment, which a `?=` inherits and then leaks as a bogus bare compiler arg.
ARCH_FLAGS ?= -march=native
CFLAGS += -O3 -ffast-math -std=gnu11 $(ARCH_FLAGS)
LDFLAGS  = -lm -lpthread
ifeq ($(OS),Windows_NT)
# -static: link winpthread/libgcc into the exe so it runs outside an MSYS2
# shell (otherwise it dies at load with STATUS_DLL_NOT_FOUND on libwinpthread-1.dll)
CFLAGS += -Werror=format-truncation -Werror=cpp
LDFLAGS += -lws2_32 -lpsapi -static   # psapi: QueryWorkingSetEx / GetProcessMemoryInfo
# tray: gdi32 (icon painting) and comdlg32 (GetOpenFileName) are not in the
# MinGW default lib set; shell32/advapi32 are but stay explicit for clarity
LDFLAGS += -lshell32 -lgdi32 -lcomdlg32 -ladvapi32
GPU_SRC  = src/cuda.c
GPU_BACKEND_DEF = -DRUNNER_GPU_CUDA
TRAY_SRC = src/tray.c src/tray_win.c
RUNNER_EXE = runner.exe
TEST_JSON_SCHEMA = test-json-schema.exe
TEST_TOKENIZER = test-tokenizer.exe
TEST_TEMPLATE = test-template.exe
TEST_TOOLS = test-tools.exe
TEST_JSON_OOM = test-json-oom.exe
TEST_TOKENIZER_OOM = test-tokenizer-oom.exe
TEST_TEMPLATE_OOM = test-template-oom.exe
TEST_SCHEMA_OOM = test-schema-oom.exe
TEST_SAMPLER = test-sampler.exe
TEST_SHARED = test-shared-weights.exe
TEST_BATCH = test-batch.exe
DIFFTOK = difftok.exe
TEST_BIND = test-bind.exe
else ifeq ($(shell uname -s),Darwin)
GPU_SRC  = src/metal.m
GPU_BACKEND_DEF = -DRUNNER_GPU_METAL
CFLAGS += -Werror=nan-infinity-disabled
LDFLAGS += -framework Metal -framework Foundation
# AppKit only on Darwin, only for the tray backend; UniformTypeIdentifiers
# for the non-deprecated NSOpenPanel file filter
LDFLAGS += -framework AppKit -framework UniformTypeIdentifiers
TRAY_SRC = src/tray.c src/tray_macos.m
RUNNER_EXE = runner
TEST_JSON_SCHEMA = test-json-schema
TEST_TOKENIZER = test-tokenizer
TEST_TEMPLATE = test-template
TEST_TOOLS = test-tools
TEST_JSON_OOM = test-json-oom
TEST_TOKENIZER_OOM = test-tokenizer-oom
TEST_TEMPLATE_OOM = test-template-oom
TEST_SCHEMA_OOM = test-schema-oom
TEST_SAMPLER = test-sampler
TEST_SHARED = test-shared-weights
TEST_BATCH = test-batch
DIFFTOK = difftok
TEST_BIND = test-bind
else
GPU_SRC  = src/cuda.c
GPU_BACKEND_DEF = -DRUNNER_GPU_CUDA
LDFLAGS += -ldl
TRAY_SRC = src/tray.c src/tray_stub.c
RUNNER_EXE = runner
TEST_JSON_SCHEMA = test-json-schema
TEST_TOKENIZER = test-tokenizer
TEST_TEMPLATE = test-template
TEST_TOOLS = test-tools
TEST_JSON_OOM = test-json-oom
TEST_TOKENIZER_OOM = test-tokenizer-oom
TEST_TEMPLATE_OOM = test-template-oom
TEST_SCHEMA_OOM = test-schema-oom
TEST_SAMPLER = test-sampler
TEST_SHARED = test-shared-weights
TEST_BATCH = test-batch
DIFFTOK = difftok
TEST_BIND = test-bind
endif

# same .exe suffix rule as every other test binary, without repeating the
# three-way platform branch above
TEST_PREFIX = $(TEST_BATCH:test-batch%=test-prefix%)
TEST_HOST_HEADER = $(TEST_BATCH:test-batch%=test-host-header%)
TEST_GRAMMAR_FF = $(TEST_BATCH:test-batch%=test-grammar-ff%)
TEST_VRAMREG = $(TEST_BATCH:test-batch%=test-vram-registry%)
TEST_KV_TOL = $(TEST_BATCH:test-batch%=test-kv-tol%)
TEST_QUANTS_SIMD = $(TEST_BATCH:test-batch%=test-quants-simd%)
TEST_INSTANCES = $(TEST_BATCH:test-batch%=test-instances%)
TEST_METAL_ADMISSION = $(TEST_BATCH:test-batch%=test-metal-admission%)
TEST_TRAY_CORE = $(TEST_BATCH:test-batch%=test-tray-core%)
TEST_TC_TOL = $(TEST_BATCH:test-batch%=test-tc-tol%)
TEST_I8_TOL = $(TEST_BATCH:test-batch%=test-i8-tol%)
TEST_MV_TOL = $(TEST_BATCH:test-batch%=test-mv-tol%)
TEST_ATTN_TOL = $(TEST_BATCH:test-batch%=test-attn-tol%)
TEST_GPU_ID = $(TEST_BATCH:test-batch%=test-gpu-identity%)
TEST_MOE_TOL = $(TEST_BATCH:test-batch%=test-moe-tol%)
TEST_MOE_ROUTER = $(TEST_BATCH:test-batch%=test-moe-router%)
TEST_PAGING_WARN = $(TEST_BATCH:test-batch%=test-paging-warn%)
TEST_AUTOFIT = $(TEST_BATCH:test-batch%=test-autofit%)
TEST_RESP_SM = $(TEST_BATCH:test-batch%=test-responses-sm%)
TEST_BUDGET = $(TEST_BATCH:test-batch%=test-prompt-budget%)
TEST_ATTRIB = $(TEST_BATCH:test-batch%=test-tool-attribution%)
TEST_STOP_CONSTRAINT = $(TEST_BATCH:test-batch%=test-stop-constraint%)
# not a test: the runner-side driver scripts/template-conformance.py renders
# through. Built here rather than by the script so it links the SAME object
# set the server does (see the template-conformance target near the bottom).
TMPL_CONF_RENDER = $(TEST_BATCH:test-batch%=template-conformance-render%)
# test_responses_sm drives the framer through a POSIX socketpair(); winsock
# has none, so on Windows the suite skips it LOUDLY (it runs in Linux CI and
# on the POSIX dev boxes) rather than shimming the transport under the test.
# test_tool_attribution reads its refusals back off the same kind of pair, so
# it skips on the same terms.
ifeq ($(OS),Windows_NT)
TEST_RESP_SM_DEP =
TEST_RESP_SM_RUN = @echo "skip: test-responses-sm (POSIX socketpair; covered by Linux CI)"
TEST_ATTRIB_DEP =
TEST_ATTRIB_RUN = @echo "skip: test-tool-attribution (POSIX socketpair; covered by Linux CI)"
TEST_MSG_OOM_DEP =
TEST_MSG_OOM_RUN = @echo "skip: test-prompt-oom (POSIX socketpair; covered by Linux CI)"
else
TEST_RESP_SM_DEP = $(TEST_RESP_SM)
TEST_RESP_SM_RUN = ./$(TEST_RESP_SM)
TEST_ATTRIB_DEP = $(TEST_ATTRIB)
TEST_ATTRIB_RUN = ./$(TEST_ATTRIB)
TEST_MSG_OOM_DEP = $(TEST_MSG_OOM)
TEST_MSG_OOM_RUN = ./$(TEST_MSG_OOM)
endif
TEST_QUANTIZE = $(TEST_BATCH:test-batch%=test-quantize%)
TEST_VRAM_ROLLBACK = $(TEST_BATCH:test-batch%=test-vram-rollback%)
TEST_GGUF_GETTERS = $(TEST_BATCH:test-batch%=test-gguf-getters%)
TEST_GGUF_SPLIT = $(TEST_BATCH:test-batch%=test-gguf-split-load%)
TEST_PARSE = $(TEST_BATCH:test-batch%=test-parse%)
TEST_METAL_OWNERSHIP = $(TEST_BATCH:test-batch%=test-metal-ownership%)
TEST_METAL_SHADERS = $(TEST_BATCH:test-batch%=test-metal-shaders%)
TEST_METAL_KQUANTS = $(TEST_BATCH:test-batch%=test-metal-kquants%)
TEST_MODEL_LOAD_FAILURE = $(TEST_BATCH:test-batch%=test-model-load-failure%)
TEST_THREAD_DEFAULT = $(TEST_BATCH:test-batch%=test-thread-default%)

# Every module header, so a change to any of them rebuilds. runner.h was one
# file until 0.1.5; the split (RNR-018) would otherwise have quietly narrowed
# what `make` considers a dependency.
HDR = $(wildcard src/*.h)

# Quant arithmetic is the numerical identity boundary.  Keep it in a separate
# translation unit with fast-math explicitly disabled; the rest of the engine
# retains -ffast-math.  The final -fno-fast-math also defeats a hostile or
# distro-supplied CFLAGS that enabled it before our filtered flags.
QUANTS_OBJ = .build/quants.o
QUANTS_CFLAGS = $(filter-out -ffast-math,$(CFLAGS)) -fno-fast-math

# Rebuild on every invocation: the requested CC/CFLAGS are not encoded in the
# object name, and reusing a native developer object in a portable release or
# cross build would be a correctness bug.
#
# .DEFAULT_GOAL, because this rule sits ABOVE `runner` and a rule is make's
# default goal by position: without the pin, bare `make` built ONE object
# file and exited 0. Local flows always name a target, so only CI's bare
# `make` jobs (windows smoke, consumer-compatibility) caught it.
.DEFAULT_GOAL := runner
$(QUANTS_OBJ): FORCE src/quants.c $(HDR)
	mkdir -p $(dir $@)
	$(CC) $(QUANTS_CFLAGS) -I src -c src/quants.c -o $@

SRC = src/gguf.c src/compat.c $(QUANTS_OBJ) src/instances.c src/tokenizer.c src/model.c src/sample.c \
      src/vramreg.c \
      src/template.c src/jsonmode.c src/schema.c src/quantize.c src/engine.c src/json.c src/http.c src/registry.c src/scheduler.c src/completion.c src/api_responses.c src/api_anthropic.c src/server.c \
      src/main.c $(GPU_SRC) $(TRAY_SRC)

# kernels_ptx.h is embedded into the binary by cuda.c — a pull that changes
# ONLY the regenerated PTX header must rebuild, or benchmarks silently run
# yesterday's kernels (this bit a publication run on 2026-07-29).
runner: $(SRC) $(HDR) src/kernels_ptx.h
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDFLAGS)

# Local negative controls for scripts/write-stall.py. These compile from the
# same sources and embedded kernels as runner; they are not release artifacts.
runner-no-write-timeout: $(SRC) $(HDR) src/kernels_ptx.h
	$(CC) $(CFLAGS) -DRUNNER_TEST_NO_WRITE_TIMEOUT $(SRC) -o $@ $(LDFLAGS)

runner-sigpipe-default: $(SRC) $(HDR) src/kernels_ptx.h
	$(CC) $(CFLAGS) -DRUNNER_TEST_SIGPIPE_DEFAULT $(SRC) -o $@ $(LDFLAGS)

debug: $(SRC) $(HDR)
	$(CC) -O0 -g -fsanitize=address,undefined -fno-fast-math -std=gnu11 -Wall \
		$(filter-out $(QUANTS_OBJ),$(SRC)) src/quants.c -o runner-debug $(LDFLAGS)

# $(CFLAGS), not a hand-rolled flag list: schema bounds compile in the same
# -ffast-math configuration as the shipped binary. Building this test without
# those flags once gated behavior in a configuration that does not ship.
$(TEST_JSON_SCHEMA): tests/test_json_schema.c src/json.c src/jsonmode.c src/schema.c $(HDR)
	$(CC) $(CFLAGS) -I src tests/test_json_schema.c src/json.c src/jsonmode.c src/schema.c -o $@ -lm

# quants.c is needed for the ggml_type_* helpers gguf.c links against; CFLAGS
# (not the plainer flags above) so the AVX2 paths match a real build
TEST_TOK_SRC = tests/test_tokenizer.c src/gguf.c src/tokenizer.c src/compat.c $(QUANTS_OBJ)
$(TEST_TOKENIZER): $(TEST_TOK_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_TOK_SRC) -o $@ -lm

# the two merge implementations must agree on every input; see the test's header
TEST_TOK_MERGE = $(TEST_BATCH:test-batch%=test-tokenizer-merge%)
TEST_TOK_MERGE_SRC = tests/test_tokenizer_merge.c src/gguf.c src/tokenizer.c \
                     src/compat.c $(QUANTS_OBJ)
$(TEST_TOK_MERGE): $(TEST_TOK_MERGE_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_TOK_MERGE_SRC) -o $@ -lm

$(TEST_GGUF_SPLIT): tests/test_gguf_split_load.c src/gguf.c src/compat.c $(QUANTS_OBJ) $(HDR)
	$(CC) $(CFLAGS) -I src tests/test_gguf_split_load.c src/gguf.c src/compat.c $(QUANTS_OBJ) -o $@ $(LDFLAGS)

# difftok: tokenizer differential harness. Not part of `make test` -- it needs a
# real multi-GB model GGUF, which models/ is gitignored for. scripts/difftok.py
# builds it on demand and compares against the HuggingFace reference.
DIFFTOK_SRC = tests/difftok.c src/gguf.c src/tokenizer.c src/compat.c $(QUANTS_OBJ)
$(DIFFTOK): $(DIFFTOK_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(DIFFTOK_SRC) -o $@ -lm

TEST_TMPL_SRC = tests/test_template.c src/gguf.c src/tokenizer.c src/template.c \
                src/json.c src/compat.c $(QUANTS_OBJ)
$(TEST_TEMPLATE): $(TEST_TMPL_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_TMPL_SRC) -o $@ -lm

# the strict tool envelope is only meaningful if the schema engine enforces
# it, so schema.c/jsonmode.c compile in and the tests drive the real validator
TEST_TOOLS_SRC = tests/test_tools.c src/gguf.c src/tokenizer.c src/template.c \
                 src/schema.c src/jsonmode.c src/json.c src/compat.c $(QUANTS_OBJ)
$(TEST_TOOLS): $(TEST_TOOLS_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_TOOLS_SRC) -o $@ -lm

# sampler presets and the greedy/penalty contract need no model, so the test
# links src/sample.c alone
$(TEST_SAMPLER): tests/test_sampler.c src/sample.c $(HDR)
	$(CC) $(CFLAGS) -I src tests/test_sampler.c src/sample.c -o $@ -lm

# compiles src/json.c directly into the test with instrumented allocators
$(TEST_JSON_OOM): tests/test_json_oom.c src/json.c src/json.h
	$(CC) $(CFLAGS) -I src tests/test_json_oom.c -o $@ -lm

# compiles src/tokenizer.c into the test with instrumented allocators; gguf.c
# and friends link normally so their allocations stay outside the failure window
TEST_TOK_OOM_SRC = tests/test_tokenizer_oom.c src/gguf.c src/compat.c $(QUANTS_OBJ)
$(TEST_TOKENIZER_OOM): $(TEST_TOK_OOM_SRC) src/tokenizer.c $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_TOK_OOM_SRC) -o $@ -lm

# same shape for the renderer: json.c and template.c are compiled into the test
# with instrumented allocators, tokenizer.c/gguf.c link normally
TEST_TMPL_OOM_SRC = tests/test_template_oom.c src/tokenizer.c src/gguf.c \
                    src/compat.c $(QUANTS_OBJ)
$(TEST_TEMPLATE_OOM): $(TEST_TMPL_OOM_SRC) src/template.c src/json.c $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_TMPL_OOM_SRC) -o $@ -lm

# schema.c and json.c both compile into the test: enum/const literals are
# serialised through jv_dump, so builder failures are schema failure paths
$(TEST_SCHEMA_OOM): tests/test_schema_oom.c src/schema.c src/json.c src/jsonmode.c $(HDR)
	$(CC) $(CFLAGS) -I src tests/test_schema_oom.c src/jsonmode.c -o $@ -lm

# shared model weights: needs the real model + backend, so it links the same
# sources the runner does minus the CLI/server front end
TEST_SHARED_SRC = tests/test_shared_weights.c src/gguf.c src/compat.c \
                  $(QUANTS_OBJ) src/model.c src/vramreg.c $(GPU_SRC)
$(TEST_SHARED): $(TEST_SHARED_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_SHARED_SRC) -o $@ $(LDFLAGS)

# ASAN_MODEL selects the fixture for the sharing / split / no-identity gates.
# It was UNSET, so all three silently fell back to the 370 KB test.gguf —
# which is physically incapable of moving a VRAM budget, overflowing a
# file-size field, or forcing an eviction. The gates were not weak; their input
# was, and a green run said nothing about that. Prefer a real model when one is
# on the box; fall back to the fixture otherwise, and SAY WHICH either way, so
# a pass is never mistaken for coverage it did not have.
ASAN_MODEL ?= $(firstword $(wildcard models/SmolLM2-135M-Instruct-Q8_0.gguf \
                                     models/e2b-q40.gguf))

.PHONY: fixture-scale-note
fixture-scale-note:
	@if [ -z "$(ASAN_MODEL)" ]; then \
	  echo "note: gates below run against the 370 KB test.gguf — too small to"; \
	  echo "      exercise VRAM budgets, eviction or file-size limits. Set"; \
	  echo "      ASAN_MODEL=<real .gguf> to make those contracts testable."; \
	else \
	  echo "note: gates below run against $(ASAN_MODEL)"; \
	fi

# same test under ASan/UBSan: the free-exactly-once half of it only fails
# loudly here. Kept out of `make test` because a sanitized model load is slow.
test-shared-asan: $(TEST_SHARED_SRC) $(HDR) test.gguf fixture-scale-note
	$(CC) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
	    -std=gnu11 -Wall -I src $(TEST_SHARED_SRC) -o test-shared-asan-bin $(LDFLAGS)
	RUNNER_TEST_GPU_OFF=1 LSAN_OPTIONS=suppressions=tests/lsan.supp \
	    ./test-shared-asan-bin $(ASAN_MODEL)

# Prove the sharing gate is not vacuous, at fixture scale and on any backend.
# With the file identity unavailable both instances load privately, so
# test-shared-weights MUST go red — and it must go red for the sharing reason,
# not something incidental. A green run here means the gate can no longer
# detect lost sharing, which is the state it was in for its whole life before
# 2026-08-04 (see the CHANGELOG entries for the shared-weights split fix).
test-shared-noid: $(TEST_SHARED) fixture-scale-note
	@set -e; \
	if RUNNER_TEST_NO_FILE_ID=1 ./$(TEST_SHARED) $(ASAN_MODEL) > shared-noid.out 2>&1; then \
		echo "FAIL: test-shared-weights passed with no file identity —"; \
		echo "      the sharing gate cannot detect lost sharing and is vacuous."; \
		cat shared-noid.out; exit 1; \
	fi; \
	grep -q "cannot be keyed by file identity" shared-noid.out || { \
		echo "FAIL: nothing on stderr named the lost file identity"; \
		cat shared-noid.out; exit 1; }; \
	grep -q "instances share one layer array" shared-noid.out || { \
		echo "FAIL: red, but not for the lost-sharing reason"; \
		cat shared-noid.out; exit 1; }; \
	echo "shared-weights no-identity gate ok (red as required, and said why)"

# file identity at real-checkpoint size: model_file_identity() must key a
# 5 GB file, because losing the identity is what silently unshares weights
# (the fixture-scale gates can never see the >2 GB stat() cliff that did it)
TEST_FILE_ID = $(TEST_BATCH:test-batch%=test-file-identity%)
TEST_FILE_ID_SRC = tests/test_file_identity.c src/gguf.c src/compat.c \
                   $(QUANTS_OBJ) src/model.c src/vramreg.c $(GPU_SRC)
$(TEST_FILE_ID): $(TEST_FILE_ID_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_FILE_ID_SRC) -o $@ $(LDFLAGS)

# A bare `runner` in a terminal starts the tray; a bare `runner` anywhere
# else must NOT — a GUI event loop on a piped stdin would hang every script
# that ever probes the binary, including this test. Piping stdin here is
# therefore both the test setup and the property under test. --no-tray is the
# documented spelling of the same suppression and must stay a known flag.
# The binary must carry the kernels currently in the tree. check-generated.py
# compares src/kernels_metal.h to src/kernels.metal, but neither can see a
# STALE BINARY, and a kernel measured against one looks exactly like an honest
# result: on 2026-08-07 a q4_K change measured as no effect and a q5_K change
# as +4.6%, both against builds that did not contain them. make rebuilds
# correctly on a header change; the hazard is an A/B whose two builds land in
# the same second, where make's newer-than test keeps the old binary.
# Skips cleanly on a build with no embedded shader source (CUDA/CPU-only).
test-shader-embed: runner
	@set -e; \
	command -v $(PYTHON) >/dev/null 2>&1 || { \
		echo "shader embed: skip (no $(PYTHON) on this box)"; exit 0; }; \
	caps=$$(./runner --caps 2>/dev/null); \
	have=$$(printf '%s' "$$caps" | $(PYTHON) -c "import sys,json; g=json.load(sys.stdin).get('gpu') or {}; print(g.get('shader_source_sha256') or '')"); \
	if [ -z "$$have" ]; then echo "shader embed: skip (no embedded shader source in this build)"; exit 0; fi; \
	want=$$($(PYTHON) -c "import hashlib;print(hashlib.sha256(open('src/kernels.metal',encoding='utf-8').read().encode()).hexdigest())"); \
	if [ "$$have" != "$$want" ]; then \
		echo "FAIL: runner was built from different Metal shaders than src/kernels.metal"; \
		echo "  binary: $$have"; echo "  source: $$want"; \
		echo "  re-run scripts/embed-metal.py and rebuild"; exit 1; fi; \
	echo "shader embed ok (binary carries src/kernels.metal, $${have}...)"

test-bare-invocation: runner
	@set -e; \
	rc=0; out=$$(echo "" | ./runner 2>&1) || rc=$$?; \
	echo "$$out" | grep -q "usage:" || { \
		echo "FAIL: bare runner with piped stdin did not print usage"; exit 1; }; \
	[ $$rc -ne 0 ] || { echo "FAIL: bare runner exited 0 without doing anything"; exit 1; }; \
	out=$$(echo "" | ./runner --no-tray 2>&1) || true; \
	echo "$$out" | grep -q "unknown option" && { \
		echo "FAIL: --no-tray is not a recognized flag"; exit 1; }; \
	echo "$$out" | grep -q "usage:" || { \
		echo "FAIL: bare --no-tray did not print usage"; exit 1; }; \
	echo "bare invocation ok (non-tty gets usage, --no-tray recognized)"

# split-guard harness: same link as the shared-weights test — the guard lives
# in the GPU registry, so it needs the real backend
# NOT `test-split-guard%`: on POSIX $(TEST_BATCH) is `test-batch`, so that
# substitution produced `test-split-guard` — colliding with the .PHONY test
# target of the same name below. make then dropped this compile recipe
# ("overriding commands"), reported a circular self-dependency, and the guard
# binary was never built. The guard whose comment says "delete the guard and
# this goes red" could not go red, and `make test` never referenced it at all.
# Windows was unaffected only because `test-batch.exe` yields a distinct name.
TEST_SPLIT_GUARD = $(TEST_BATCH:test-batch%=test-split-guard-bin%)
TEST_SPLIT_GUARD_SRC = tests/test_split_guard.c src/gguf.c src/compat.c \
                       $(QUANTS_OBJ) src/model.c src/vramreg.c $(GPU_SRC)
$(TEST_SPLIT_GUARD): $(TEST_SPLIT_GUARD_SRC) $(HDR)
	$(CC) $(CFLAGS) $(GPU_BACKEND_DEF) -I src $(TEST_SPLIT_GUARD_SRC) -o $@ $(LDFLAGS)

# The split guard must be able to fire: a no-identity load of an already-
# resident path, forced to a different split, must be reported loudly. The
# harness self-skips (exit 0, says so) without a GPU backend; when it does
# run, the report line is the gate — delete the guard and this goes red.
test-split-guard: $(TEST_SPLIT_GUARD) test.gguf
	@set -e; \
	./$(TEST_SPLIT_GUARD) $(ASAN_MODEL) > split-guard.out 2>&1; \
	if grep -q "^skip:" split-guard.out; then cat split-guard.out; exit 0; fi; \
	grep -q "re-decided the CPU/GPU split without a file identity" split-guard.out || { \
		echo "FAIL: forced split disagreement produced no loud report"; \
		cat split-guard.out; exit 1; }; \
	echo "split guard ok (no-identity split disagreement is reported loudly)"

# batched decode: same sources as the shared-weights test (real model +
# backend), because the property under test is a backend property
TEST_BATCH_SRC = tests/test_batch.c src/gguf.c src/compat.c \
                 $(QUANTS_OBJ) src/model.c src/vramreg.c $(GPU_SRC)
$(TEST_BATCH): $(TEST_BATCH_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_BATCH_SRC) -o $@ $(LDFLAGS)

# the loopback-only bind. Links nothing: it reads src/server.c and src/main.c
# and interrogates the built ./runner, so a bind address introduced anywhere --
# constant, flag, env var -- trips it. Depends on `runner` because the CLI half
# of the check needs the shipped binary rather than a comment about it.
$(TEST_BIND): tests/test_bind.c src/server.c src/main.c
	$(CC) $(CFLAGS) -I src tests/test_bind.c -o $@

$(TEST_HOST_HEADER): tests/test_host_header.c src/http.c src/json.c src/compat.c $(HDR)
	$(CC) $(CFLAGS) -I src tests/test_host_header.c src/http.c src/json.c \
		src/compat.c -o $@ $(LDFLAGS)

# forkable KV prefixes: needs the real model, the real tokenizer and the real
# engine, because the property under test is that a forked cache produces the
# same logits the model would have produced by prefilling
TEST_PREFIX_SRC = tests/test_prefix.c src/gguf.c src/compat.c $(QUANTS_OBJ) \
                  src/tokenizer.c src/model.c src/sample.c src/jsonmode.c \
                  src/schema.c src/json.c src/engine.c src/vramreg.c $(GPU_SRC)
$(TEST_PREFIX): $(TEST_PREFIX_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_PREFIX_SRC) -o $@ $(LDFLAGS)

# grammar fast-forward: same full-engine link as the prefix test — the gate
# is byte identity of a real constrained generation with the walk on and off
TEST_GRAMMAR_FF_SRC = tests/test_grammar_ff.c src/gguf.c src/compat.c \
                  $(QUANTS_OBJ) src/tokenizer.c src/model.c src/sample.c \
                  src/jsonmode.c src/schema.c src/json.c src/engine.c \
                  src/vramreg.c $(GPU_SRC)
$(TEST_GRAMMAR_FF): $(TEST_GRAMMAR_FF_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_GRAMMAR_FF_SRC) -o $@ $(LDFLAGS)

# the cross-process VRAM registry. Links only vramreg.c and compat.c: the
# free-VRAM figure arrives through a callback, so the whole module is drivable
# with synthetic numbers and the test needs no GPU, no model and no driver --
# which is what lets it run in CI.
$(TEST_VRAMREG): tests/test_vram_registry.c src/vramreg.c src/compat.c $(HDR)
	$(CC) $(CFLAGS) -I src tests/test_vram_registry.c src/vramreg.c src/compat.c -o $@ $(LDFLAGS)

# q8 KV tolerance gate: needs the tokenizer too, because it teacher-forces a
# fixed piece of real text rather than synthetic token ids
TEST_KV_TOL_SRC = tests/test_kv_tol.c src/gguf.c src/compat.c $(QUANTS_OBJ) \
                  src/tokenizer.c src/model.c src/vramreg.c $(GPU_SRC)
$(TEST_KV_TOL): $(TEST_KV_TOL_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_KV_TOL_SRC) -o $@ $(LDFLAGS)

# SIMD (AVX2/NEON) dot and dequant kernels vs an independent double-precision
# reference; also pins q8_quant_row byte-identical to its scalar definition
TEST_QUANTS_SIMD_SRC = tests/test_quants_simd.c $(QUANTS_OBJ)
# QUANTS_CFLAGS, not CFLAGS: this test carries inline scalar REFERENCE
# implementations of the quant kernels, and a reference compiled under
# -ffast-math is checking the confined -fno-fast-math engine against a
# differently-rounded definition of itself. Caught on ARM the day the
# int8 activation quantizer (built on x86) met the fast-math confinement
# (built the same night): identical scalar source, two float regimes,
# constructed half-way ties diverged. The verifier must share the
# engine's float semantics or it verifies nothing.
$(TEST_QUANTS_SIMD): $(TEST_QUANTS_SIMD_SRC) $(HDR)
	$(CC) $(QUANTS_CFLAGS) -I src $(TEST_QUANTS_SIMD_SRC) -o $@ -lm -lpthread

# discovery registry: pure-C, runs against a private HOME/APPDATA
TEST_INSTANCES_SRC = tests/test_instances.c src/instances.c src/json.c
$(TEST_INSTANCES): $(TEST_INSTANCES_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_INSTANCES_SRC) -o $@ -lm

# the same registry under injected allocation failure. instances.c is compiled
# INTO the test (macro-substituted allocators), so it is deliberately absent
# from the source list here.
TEST_INSTANCES_OOM = $(TEST_BATCH:test-batch%=test-instances-oom%)
$(TEST_INSTANCES_OOM): tests/test_instances_oom.c src/instances.c src/json.c $(HDR)
	$(CC) $(CFLAGS) -I src tests/test_instances_oom.c src/json.c -o $@ -lm

# Pure policy seam: simulates a Mac whose model is larger than one MTLBuffer
# but still inside the aggregate Metal working set.
$(TEST_METAL_ADMISSION): tests/test_metal_admission.c src/metal_admission.h
	$(CC) $(CFLAGS) -I src tests/test_metal_admission.c -o $@

# links the stub backend on EVERY platform: this gate tests the portable
# core (menu model, managed spawn/stop, quit semantics), not the GUI.
# ws2_32: the core's /v1/models enrichment uses sockets on Windows
ifeq ($(OS),Windows_NT)
TRAY_TEST_LIBS = -lws2_32
else
TRAY_TEST_LIBS =
endif
TEST_TRAY_CORE_SRC = tests/test_tray_core.c src/tray.c src/tray_stub.c \
                     src/instances.c src/json.c
$(TEST_TRAY_CORE): $(TEST_TRAY_CORE_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_TRAY_CORE_SRC) -o $@ -lm $(TRAY_TEST_LIBS)

# TC tolerance gate: same shape as the q8-KV gate — teacher-forced logits,
# top-1 + bounded-deviation criteria, per (type, arch) via the model argument
TEST_TC_TOL_SRC = tests/test_tc_tol.c src/gguf.c src/compat.c $(QUANTS_OBJ) \
                  src/tokenizer.c src/model.c src/vramreg.c $(GPU_SRC)
$(TEST_TC_TOL): $(TEST_TC_TOL_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_TC_TOL_SRC) -o $@ $(LDFLAGS)

# fused-int8 CPU dot tolerance gate: the CPU twin of the TC gate — teacher-
# forced logits, 0/64 top-1 flips + bounded deviation, per (type, model) via
# the model argument. CPU-only, so it needs no GPU backend to be meaningful.
TEST_I8_TOL_SRC = tests/test_i8_tol.c src/gguf.c src/compat.c src/quants.c \
                  src/tokenizer.c src/model.c src/vramreg.c $(GPU_SRC)
$(TEST_I8_TOL): $(TEST_I8_TOL_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_I8_TOL_SRC) -o $@ $(LDFLAGS)

# tolerance-gated Metal decode matvec: the GPU twin of the i8 gate — teacher-
# forced logits, 0/64 top-1 flips + bounded deviation, per (type, model) via
# the model argument. Self-skips on backends with no fast matvec kernel.
TEST_MV_TOL_SRC = tests/test_mv_tol.c src/gguf.c src/compat.c $(QUANTS_OBJ) \
                  src/tokenizer.c src/model.c src/vramreg.c $(GPU_SRC)
$(TEST_MV_TOL): $(TEST_MV_TOL_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_MV_TOL_SRC) -o $@ $(LDFLAGS)

# cooperative-KV attention read: same shape as the fast-matvec gate, for the
# other reassociating Metal route. Self-skips where it never dispatches.
TEST_ATTN_TOL_SRC = tests/test_attn_tol.c src/gguf.c src/compat.c $(QUANTS_OBJ) \
                    src/tokenizer.c src/model.c src/vramreg.c $(GPU_SRC)
$(TEST_ATTN_TOL): $(TEST_ATTN_TOL_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_ATTN_TOL_SRC) -o $@ $(LDFLAGS)

# CPU/GPU byte identity at LOGIT precision. The text-comparison gates
# (test-metal-kquant and friends) are sound on real models and blind on toy
# fixtures, where greedy argmax absorbs large numeric differences -- so a
# fixture-scale backend feature could be wrong and pass everything. This
# compares the logit vectors themselves.
TEST_GPU_ID_SRC = tests/test_gpu_identity.c src/gguf.c src/compat.c $(QUANTS_OBJ) \
                  src/tokenizer.c src/model.c src/vramreg.c $(GPU_SRC)
$(TEST_GPU_ID): $(TEST_GPU_ID_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_GPU_ID_SRC) -o $@ $(LDFLAGS)

# fused-vs-eager MoE routing tolerance: same full-engine link as tc-tol, and
# the same self-skipping shape (no GPU / not MoE / no full offload / the fused
# router never engaged all skip rather than pass)
TEST_MOE_TOL_SRC = tests/test_moe_tol.c src/gguf.c src/compat.c $(QUANTS_OBJ) \
                  src/tokenizer.c src/model.c src/vramreg.c $(GPU_SRC)
$(TEST_MOE_TOL): $(TEST_MOE_TOL_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_MOE_TOL_SRC) -o $@ $(LDFLAGS)

TEST_MOE_ROUTER_SRC = tests/test_moe_router.c src/gguf.c src/compat.c $(QUANTS_OBJ) \
                  src/tokenizer.c src/model.c src/vramreg.c $(GPU_SRC)
$(TEST_MOE_ROUTER): $(TEST_MOE_ROUTER_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_MOE_ROUTER_SRC) -o $@ $(LDFLAGS)

# residency-warning wording: needs the loader (the hot-set estimate has to
# agree with the actual tensor set), so it takes the same link as the two above
TEST_PAGING_WARN_SRC = tests/test_paging_warn.c src/gguf.c src/compat.c $(QUANTS_OBJ) \
                  src/tokenizer.c src/model.c src/vramreg.c $(GPU_SRC)
$(TEST_PAGING_WARN): $(TEST_PAGING_WARN_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_PAGING_WARN_SRC) -o $@ $(LDFLAGS)
# reservation auto-fit arithmetic: no model file, no GPU, no fixture -- the
# regime it covers is unreachable on a dev machine, so it is fed real 7B/24 GB
# numbers directly. See the header of tests/test_autofit.c.
TEST_AUTOFIT_SRC = tests/test_autofit.c src/gguf.c src/compat.c $(QUANTS_OBJ) \
                  src/tokenizer.c src/model.c src/vramreg.c $(GPU_SRC)
$(TEST_AUTOFIT): $(TEST_AUTOFIT_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_AUTOFIT_SRC) -o $@ $(LDFLAGS)
# the Responses framing state machine, driven directly over a socketpair.
# Includes completion.c (the framer is static there) and links the engine
# around it — the runner's object set minus main.c, server.c and the file the
# test itself includes.
TEST_RESP_SM_SRC = tests/test_responses_sm.c src/gguf.c src/compat.c \
                  $(QUANTS_OBJ) src/tokenizer.c src/model.c src/sample.c \
                  src/jsonmode.c src/schema.c src/json.c src/engine.c \
                  src/template.c src/vramreg.c src/http.c src/registry.c src/scheduler.c $(GPU_SRC)
$(TEST_RESP_SM): $(TEST_RESP_SM_SRC) src/completion.c $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_RESP_SM_SRC) -o $@ $(LDFLAGS)

# a client "stop" sequence under constrained output. Same link as the framing
# test above and for the same reason -- gen_ctx and stop_feed are static in
# completion.c, and the property under test is that the engine's validator and
# that sink agree on what the client received, so neither side may be stubbed.
TEST_STOP_CONSTRAINT_SRC = tests/test_stop_constraint.c src/gguf.c src/compat.c \
                  $(QUANTS_OBJ) src/tokenizer.c src/model.c src/sample.c \
                  src/jsonmode.c src/schema.c src/json.c src/engine.c \
                  src/template.c src/vramreg.c src/http.c src/registry.c \
                  src/scheduler.c $(GPU_SRC)
$(TEST_STOP_CONSTRAINT): $(TEST_STOP_CONSTRAINT_SRC) src/completion.c $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_STOP_CONSTRAINT_SRC) -o $@ $(LDFLAGS)

# the per-request prompt BUDGET, on all three tool-calling surfaces. The
# renderer truncates silently at its cap, so a buffer estimate that misses a
# term drops the tail of the prompt -- the user's own turn -- with no error.
# handle_chat is static in server.c, so that TU is #included by the test and
# left out of the link; completion.c is out too, because the test captures the
# prompt instead of generating from it.
TEST_BUDGET_SRC = tests/test_prompt_budget.c src/gguf.c src/compat.c \
                  $(QUANTS_OBJ) src/tokenizer.c src/model.c src/sample.c \
                  src/jsonmode.c src/schema.c src/json.c src/engine.c \
                  src/template.c src/vramreg.c src/http.c src/registry.c \
                  src/scheduler.c src/api_responses.c src/api_anthropic.c \
                  $(GPU_SRC)
$(TEST_BUDGET): $(TEST_BUDGET_SRC) src/server.c $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_BUDGET_SRC) -o $@ $(LDFLAGS)

# a replayed tool result must name the function it came from, on all three
# tool-calling surfaces. Same recipe as the budget test -- handle_chat is
# static in server.c, so that TU is #included by the test and left out of the
# link -- plus a socketpair, because half of what is asserted is the 400 a
# request that cannot be attributed gets back.
TEST_ATTRIB_SRC = tests/test_tool_attribution.c src/gguf.c src/compat.c \
                  $(QUANTS_OBJ) src/tokenizer.c src/model.c src/sample.c \
                  src/jsonmode.c src/schema.c src/json.c src/engine.c \
                  src/template.c src/vramreg.c src/http.c src/registry.c \
                  src/scheduler.c src/api_responses.c src/api_anthropic.c \
                  $(GPU_SRC)
$(TEST_ATTRIB): $(TEST_ATTRIB_SRC) src/server.c $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_ATTRIB_SRC) -o $@ $(LDFLAGS)

# The same inbound translations under injected allocation failure. Same source
# set minus the three files that DO the translating -- server.c, api_responses.c
# and api_anthropic.c are compiled INTO the test with substituted allocators.
TEST_MSG_OOM = $(TEST_BATCH:test-batch%=test-prompt-oom%)
TEST_MSG_OOM_SRC = tests/test_prompt_oom.c \
                   $(filter-out src/api_anthropic.c src/api_responses.c \
                                src/json.c,\
                     $(TEST_ATTRIB_SRC:tests/test_tool_attribution.c=))
$(TEST_MSG_OOM): $(TEST_MSG_OOM_SRC) src/server.c src/api_anthropic.c \
                 src/api_responses.c src/json.c $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_MSG_OOM_SRC) -o $@ $(LDFLAGS)

# The runner side of the template-conformance gate. Same recipe as the two
# tests above, and for the same reason: the flattening from an OpenAI request
# to the renderer's flat turns is handle_chat's, it is static in server.c, and
# a harness that RE-IMPLEMENTED it would be comparing jinja against a copy of
# runner rather than against runner. So server.c is #included and the
# generation path is stubbed out of the link -- the prompt is the artifact,
# nothing is generated from it.
#
# api_responses.c / api_anthropic.c are additionally left out: only server.c's
# route table refers to them, the gate never reaches it, and dropping them
# keeps the add_generation_prompt interception (see the driver's header
# comment) confined to one translation unit.
TMPL_CONF_RENDER_SRC = scripts/template-conformance-render.c src/gguf.c \
                  src/compat.c $(QUANTS_OBJ) src/tokenizer.c src/model.c \
                  src/sample.c src/jsonmode.c src/schema.c src/json.c \
                  src/engine.c src/template.c src/vramreg.c src/http.c \
                  src/registry.c src/scheduler.c $(GPU_SRC)
$(TMPL_CONF_RENDER): $(TMPL_CONF_RENDER_SRC) src/server.c $(HDR)
	$(CC) $(CFLAGS) -I src $(TMPL_CONF_RENDER_SRC) -o $@ $(LDFLAGS)

# server_run twice in one process: the property a once-per-process global can
# hide forever, because nothing ever asks the state to come back.
TEST_RESTART = $(TEST_BATCH:test-batch%=test-server-restart%)
TEST_RESTART_SRC = tests/test_server_restart.c src/gguf.c src/compat.c \
                   $(QUANTS_OBJ) src/tokenizer.c src/model.c src/sample.c \
                   src/jsonmode.c src/schema.c src/json.c src/engine.c \
                   src/template.c src/vramreg.c src/http.c src/registry.c \
                   src/scheduler.c src/completion.c src/api_responses.c \
                   src/api_anthropic.c src/server.c $(GPU_SRC)
$(TEST_RESTART): $(TEST_RESTART_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_RESTART_SRC) -o $@ $(LDFLAGS)

# /v1/capabilities is answered on the accept thread and dereferences the
# resident model; a slot thread frees that model on any swap. Only ASan makes
# the read of freed bytes loud, so this gate is BUILT sanitized rather than
# added to `make test` -- same shape as test-shared-asan above. Kept out of the
# default run because it starts a real server and drives it for a few seconds.
TEST_SWAP_RACE_SRC = tests/test_swap_race.c src/gguf.c src/compat.c \
                     src/quants.c src/tokenizer.c src/model.c src/sample.c \
                     src/jsonmode.c src/schema.c src/json.c src/engine.c \
                     src/template.c src/vramreg.c src/http.c src/registry.c \
                     src/scheduler.c src/completion.c src/api_responses.c \
                     src/api_anthropic.c src/server.c $(GPU_SRC)
test-swap-race: $(TEST_SWAP_RACE_SRC) $(HDR) test.gguf
	$(CC) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
	    -fno-fast-math -std=gnu11 -Wall -I src $(TEST_SWAP_RACE_SRC) \
	    -o test-swap-race-bin $(LDFLAGS)
	ASAN_OPTIONS=detect_leaks=0 ./test-swap-race-bin test.gguf

# the weight-residency platform layer: mlock, mincore, major faults, available
# RAM. compat.c only -- these are platform shims, not engine code.
TEST_RESIDENCY = $(TEST_BATCH:test-batch%=test-residency%)
$(TEST_RESIDENCY): tests/test_residency.c src/compat.c $(HDR)
	$(CC) $(CFLAGS) -I src tests/test_residency.c src/compat.c -o $@ $(LDFLAGS)

# the device turn is FIFO, not just exclusive. scheduler.c is #included by the
# test (the turnstile is static) so it is NOT linked here.
TEST_SCHED_TURN = $(TEST_BATCH:test-batch%=test-sched-turn%)
TEST_SCHED_TURN_SRC = tests/test_sched_turn.c src/gguf.c src/compat.c \
                      $(QUANTS_OBJ) src/tokenizer.c src/model.c src/sample.c \
                      src/jsonmode.c src/schema.c src/json.c src/engine.c \
                      src/template.c src/vramreg.c src/http.c src/registry.c $(GPU_SRC)
$(TEST_SCHED_TURN): $(TEST_SCHED_TURN_SRC) src/scheduler.c $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_SCHED_TURN_SRC) -o $@ $(LDFLAGS)

# snapshot persistence: the round trip, and the refusals that matter more
TEST_PFX_PERSIST = $(TEST_BATCH:test-batch%=test-prefix-persist%)
TEST_PFX_PERSIST_SRC = tests/test_prefix_persist.c src/gguf.c src/compat.c \
                       $(QUANTS_OBJ) src/tokenizer.c src/model.c src/sample.c \
                       src/jsonmode.c src/schema.c src/json.c src/engine.c \
                       src/vramreg.c $(GPU_SRC)
$(TEST_PFX_PERSIST): $(TEST_PFX_PERSIST_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_PFX_PERSIST_SRC) -o $@ $(LDFLAGS)

TEST_QUANTIZE_SRC = tests/test_quantize.c src/quantize.c src/gguf.c \
                    src/compat.c $(QUANTS_OBJ) src/json.c
$(TEST_QUANTIZE): $(TEST_QUANTIZE_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_QUANTIZE_SRC) -o $@ $(LDFLAGS)

# vramreg.c is #included (calloc-hooked) by the test, so it is not linked here
$(TEST_VRAM_ROLLBACK): tests/test_vram_rollback.c src/compat.c $(HDR)
	$(CC) $(CFLAGS) -I src tests/test_vram_rollback.c src/compat.c -o $@ $(LDFLAGS)

$(TEST_GGUF_GETTERS): tests/test_gguf_getters.c src/gguf.c src/compat.c $(QUANTS_OBJ) $(HDR)
	$(CC) $(CFLAGS) -I src tests/test_gguf_getters.c src/gguf.c src/compat.c $(QUANTS_OBJ) -o $@ $(LDFLAGS)

$(TEST_PARSE): tests/test_parse.c src/compat.c src/compat.h
	$(CC) $(CFLAGS) -I src tests/test_parse.c src/compat.c -o $@ $(LDFLAGS)

# quants.c joins for tpool_create/tpool_destroy: the test now also pins that an
# over-large -t is clamped to TP_MAX rather than silently discarded.
$(TEST_THREAD_DEFAULT): tests/test_thread_default.c src/compat.c src/compat.h $(QUANTS_OBJ)
	$(CC) $(CFLAGS) -I src tests/test_thread_default.c src/compat.c $(QUANTS_OBJ) -o $@ $(LDFLAGS)

TEST_MODEL_LOAD_FAILURE_SRC = tests/test_model_load_failure.c src/gguf.c \
                              src/compat.c $(QUANTS_OBJ) src/model.c \
                              src/vramreg.c $(GPU_SRC)
$(TEST_MODEL_LOAD_FAILURE): $(TEST_MODEL_LOAD_FAILURE_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_MODEL_LOAD_FAILURE_SRC) -o $@ $(LDFLAGS)

test.gguf: scripts/make-test-model.py
	$(PYTHON) scripts/make-test-model.py test.gguf

# Ornith/Qwen3.5 CPU tracer: a committed generator builds a tiny hybrid model
# with three recurrent DeltaNet blocks and one full-attention block.
test-ornith-cpu: runner
	$(PYTHON) -m pytest -q tests/test_ornith_cpu.py

test-apertus: runner
	$(PYTHON) -m pytest -q tests/test_apertus.py

test-moe: runner
	$(PYTHON) -m pytest -q tests/test_moe.py

# --prune-experts: stacked-layout MoE expert pruning in the quantize path.
test-prune-experts: runner
	$(PYTHON) -m pytest -q tests/test_prune_experts.py

test-weight-io-bench:
	$(PYTHON) -m pytest -q tests/test_weight_io_bench.py

$(TEST_METAL_SHADERS): tests/test_metal_shaders.m src/kernels_metal.h
	$(CC) -std=gnu11 -Wall -Wextra -Wno-unused-parameter -I src \
	    tests/test_metal_shaders.m -o $@ -framework Metal -framework Foundation

$(TEST_METAL_KQUANTS): tests/test_metal_kquants.m src/kernels_metal.h $(QUANTS_OBJ) $(HDR)
	$(CC) -std=gnu11 -Wall -Wextra -Wno-unused-parameter -I src \
	    tests/test_metal_kquants.m $(QUANTS_OBJ) -o $@ -lm -lpthread \
	    -framework Metal -framework Foundation

# compat.c joins the link because the partial-offload residency guard in
# metal.m calls plat_ram_available_bytes(): deciding whether a split pays
# needs to know how much RAM the CPU tail would have to stream through.
$(TEST_METAL_OWNERSHIP): tests/test_metal_ownership.m src/metal.m src/compat.c $(HDR)
	$(CC) -std=gnu11 -Wall -Wextra -Wno-unused-parameter -I src \
	    tests/test_metal_ownership.m src/compat.c -o $@ $(LDFLAGS)

# Runs inside `make test` on macOS: a shader that does not compile makes every
# run fall back to the CPU silently, which no correctness gate can see.
test-metal-shader-gate:
ifeq ($(shell uname -s),Darwin)
	@$(MAKE) --no-print-directory $(TEST_METAL_SHADERS) $(TEST_METAL_KQUANTS) >/dev/null
	@./$(TEST_METAL_SHADERS)
	@./$(TEST_METAL_KQUANTS)
else
	@echo "metal shader gate skipped: macOS-only backend"
endif

test-metal-fallback: runner test.gguf
ifeq ($(shell uname -s),Darwin)
	$(MAKE) --no-print-directory $(TEST_METAL_OWNERSHIP)
	./$(TEST_METAL_OWNERSHIP)
	@set -e; \
	if ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
		./$(RUNNER_EXE) -m test.gguf -p "hello" -n 8 --temp 0 --gpu off > metal-cpu.out 2>/dev/null; \
		env RUNNER_METAL_INJECT_FAILURE=once MallocScribble=1 MallocGuardEdges=1 \
		    env RUNNER_METAL_MM=0 RUNNER_METAL_ATTN_COOP=0 ./$(RUNNER_EXE) -m test.gguf -p "hello" -n 8 --temp 0 --gpu auto > metal-fallback.out 2> metal-fallback.err; \
		cmp -s metal-cpu.out metal-fallback.out; \
		grep -q "falling back to CPU" metal-fallback.err; \
		env RUNNER_METAL_INIT_INJECT_FAILURE=after-kv MallocScribble=1 MallocGuardEdges=1 \
		    env RUNNER_METAL_MM=0 RUNNER_METAL_ATTN_COOP=0 ./$(RUNNER_EXE) -m test.gguf -p "hello" -n 8 --temp 0 --gpu auto > metal-init-fallback.out 2> metal-init-fallback.err; \
		cmp -s metal-cpu.out metal-init-fallback.out; \
		grep -q "Metal initialization failed" metal-init-fallback.err; \
		echo "metal fallback ownership ok"; \
	else \
		echo "metal fallback runtime smoke skipped: no Metal device reported by --caps"; \
	fi
else
	@echo "metal fallback tests skipped: macOS-only backend"
endif

# Byte-identity for the Metal paths that only just got new kernels or widening:
# q2_K/q3_K/iq4_xs tiled+matvec coverage, and f16/bf16 matvec widening.
#
# They are tested TOGETHER because a checkpoint exercising only one does not
# exist in the wild: every real "Q2_K" GGUF is a mix, and llama.cpp's mix pairs
# q2_K with q3_K (measured: tinyllama-1.1b Q2_K is 45 q2_K + 110 q3_K tensors).
# Shipping q2_K alone would have run exactly nothing. f16 and bf16 are included
# here because their widened decode kernels carry the broadest blast radius.
#
# Short AND long prompts, because they take different code paths: a batch of
# more than one token uses the tiled GEMM (k_mm_*) and decode uses the matvec
# (k_mv_*), so a short-prompt-only check leaves half of each format's kernels
# ungated. A long prompt also spans several K-tiles and column-tiles rather
# than one.
#
# The checkpoint is not in the repo, so this skips loudly rather than passing
# vacuously -- and if the model IS present but falls back to CPU, that is a
# FAILURE, not a skip: a parity check with both sides on the CPU compares
# nothing at all, which is the exact defect class the 2026-08-09 gate audit
# found three times over.
KQUANT_MODELS ?= $(wildcard models/tinyllama-q2k.gguf models/*Q2_K*.gguf \
                   models/*q2_k*.gguf models/*IQ4_XS*.gguf models/*iq4xs*.gguf \
                   models/*-f16*.gguf models/*-F16*.gguf \
                   models/*_f16*.gguf models/*_F16*.gguf \
                   models/*bf16*.gguf models/*BF16*.gguf)
test-metal-kquant: runner
ifeq ($(shell uname -s),Darwin)
	@set -e; \
	if [ -z "$(KQUANT_MODELS)" ]; then \
	  echo "metal quant parity: SKIP (no q2_K/q3_K/iq4_xs/f16/bf16 checkpoint in models/)"; \
	  exit 0; fi; \
	if ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
	  long=$$($(PYTHON) -c "print(' '.join(['the quick brown fox jumps over the lazy dog']*40))"); \
	  for m in $(KQUANT_MODELS); do \
	    for which in short long; do \
	      if [ $$which = short ]; then prompt="The capital of France is"; \
	      else prompt="$$long"; fi; \
	      ./$(RUNNER_EXE) -m $$m -p "$$prompt" -n 12 --temp 0 --gpu off \
	        > metal-kquant-cpu.out 2>/dev/null; \
	      env RUNNER_METAL_ATTN_COOP=0 ./$(RUNNER_EXE) -m $$m -p "$$prompt" -n 12 --temp 0 --gpu auto \
	        > metal-kquant-gpu.out 2> metal-kquant-gpu.err; \
	      if grep -q "without a Metal kernel" metal-kquant-gpu.err; then \
	        echo "FAIL: $$m fell back to CPU — this parity check would compare"; \
	        echo "  the CPU against itself and pass for the wrong reason"; \
	        exit 1; fi; \
	      grep -q "Metal backend" metal-kquant-gpu.err; \
	      cmp -s metal-kquant-cpu.out metal-kquant-gpu.out || { \
	        echo "FAIL: $$m Metal output differs from CPU ($$which prompt)"; \
	        exit 1; }; \
	    done; \
	    echo "  metal quant parity ok ($$m, byte-identical, short+long)"; \
	  done; \
	else \
	  echo "metal quant parity: SKIP (no Metal device reported by --caps)"; \
	fi
else
	@echo "metal quant parity: SKIP (macOS-only backend)"
endif

test-metal-prefill: runner test.gguf
ifeq ($(shell uname -s),Darwin)
	@set -e; \
	if ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
		prompt="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"; \
		./$(RUNNER_EXE) -m test.gguf -p "$$prompt" -n 8 -b 8 --temp 0 --gpu off > metal-prefill-cpu.out 2>/dev/null; \
		env RUNNER_METAL_STATS=1 RUNNER_METAL_MM=0 ./$(RUNNER_EXE) -m test.gguf -p "$$prompt" -n 8 -b 8 --temp 0 --gpu auto > metal-prefill-native.out 2> metal-prefill-native.err; \
		cmp -s metal-prefill-cpu.out metal-prefill-native.out; \
		grep -q "metal: native prompt batch" metal-prefill-native.err; \
		grep -q "Metal backend" metal-prefill-native.err; \
		echo "metal prompt batch ok"; \
	else \
		echo "metal prompt batch smoke skipped: no Metal device reported by --caps"; \
	fi
else
	@echo "metal prompt batch smoke skipped: macOS-only backend"
endif

test-metal-kv-q8: runner $(TEST_KV_TOL)
ifeq ($(shell uname -s),Darwin)
	@set -e; \
	if ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
		model="$${ASAN_MODEL:-models/SmolLM2-135M-Instruct-Q8_0.gguf}"; \
		if [ ! -f "$$model" ]; then echo "metal q8 KV smoke skipped: $$model not found"; exit 0; fi; \
		./$(RUNNER_EXE) -m "$$model" -p "hello" -n 1 --kv q8 --gpu auto -v 2> metal-kv-q8.err >/dev/null; \
		grep -q "q8_0" metal-kv-q8.err; \
		./$(TEST_KV_TOL) "$$model"; \
		echo "metal q8 KV ok"; \
	else \
		echo "metal q8 KV smoke skipped: no Metal device reported by --caps"; \
	fi
else
	@echo "metal q8 KV smoke skipped: macOS-only backend"
endif

test-metal-moe: runner
ifeq ($(shell uname -s),Darwin)
	@set -e; \
	if ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
		$(PYTHON) scripts/make-test-moe.py test-moe-fixture; \
		./$(RUNNER_EXE) -m test-moe-fixture.dense.gguf -p "hello world" -n 12 --temp 0 --gpu off > metal-moe-dense.out 2>/dev/null; \
		env RUNNER_METAL_MM=0 RUNNER_METAL_ATTN_COOP=0 ./$(RUNNER_EXE) -m test-moe-fixture.moe1.gguf -p "hello world" -n 12 --temp 0 --gpu auto > metal-moe1.out 2> metal-moe1.err; \
		cmp -s metal-moe-dense.out metal-moe1.out; \
		grep -q "Metal backend" metal-moe1.err; \
		env RUNNER_METAL_MM=0 RUNNER_METAL_ATTN_COOP=0 ./$(RUNNER_EXE) -m test-moe-fixture.moe2.gguf -p "hello world" -n 12 --temp 0 --gpu auto > metal-moe2.out 2> metal-moe2.err; \
		cmp -s metal-moe-dense.out metal-moe2.out; \
		grep -q "Metal backend" metal-moe2.err; \
		echo "metal MoE ok"; \
	else \
		echo "metal MoE smoke skipped: no Metal device reported by --caps"; \
	fi
else
	@echo "metal MoE smoke skipped: macOS-only backend"
endif

test-metal-gptoss-moe: runner
ifeq ($(shell uname -s),Darwin)
	@set -e; \
	if ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
		$(PYTHON) scripts/make-test-moe.py test-moe-fixture; \
		./$(RUNNER_EXE) -m test-moe-fixture.gptoss-mxfp4.gguf -p "hello world" -n 8 --temp 0 --gpu off > metal-gptoss-moe-cpu.out 2>/dev/null; \
		env RUNNER_METAL_MM=0 RUNNER_METAL_ATTN_COOP=0 ./$(RUNNER_EXE) -m test-moe-fixture.gptoss-mxfp4.gguf -p "hello world" -n 8 --temp 0 --gpu auto > metal-gptoss-moe-gpu.out 2> metal-gptoss-moe-gpu.err; \
		cmp -s metal-gptoss-moe-cpu.out metal-gptoss-moe-gpu.out; \
		grep -q "Metal backend" metal-gptoss-moe-gpu.err; \
		echo "metal gpt-oss MoE ok"; \
	else \
		echo "metal gpt-oss MoE smoke skipped: no Metal device reported by --caps"; \
	fi
else
	@echo "metal gpt-oss MoE smoke skipped: macOS-only backend"
endif

test-metal-gemma4-moe: runner
ifeq ($(shell uname -s),Darwin)
	@set -e; \
	if ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
		$(PYTHON) scripts/make-test-moe.py test-moe-fixture; \
		./$(RUNNER_EXE) -m test-moe-fixture.gemma4-moe.gguf -p "hello world" -n 8 --temp 0 --gpu off > metal-gemma4-moe-cpu.out 2>/dev/null; \
		env RUNNER_METAL_MM=0 RUNNER_METAL_ATTN_COOP=0 ./$(RUNNER_EXE) -m test-moe-fixture.gemma4-moe.gguf -p "hello world" -n 8 --temp 0 --gpu auto > metal-gemma4-moe-gpu.out 2> metal-gemma4-moe-gpu.err; \
		cmp -s metal-gemma4-moe-cpu.out metal-gemma4-moe-gpu.out; \
		grep -q "Metal backend" metal-gemma4-moe-gpu.err; \
		echo "metal gemma4 MoE ok"; \
	else \
		echo "metal gemma4 MoE smoke skipped: no Metal device reported by --caps"; \
	fi
else
	@echo "metal gemma4 MoE smoke skipped: macOS-only backend"
endif

test-metal-gemma4-hetero: runner
ifeq ($(shell uname -s),Darwin)
	@set -e; \
	if ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
		$(PYTHON) scripts/make-test-model.py --gemma4-hetero test-g4h.gguf; \
		./$(RUNNER_EXE) -m test-g4h.gguf -p "hello world" -n 8 --temp 0 --gpu off > metal-g4h-dense-cpu.out 2>/dev/null; \
		env RUNNER_METAL_MM=0 RUNNER_METAL_ATTN_COOP=0 ./$(RUNNER_EXE) -m test-g4h.gguf -p "hello world" -n 8 --temp 0 --gpu auto > metal-g4h-dense-gpu.out 2> metal-g4h-dense-gpu.err; \
		cmp -s metal-g4h-dense-cpu.out metal-g4h-dense-gpu.out; \
		grep -q "Metal backend" metal-g4h-dense-gpu.err; \
		$(PYTHON) scripts/make-test-moe.py test-moe-fixture; \
		./$(RUNNER_EXE) -m test-moe-fixture.gemma4-moe-hetero.gguf -p "hello world" -n 8 --temp 0 --gpu off > metal-g4h-moe-cpu.out 2>/dev/null; \
		env RUNNER_METAL_MM=0 RUNNER_METAL_ATTN_COOP=0 ./$(RUNNER_EXE) -m test-moe-fixture.gemma4-moe-hetero.gguf -p "hello world" -n 8 --temp 0 --gpu auto > metal-g4h-moe-gpu.out 2> metal-g4h-moe-gpu.err; \
		cmp -s metal-g4h-moe-cpu.out metal-g4h-moe-gpu.out; \
		grep -q "Metal backend" metal-g4h-moe-gpu.err; \
		echo "metal gemma4 heterogeneous ok"; \
	else \
		echo "metal gemma4 hetero smoke skipped: no Metal device reported by --caps"; \
	fi
else
	@echo "metal gemma4 hetero smoke skipped: macOS-only backend"
endif

test-metal-gelu-overflow: runner
ifeq ($(shell uname -s),Darwin)
	@set -e; \
	if ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
		$(PYTHON) scripts/make-test-model.py --arch gemma3 --act-overflow test-actovf.gguf; \
		./$(RUNNER_EXE) -m test-actovf.gguf -p "hello world" -n 8 --temp 0 --gpu off > metal-actovf-cpu.out 2>/dev/null; \
		env RUNNER_METAL_MM=0 RUNNER_METAL_ATTN_COOP=0 ./$(RUNNER_EXE) -m test-actovf.gguf -p "hello world" -n 8 --temp 0 --gpu auto > metal-actovf-gpu.out 2> metal-actovf-gpu.err; \
		cmp -s metal-actovf-cpu.out metal-actovf-gpu.out; \
		grep -q "Metal backend" metal-actovf-gpu.err; \
		echo "metal GELU overflow ok"; \
	else \
		echo "metal GELU overflow smoke skipped: no Metal device reported by --caps"; \
	fi
else
	@echo "metal GELU overflow smoke skipped: macOS-only backend"
endif

test-metal-eseries: runner
ifeq ($(shell uname -s),Darwin)
	@set -e; \
	if ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
		for cfg in 0,16 3,0 3,16; do \
		  $(PYTHON) scripts/make-test-model.py --eseries $$cfg test-es.gguf; \
		  ./$(RUNNER_EXE) -m test-es.gguf -p "hello world" -n 8 --temp 0 --gpu off > metal-es-cpu.out 2>/dev/null; \
		  env RUNNER_METAL_MM=0 RUNNER_METAL_ATTN_COOP=0 ./$(RUNNER_EXE) -m test-es.gguf -p "hello world" -n 8 --temp 0 --gpu auto > metal-es-gpu.out 2> metal-es-gpu.err; \
		  cmp -s metal-es-cpu.out metal-es-gpu.out || { echo "eseries $$cfg differs"; exit 1; }; \
		  grep -q "Metal backend" metal-es-gpu.err || { echo "eseries $$cfg: Metal never engaged"; exit 1; }; \
		done; \
		echo "metal E-series ok"; \
	else \
		echo "metal E-series smoke skipped: no Metal device reported by --caps"; \
	fi
else
	@echo "metal E-series smoke skipped: macOS-only backend"
endif

# Multi-buffer weight wrap, forced. Every model this is developed on fits a
# single MTLBuffer, so the split path would otherwise ship untested:
# RUNNER_METAL_MAX_BUF shrinks the per-buffer ceiling until a fixture has to
# span several wraps. What is being gated is that the split changes NOTHING --
# the wrap is at tensor boundaries, so output must stay byte-identical.
test-metal-multibuf: runner $(TEST_GPU_ID) test.gguf
ifeq ($(shell uname -s),Darwin)
	@set -e; \
	if ! ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
	  echo "metal multi-buffer: SKIP (no Metal device)"; exit 0; fi; \
	prompt="The city of Lisbon sits on seven hills above the Tagus estuary"; \
	for pair in test.gguf:262144 \
	            $(if $(wildcard test-moe-fixture.moe1.gguf),test-moe-fixture.moe1.gguf:65536) \
	            $(if $(wildcard models/SmolLM2-135M-Instruct-Q8_0.gguf),models/SmolLM2-135M-Instruct-Q8_0.gguf:33554432); do \
	  m=$${pair%%:*}; cap=$${pair##*:}; \
	  ./$(RUNNER_EXE) -m $$m -p "$$prompt" -n 12 -b 8 --temp 0 --gpu auto \
	    > mb-one.out 2>/dev/null; \
	  RUNNER_METAL_MAX_BUF=$$cap ./$(RUNNER_EXE) -m $$m -p "$$prompt" -n 12 -b 8 \
	    --temp 0 --gpu auto > mb-many.out 2> mb-many.err; \
	  grep -q "Metal backend" mb-many.err || { \
	    echo "FAIL: $$m did not engage Metal under a forced buffer cap"; exit 1; }; \
	  grep -q "weights copied" mb-many.err && { \
	    echo "FAIL: $$m fell back to a copied buffer — the split path was not exercised"; exit 1; }; \
	  n=$$(grep -oE "wrapped in [0-9]+" mb-many.err | grep -oE "[0-9]+" | head -1); \
	  [ -n "$$n" ] && [ "$$n" -ge 2 ] || { \
	    echo "FAIL: $$m did not split (wrapped in $${n:-1}) — the gate would pass vacuously"; exit 1; }; \
	  cmp -s mb-one.out mb-many.out || { \
	    echo "FAIL: $$m output differs across a $$n-buffer split"; exit 1; }; \
	  echo "  metal multi-buffer ok ($$m, $$n buffers, byte-identical)"; \
	done; \
	RUNNER_METAL_MAX_BUF=262144 ./$(TEST_GPU_ID) test.gguf; \
	rm -f mb-one.out mb-many.out mb-many.err
else
	@echo "metal multi-buffer: SKIP (macOS-only backend)"
endif

test-metal-swa: runner
ifeq ($(shell uname -s),Darwin)
	@set -e; \
	if ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
		$(PYTHON) scripts/make-test-model.py --arch qwen3 --swa 8,2 test-swa.gguf; \
		./$(RUNNER_EXE) -m test-swa.gguf -p "abcdefghijklmnopqrstuvwxyz0123456789" -n 12 --temp 0 --gpu off > metal-swa-cpu.out 2>/dev/null; \
		env RUNNER_METAL_MM=0 RUNNER_METAL_ATTN_COOP=0 ./$(RUNNER_EXE) -m test-swa.gguf -p "abcdefghijklmnopqrstuvwxyz0123456789" -n 12 --temp 0 --gpu auto > metal-swa-gpu.out 2> metal-swa-gpu.err; \
		cmp -s metal-swa-cpu.out metal-swa-gpu.out; \
		grep -q "Metal backend" metal-swa-gpu.err; \
		echo "metal SWA ok"; \
	else \
		echo "metal SWA smoke skipped: no Metal device reported by --caps"; \
	fi
else
	@echo "metal SWA smoke skipped: macOS-only backend"
endif

test: $(TEST_JSON_SCHEMA) $(TEST_JSON_OOM) $(TEST_SCHEMA_OOM) $(TEST_SAMPLER) \
      $(TEST_TOKENIZER) $(TEST_TOK_MERGE) $(TEST_TOKENIZER_OOM) $(TEST_TEMPLATE) \
      $(TEST_TEMPLATE_OOM) \
      $(TEST_TOOLS) $(TEST_SHARED) $(TEST_FILE_ID) $(TEST_BATCH) $(TEST_BIND) $(TEST_HOST_HEADER) \
      $(TEST_PREFIX) $(TEST_GRAMMAR_FF) $(TEST_VRAMREG) $(TEST_KV_TOL) $(TEST_TC_TOL) $(TEST_I8_TOL) $(TEST_MV_TOL) $(TEST_ATTN_TOL) $(TEST_GPU_ID) $(TEST_MOE_TOL) $(TEST_MOE_ROUTER) $(TEST_PAGING_WARN) $(TEST_AUTOFIT) $(TEST_RESP_SM_DEP) \
      $(TEST_QUANTS_SIMD) $(TEST_INSTANCES) $(TEST_INSTANCES_OOM) $(TEST_METAL_ADMISSION) $(TEST_TRAY_CORE) \
      $(TEST_QUANTIZE) \
      $(TEST_VRAM_ROLLBACK) $(TEST_GGUF_GETTERS) $(TEST_GGUF_SPLIT) $(TEST_PARSE) \
      $(TEST_THREAD_DEFAULT) \
      $(TEST_MODEL_LOAD_FAILURE) $(TEST_RESTART) $(TEST_PFX_PERSIST) \
      $(TEST_SCHED_TURN) $(TEST_RESIDENCY) $(TEST_BUDGET) $(TEST_ATTRIB_DEP) \
      $(TEST_STOP_CONSTRAINT) $(TEST_MSG_OOM_DEP) runner test.gguf
	./$(TEST_BIND)
	./$(TEST_HOST_HEADER)
	./$(TEST_RESIDENCY)
	./$(TEST_RESTART)
	./$(TEST_PFX_PERSIST)
	./$(TEST_SCHED_TURN)
	./$(TEST_VRAMREG)
	./$(TEST_JSON_SCHEMA)
	./$(TEST_JSON_OOM)
	./$(TEST_SCHEMA_OOM)
	./$(TEST_SAMPLER)
	./$(TEST_TOKENIZER)
	./$(TEST_TOK_MERGE)
	./$(TEST_TOKENIZER_OOM)
	./$(TEST_TEMPLATE)
	./$(TEST_TEMPLATE_OOM)
	./$(TEST_TOOLS)
	./$(TEST_BUDGET)
	$(TEST_ATTRIB_RUN)
	$(TEST_MSG_OOM_RUN)
	./$(TEST_SHARED)
	./$(TEST_FILE_ID)
	./$(TEST_BATCH)
	./$(TEST_PREFIX)
	./$(TEST_GRAMMAR_FF)
	./$(TEST_STOP_CONSTRAINT)
	$(TEST_RESP_SM_RUN)
	./$(TEST_KV_TOL)
	./$(TEST_TC_TOL)
	./$(TEST_I8_TOL)
	./$(TEST_MV_TOL)
	./$(TEST_ATTN_TOL)
	./$(TEST_GPU_ID)
	./$(TEST_QUANTS_SIMD)
	./$(TEST_INSTANCES)
	./$(TEST_INSTANCES_OOM)
	./$(TEST_METAL_ADMISSION)
	./$(TEST_TRAY_CORE)
	@# The split guard was absent from this list entirely, which is how a
	@# target-name collision kept it unbuilt and unnoticed. It self-skips
	@# without CUDA, so it costs a Mac nothing and actually fires on the boxes
	@# that have the backend it guards.
	$(MAKE) --no-print-directory test-split-guard
	$(MAKE) --no-print-directory test-makefile-sane
	@# the fused-vs-eager routing gate needs a fixture whose router is not
	@# zero: the dense-oracle MoE fixtures are 0.5/0.5 either way and can only
	@# compare a routing path with itself (it self-skips on those, correctly)
	$(PYTHON) scripts/make-test-moe.py test-moe-fixture
	./$(TEST_MOE_TOL) test-moe-fixture.moe4.gguf
	./$(TEST_MOE_ROUTER) test-moe-fixture
	./$(TEST_PAGING_WARN) test-moe-fixture
	./$(TEST_AUTOFIT)
	@# Llama-4 attention knobs: NoPE and the position-dependent temperature
	@mkdir -p test-attn
	$(PYTHON) scripts/make-test-model.py test-attn/k_off.gguf
	$(PYTHON) scripts/make-test-model.py --attn-knobs 1,0.0 test-attn/k_nope.gguf
	$(PYTHON) scripts/make-test-model.py --attn-knobs 2,0.0 test-attn/k_half.gguf
	$(PYTHON) scripts/make-test-model.py --attn-knobs 1,0.1 test-attn/k_temp.gguf
	@# floor_scale 4 instead of llama-4's 8192: at 8192 the temperature is
	@# exactly 1.0 for every position a test prompt reaches, so the knob's
	@# arithmetic is never actually exercised. This fixture makes it live.
	$(PYTHON) scripts/make-test-model.py --attn-knobs 1,5.0,4 test-attn/k_temp_live.gguf
	$(PYTHON) -m pytest -q tests/test_attn_knobs.py
	@# CPU/GPU agreement at logit precision on each knob. Token identity (the
	@# text gates) is blind at fixture scale -- see the test's header.
	./$(TEST_GPU_ID) test-attn/k_off.gguf
	./$(TEST_GPU_ID) test-attn/k_nope.gguf
	./$(TEST_GPU_ID) test-attn/k_half.gguf
	./$(TEST_GPU_ID) test-attn/k_temp_live.gguf
	./$(TEST_QUANTIZE)
	./$(TEST_VRAM_ROLLBACK)
	./$(TEST_GGUF_GETTERS)
	@mkdir -p test-gguf-split
	$(PYTHON) scripts/make-test-model.py test-gguf-split/whole.gguf
	$(PYTHON) scripts/gguf-split.py test-gguf-split/whole.gguf test-gguf-split/part 3
	./$(TEST_GGUF_SPLIT) test-gguf-split/whole.gguf test-gguf-split/part-00001-of-00003.gguf
	./$(TEST_PARSE)
	./$(TEST_THREAD_DEFAULT)
	./$(TEST_MODEL_LOAD_FAILURE)
	$(MAKE) --no-print-directory test-bare-invocation
	$(MAKE) --no-print-directory test-shader-embed
	$(MAKE) --no-print-directory test-metal-shader-gate
	$(MAKE) --no-print-directory test-metal-kquant
	$(MAKE) --no-print-directory test-metal-multibuf
	$(PYTHON) scripts/check-generated.py
	@if $(PYTHON) -c "import pytest" >/dev/null 2>&1; then \
		set -e; \
		PYTHONPATH=python/src $(PYTHON) -m pytest python/tests/test_client.py; \
		$(PYTHON) -m pytest -q tests/test_fit_check.py tests/test_apertus.py tests/test_ornith_cpu.py tests/test_ornith_reference.py tests/test_compat_matrix.py tests/test_arch_admission.py tests/test_hostile_geometry.py tests/test_cli_files.py tests/test_chat_template_flag.py tests/test_split_gguf.py tests/test_metal_coverage.py tests/test_caps.py tests/test_bench_json.py tests/test_mtp_admission.py tests/test_compare_llamacpp.py tests/test_release_check.py tests/test_eseries.py tests/test_stress_models.py tests/test_moe_prune_plan.py tests/test_kld_compare.py tests/test_quant_fidelity.py; \
		$(MAKE) --no-print-directory test-moe PYTHON="$(PYTHON)"; \
		$(MAKE) --no-print-directory test-prune-experts PYTHON="$(PYTHON)"; \
	else \
		echo "Python client tests skipped: pytest is not installed; install it with '$(PYTHON) -m pip install pytest'"; \
	fi

# --------------------------------------------------------------------------
# Template conformance: runner's native C chat renderer vs the model's OWN
# jinja chat template, byte for byte.
#
# Runner renders chat templates in C rather than executing jinja. That is
# deliberate, but it makes silent drift the DEFAULT failure mode, and
# tests/test_template.c cannot catch it: its goldens were written from
# runner's own output, so they prove self-consistency and nothing else. This
# target is the missing oracle.
#
# NOT part of `make test`, on purpose. Three reasons, in order of weight:
#
#   1. It needs an oracle that is not in the tree. The HF-backed families are
#      fetched over the network; the rest are read out of multi-GB GGUFs under
#      models/. `make test` is the offline gate, and a gate that goes red
#      because huggingface.co blinked is a gate people learn to ignore --
#      which is exactly how the goldens rotted in the first place.
#   2. The cache cannot rescue it either: the oracles live under .build/, and
#      `make clean` wipes .build. `make clean && make test` would therefore
#      always need the network back.
#   3. It reports THREE outcomes, not two (exit 2 = "not checked"), because an
#      unavailable oracle must never be counted as conformance. `make test` is
#      binary. Folding a three-state gate into it loses the state that matters.
#
# So it gets its own target, and CI runs it as its own job (.github/workflows/
# ci.yml, job `template-conformance`) restricted to the network-backed
# families, where a red is unambiguous.
#
#   make template-conformance           run the gate (fetches missing oracles)
#   make template-conformance-refresh   re-fetch every oracle first
#   make template-conformance-baseline  re-record the known-difference backlog
#
# The backlog (scripts/template-conformance-baseline.json) holds the
# differences that are BUGS. The gate fails on NEW drift; the backlog keeps
# the existing ones counted and visible instead of quietly allowlisted.
# Intentional deviations go in scripts/template-conformance-allowlist.json,
# and only with a source citation the harness re-verifies on every run.
TEMPLATE_CONFORMANCE = scripts/template-conformance.py

template-conformance:
	$(PYTHON) $(TEMPLATE_CONFORMANCE)

template-conformance-refresh:
	$(PYTHON) $(TEMPLATE_CONFORMANCE) --refresh

template-conformance-baseline:
	$(PYTHON) $(TEMPLATE_CONFORMANCE) --write-baseline

# Harmony's oracle is the openai-harmony LIBRARY, not the jinja in the GGUF:
# that jinja is a reimplementation of the spec with gaps it structurally
# cannot close (no content_type field, so it can never emit <|constrain|>).
# This target records the library's renders so a machine without it still
# checks harmony instead of skipping it. Needs `pip install openai-harmony`,
# or RUNNER_HARMONY_PYTHON pointing at an interpreter that has it.
template-conformance-harmony-oracle:
	$(PYTHON) $(TEMPLATE_CONFORMANCE) --write-harmony-oracle

smoke: runner test.gguf
	./$(RUNNER_EXE) --version
	./$(RUNNER_EXE) --caps
	./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; p=json.load(sys.stdin)['sampling_presets']; assert {x['name'] for x in p} >= {'generic','qwen3','llama3','gemma3','phi3'} and all(x['source'] for x in p); print('preset table ok')"
	./$(RUNNER_EXE) -m test.gguf -p "hello" -n 8 --temp 0 --gpu off
	./$(RUNNER_EXE) -m test.gguf -p "hi" -n 24 --temp 0 --json --gpu off 2>/dev/null | $(PYTHON) -c "import json,sys; json.load(sys.stdin); print('valid json')"
	./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; c=json.load(sys.stdin); assert c['kv_types'] == ['f16','q8'], c['kv_types']; assert c['kv_type_default'] == 'f16', 'q8 KV is lossy: f16 must stay the default'; print('kv cache types ok')"
	./$(RUNNER_EXE) -m test.gguf -p "hello" -n 8 --temp 0 --gpu off --kv q8 2>&1 | grep -q "head_dim not a multiple of 32" && echo "kv q8 fallback ok"

release-check: runner
	@set -e; \
	tag="$${TAG:-v$$(./$(RUNNER_EXE) --version | sed 's/^runner //')}"; \
	tmp="$$(mktemp)"; \
	trap 'rm -f "$$tmp"' EXIT; \
	printf '%s\n' "$$(./$(RUNNER_EXE) --version)" > "$$tmp"; \
	printf 'tag:        %s\ncommit:     %s\nbuilt:      local\n' "$$tag" "$$(git rev-parse HEAD 2>/dev/null || echo unknown)" >> "$$tmp"; \
	$(PYTHON) scripts/check-release.py --tag "$$tag" --binary ./$(RUNNER_EXE) \
		--build-info "$$tmp" --commit "$$(git rev-parse HEAD 2>/dev/null || echo unknown)"

# Optional ecosystem gate. Install the pinned Python and Node dependencies in
# tests/compatibility first; Runner itself remains dependency-free.
compat-consumers: runner test.gguf
	$(PYTHON) scripts/consumer_compat.py

# ---------------------------------------------------------------- fuzzing
#
# libFuzzer harnesses for the hand-written parsers that eat untrusted input.
# clang-only: `make fuzz` prints a notice and succeeds when clang is absent
# (the Windows dev box is msys2/gcc), so it never breaks a normal build.
#
# Runs are deliberately short and memory-capped so CI can afford them. Seeds
# are the committed corpora under tests/fuzz/corpus/<target>/; libFuzzer's own
# discoveries and any crash artifacts go to the throwaway fuzz-corpus/ tree
# rather than dirtying the checkout.
FUZZ_CLANG   ?= clang
FUZZ_TIME    ?= 20
FUZZ_RSS_MB  ?= 2048
FUZZ_TARGETS = json_parse schema_compile sval_feed jsonv_feed gguf_open
# TODO: tok_encode (src/tokenizer.c) is deliberately absent. It needs a loaded
# tokenizer rather than a bare buffer, so the harness has to stand up a vocab
# first -- and tokenizer.c has been rewritten substantially since the original
# fuzz plan was drafted, so re-read the current code before trusting a harness.
# The committed tests/fixtures/vocab-*.gguf are the natural fixture when
# someone picks this up.

# -O1 -g: libFuzzer wants speed but ASan reports want frames.
# No -march=native and no -ffast-math: the point here is defined behaviour,
# and UBSan must abort rather than warn or the run cannot gate anything.
FUZZ_FLAGS = -g -O1 -std=gnu11 -Wall -Wextra -Wno-unused-parameter -I src \
             -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=undefined \
             -fno-omit-frame-pointer

FUZZ_SRC_json_parse     = src/json.c
FUZZ_SRC_schema_compile = src/json.c src/schema.c src/jsonmode.c
FUZZ_SRC_sval_feed      = src/json.c src/schema.c src/jsonmode.c
FUZZ_SRC_jsonv_feed     = src/jsonmode.c
FUZZ_SRC_gguf_open      = src/gguf.c src/compat.c src/quants.c

fuzz-%: tests/fuzz/fuzz_%.c $(wildcard src/*.c) $(HDR)
	$(FUZZ_CLANG) $(FUZZ_FLAGS) tests/fuzz/fuzz_$*.c $(FUZZ_SRC_$*) -o $@ -lm

# build only; useful on its own to check the harnesses still compile
fuzz-build: $(addprefix fuzz-,$(FUZZ_TARGETS))

# allocator_may_return_null: a size read straight out of an untrusted file can
# ask for tens of GB. Aborting on that turns every run into the same
# already-known resource finding and hides everything behind it; returning NULL
# instead makes the allocation *fail*, which is the behaviour on any host
# without memory overcommit and which these parsers are supposed to handle. It
# also means the OOM paths get fuzzed rather than skipped.
FUZZ_SAN_OPTS = allocator_may_return_null=1:max_allocation_size_mb=1024

# gguf_open mutes its own stderr per call (see the harness); log_path keeps
# sanitizer reports that are raised inside the muted window.
FUZZ_ENV_gguf_open = ASAN_OPTIONS=$(FUZZ_SAN_OPTS):log_path=fuzz-corpus/gguf_open/asan \
                     UBSAN_OPTIONS=log_path=fuzz-corpus/gguf_open/ubsan
# a valid GGUF header is ~8 KB; without a cap libFuzzer sizes inputs from the
# largest seed and spends the budget copying weights instead of parsing
FUZZ_ARGS_gguf_open = -max_len=16384

# $(foreach) not a shell loop: the per-target FUZZ_ENV_*/FUZZ_ARGS_* lookups
# have to happen while make is expanding, which `for t in ...; $(VAR_$$t)`
# cannot do (make would resolve the name before the shell ever sets $t)
fuzz-run: fuzz-build
	@$(foreach t,$(FUZZ_TARGETS), \
		echo "== fuzzing $(t) for $(FUZZ_TIME)s =="; \
		mkdir -p fuzz-corpus/$(t); \
		env ASAN_OPTIONS=$(FUZZ_SAN_OPTS) $(FUZZ_ENV_$(t)) \
		    ./fuzz-$(t) fuzz-corpus/$(t) tests/fuzz/corpus/$(t) \
			-max_total_time=$(FUZZ_TIME) -rss_limit_mb=$(FUZZ_RSS_MB) \
			-malloc_limit_mb=1024 \
			-timeout=25 -artifact_prefix=fuzz-corpus/$(t)/crash- \
			-print_final_stats=1 $(FUZZ_ARGS_$(t)) \
			|| { cat fuzz-corpus/$(t)/asan.* fuzz-corpus/$(t)/ubsan.* 2>/dev/null; exit 1; }; \
	)
	@echo "fuzz: all targets clean"

fuzz:
	@if command -v $(FUZZ_CLANG) > /dev/null 2>&1; then \
		$(MAKE) --no-print-directory fuzz-run; \
	else \
		echo "make fuzz: skipped -- '$(FUZZ_CLANG)' is not on PATH."; \
		echo "            libFuzzer needs clang; install it or set FUZZ_CLANG=<path>."; \
	fi

clean:
	rm -f test-moe-fixture.*.gguf runner runner-debug $(TEST_JSON_SCHEMA) $(TEST_JSON_OOM) \
		$(TEST_TEMPLATE_OOM) \
	      $(TEST_SCHEMA_OOM) $(TEST_SAMPLER) $(TEST_TOKENIZER) \
	      $(TEST_TOKENIZER_OOM) $(TEST_TEMPLATE) $(TEST_SHARED) \
	      $(TEST_BATCH) $(TEST_BIND) $(TEST_HOST_HEADER) $(TEST_VRAMREG) test-shared-asan-bin \
	      $(TEST_KV_TOL) $(TEST_TC_TOL) $(TEST_I8_TOL) $(TEST_MV_TOL) $(TEST_ATTN_TOL) $(TEST_GPU_ID) $(TEST_MOE_TOL) $(TEST_MOE_ROUTER) $(TEST_PAGING_WARN) $(TEST_AUTOFIT) $(TEST_RESP_SM) $(TEST_PREFIX) $(TEST_GRAMMAR_FF) $(TEST_TOOLS) $(DIFFTOK) \
	      $(TEST_QUANTS_SIMD) $(TEST_INSTANCES) $(TEST_INSTANCES_OOM) $(TEST_METAL_ADMISSION) $(TEST_TRAY_CORE) \
	      $(TEST_QUANTIZE) $(TEST_VRAM_ROLLBACK) $(TEST_GGUF_GETTERS) \
	      $(TEST_PARSE) $(TEST_THREAD_DEFAULT) $(TEST_METAL_OWNERSHIP) $(TEST_METAL_SHADERS) $(TEST_METAL_KQUANTS) $(TEST_MODEL_LOAD_FAILURE) \
	      $(TEST_FILE_ID) test-file-identity.tmp \
	      $(TEST_BUDGET) $(TEST_ATTRIB) $(TEST_MSG_OOM) $(TEST_STOP_CONSTRAINT) \
	      $(TMPL_CONF_RENDER) \
	      $(TEST_SPLIT_GUARD) split-guard.out test-swap-race-bin
	rm -rf test-attn
	rm -rf .build
	rm -f shared-noid.out
	rm -f metal-cpu.out metal-fallback.out metal-fallback.err
	rm -f metal-init-fallback.out metal-init-fallback.err
	rm -f metal-prefill-loop.out metal-prefill-native.out metal-prefill-native.err
	rm -f metal-kv-q8.err metal-moe-dense.out metal-moe1.out metal-moe1.err
	rm -f metal-moe2.out metal-moe2.err
	rm -f metal-gptoss-moe-cpu.out metal-gptoss-moe-gpu.out metal-gptoss-moe-gpu.err
	rm -f metal-gemma4-moe-cpu.out metal-gemma4-moe-gpu.out metal-gemma4-moe-gpu.err
	rm -f metal-swa-cpu.out metal-swa-gpu.out metal-swa-gpu.err test-swa.gguf
	rm -f $(addprefix fuzz-,$(FUZZ_TARGETS))
	rm -rf fuzz-corpus

# regenerate the committed PTX header (dev machines only: needs nvcc + a host
# compiler). Normal builds and CI use the committed src/kernels_ptx.h.
NVCC ?= nvcc
ptx: src/kernels.cu
	$(NVCC) -ptx -arch=compute_75 -O3 -o src/kernels.ptx src/kernels.cu
	python3 scripts/embed-ptx.py || python scripts/embed-ptx.py

# A duplicate target name makes make DISCARD one recipe and say so — but only
# as a warning, scrolling past in the build noise. That is exactly how
# test-split-guard sat unbuilt and unreferenced for as long as it existed: the
# diagnosis was printed on every single build and nobody was reading it.
#
# "overriding commands for target" is never benign. It always means a recipe
# was silently thrown away. Promote it to a failure rather than a warning.
#
# -n on a do-nothing target is enough: these warnings are emitted while make
# PARSES the makefile, before any recipe runs, so this costs a parse and no
# work at all.
makefile-noop:
	@:

FORCE:
	@:

test-makefile-sane:
	@out=$$($(MAKE) -n --no-print-directory makefile-noop 2>&1); \
	case "$$out" in \
	  *"overriding commands"*) \
	    echo "FAIL: duplicate make target — a recipe is being discarded:"; \
	    echo "$$out" | grep -E "overriding commands|ignoring old commands"; \
	    exit 1;; \
	esac; \
	qline=$$($(MAKE) -Bn --no-print-directory runner | grep -- ' -c src/quants.c '); \
	test -n "$$qline" || { echo "FAIL: quants.c is not a separate translation unit"; exit 1; }; \
	echo "$$qline" | grep -q -- ' -fno-fast-math ' || { echo "FAIL: quants.c lacks -fno-fast-math"; exit 1; }; \
	case "$$qline" in *" -ffast-math "*) echo "FAIL: quants.c still has -ffast-math"; exit 1;; esac; \
	if grep -q 'system(' src/tray.c src/tray_*.c src/tray_*.m; then echo "FAIL: tray launches through a shell"; exit 1; fi; \
	echo "makefile ok (no discarded recipes)"


.PHONY: template-conformance template-conformance-refresh template-conformance-baseline template-conformance-harmony-oracle
.PHONY: FORCE makefile-noop test-makefile-sane fixture-scale-note clean debug ptx test test-bare-invocation test-shader-embed test-metal-shader-gate test-apertus test-moe test-prune-experts test-metal-fallback test-metal-prefill test-metal-kquant test-metal-kv-q8 test-metal-moe test-metal-gptoss-moe test-metal-gemma4-moe test-metal-gemma4-hetero test-metal-gelu-overflow test-metal-eseries test-metal-swa smoke release-check fuzz fuzz-build fuzz-run test-shared-asan test-shared-noid test-split-guard test-swap-race
