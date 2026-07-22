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
# Cross-compile a local build with RUNNER_ARCH=-march=<target>.
RUNNER_ARCH ?= -march=native
CFLAGS += -O3 -ffast-math -std=gnu11 $(RUNNER_ARCH)
LDFLAGS  = -lm -lpthread
ifeq ($(OS),Windows_NT)
# -static: link winpthread/libgcc into the exe so it runs outside an MSYS2
# shell (otherwise it dies at load with STATUS_DLL_NOT_FOUND on libwinpthread-1.dll)
LDFLAGS += -lws2_32 -static
GPU_SRC  = src/cuda.c
RUNNER_EXE = runner.exe
TEST_JSON_SCHEMA = test-json-schema.exe
TEST_TOKENIZER = test-tokenizer.exe
TEST_TEMPLATE = test-template.exe
TEST_TOOLS = test-tools.exe
TEST_JSON_OOM = test-json-oom.exe
TEST_TOKENIZER_OOM = test-tokenizer-oom.exe
TEST_SCHEMA_OOM = test-schema-oom.exe
TEST_SAMPLER = test-sampler.exe
TEST_SHARED = test-shared-weights.exe
TEST_BATCH = test-batch.exe
DIFFTOK = difftok.exe
TEST_BIND = test-bind.exe
else ifeq ($(shell uname -s),Darwin)
GPU_SRC  = src/metal.m
LDFLAGS += -framework Metal -framework Foundation
RUNNER_EXE = runner
TEST_JSON_SCHEMA = test-json-schema
TEST_TOKENIZER = test-tokenizer
TEST_TEMPLATE = test-template
TEST_TOOLS = test-tools
TEST_JSON_OOM = test-json-oom
TEST_TOKENIZER_OOM = test-tokenizer-oom
TEST_SCHEMA_OOM = test-schema-oom
TEST_SAMPLER = test-sampler
TEST_SHARED = test-shared-weights
TEST_BATCH = test-batch
DIFFTOK = difftok
TEST_BIND = test-bind
else
GPU_SRC  = src/cuda.c
LDFLAGS += -ldl
RUNNER_EXE = runner
TEST_JSON_SCHEMA = test-json-schema
TEST_TOKENIZER = test-tokenizer
TEST_TEMPLATE = test-template
TEST_TOOLS = test-tools
TEST_JSON_OOM = test-json-oom
TEST_TOKENIZER_OOM = test-tokenizer-oom
TEST_SCHEMA_OOM = test-schema-oom
TEST_SAMPLER = test-sampler
TEST_SHARED = test-shared-weights
TEST_BATCH = test-batch
DIFFTOK = difftok
TEST_BIND = test-bind
endif

# same .exe suffix rule as every other test binary, without repeating the
# three-way platform branch above
TEST_PREFIX = $(TEST_BATCH:test-batch=test-prefix)
TEST_VRAMREG = $(TEST_BATCH:test-batch=test-vram-registry)
TEST_KV_TOL = $(TEST_BATCH:test-batch%=test-kv-tol%)

SRC = src/gguf.c src/compat.c src/quants.c src/tokenizer.c src/model.c src/sample.c \
      src/vramreg.c \
      src/template.c src/jsonmode.c src/schema.c src/quantize.c src/engine.c src/json.c src/server.c \
      src/main.c $(GPU_SRC)

runner: $(SRC) src/runner.h
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDFLAGS)

debug: $(SRC) src/runner.h
	$(CC) -O0 -g -fsanitize=address,undefined -std=gnu11 -Wall $(SRC) -o runner-debug $(LDFLAGS)

$(TEST_JSON_SCHEMA): tests/test_json_schema.c src/json.c src/jsonmode.c src/schema.c
	$(CC) -std=gnu11 -Wall -Wextra -I src tests/test_json_schema.c src/json.c src/jsonmode.c src/schema.c -o $@ -lm

# quants.c is needed for the ggml_type_* helpers gguf.c links against; CFLAGS
# (not the plainer flags above) so the AVX2 paths match a real build
TEST_TOK_SRC = tests/test_tokenizer.c src/gguf.c src/tokenizer.c src/compat.c src/quants.c
$(TEST_TOKENIZER): $(TEST_TOK_SRC) src/runner.h
	$(CC) $(CFLAGS) -I src $(TEST_TOK_SRC) -o $@ -lm

# difftok: tokenizer differential harness. Not part of `make test` -- it needs a
# real multi-GB model GGUF, which models/ is gitignored for. scripts/difftok.py
# builds it on demand and compares against the HuggingFace reference.
DIFFTOK_SRC = tests/difftok.c src/gguf.c src/tokenizer.c src/compat.c src/quants.c
$(DIFFTOK): $(DIFFTOK_SRC) src/runner.h
	$(CC) $(CFLAGS) -I src $(DIFFTOK_SRC) -o $@ -lm

TEST_TMPL_SRC = tests/test_template.c src/gguf.c src/tokenizer.c src/template.c \
                src/json.c src/compat.c src/quants.c
$(TEST_TEMPLATE): $(TEST_TMPL_SRC) src/runner.h
	$(CC) $(CFLAGS) -I src $(TEST_TMPL_SRC) -o $@ -lm

# the strict tool envelope is only meaningful if the schema engine enforces
# it, so schema.c/jsonmode.c compile in and the tests drive the real validator
TEST_TOOLS_SRC = tests/test_tools.c src/gguf.c src/tokenizer.c src/template.c \
                 src/schema.c src/jsonmode.c src/json.c src/compat.c src/quants.c
$(TEST_TOOLS): $(TEST_TOOLS_SRC) src/runner.h
	$(CC) $(CFLAGS) -I src $(TEST_TOOLS_SRC) -o $@ -lm

# sampler presets and the greedy/penalty contract need no model, so the test
# links src/sample.c alone
$(TEST_SAMPLER): tests/test_sampler.c src/sample.c src/runner.h
	$(CC) $(CFLAGS) -I src tests/test_sampler.c src/sample.c -o $@ -lm

# compiles src/json.c directly into the test with instrumented allocators
$(TEST_JSON_OOM): tests/test_json_oom.c src/json.c src/json.h
	$(CC) $(CFLAGS) -I src tests/test_json_oom.c -o $@ -lm

# compiles src/tokenizer.c into the test with instrumented allocators; gguf.c
# and friends link normally so their allocations stay outside the failure window
TEST_TOK_OOM_SRC = tests/test_tokenizer_oom.c src/gguf.c src/compat.c src/quants.c
$(TEST_TOKENIZER_OOM): $(TEST_TOK_OOM_SRC) src/tokenizer.c src/runner.h
	$(CC) $(CFLAGS) -I src $(TEST_TOK_OOM_SRC) -o $@ -lm

# schema.c and json.c both compile into the test: enum/const literals are
# serialised through jv_dump, so builder failures are schema failure paths
$(TEST_SCHEMA_OOM): tests/test_schema_oom.c src/schema.c src/json.c src/jsonmode.c src/runner.h
	$(CC) $(CFLAGS) -I src tests/test_schema_oom.c src/jsonmode.c -o $@ -lm

# shared model weights: needs the real model + backend, so it links the same
# sources the runner does minus the CLI/server front end
TEST_SHARED_SRC = tests/test_shared_weights.c src/gguf.c src/compat.c \
                  src/quants.c src/model.c src/vramreg.c $(GPU_SRC)
$(TEST_SHARED): $(TEST_SHARED_SRC) src/runner.h
	$(CC) $(CFLAGS) -I src $(TEST_SHARED_SRC) -o $@ $(LDFLAGS)

# same test under ASan/UBSan: the free-exactly-once half of it only fails
# loudly here. Kept out of `make test` because a sanitized model load is slow.
test-shared-asan: $(TEST_SHARED_SRC) src/runner.h test.gguf
	$(CC) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
	    -std=gnu11 -Wall -I src $(TEST_SHARED_SRC) -o test-shared-asan-bin $(LDFLAGS)
	./test-shared-asan-bin $(ASAN_MODEL)

# batched decode: same sources as the shared-weights test (real model +
# backend), because the property under test is a backend property
TEST_BATCH_SRC = tests/test_batch.c src/gguf.c src/compat.c \
                 src/quants.c src/model.c src/vramreg.c $(GPU_SRC)
$(TEST_BATCH): $(TEST_BATCH_SRC) src/runner.h
	$(CC) $(CFLAGS) -I src $(TEST_BATCH_SRC) -o $@ $(LDFLAGS)

# the loopback-only bind. Links nothing: it reads src/server.c and src/main.c
# and interrogates the built ./runner, so a bind address introduced anywhere --
# constant, flag, env var -- trips it. Depends on `runner` because the CLI half
# of the check needs the shipped binary rather than a comment about it.
$(TEST_BIND): tests/test_bind.c src/server.c src/main.c
	$(CC) $(CFLAGS) -I src tests/test_bind.c -o $@

# forkable KV prefixes: needs the real model, the real tokenizer and the real
# engine, because the property under test is that a forked cache produces the
# same logits the model would have produced by prefilling
TEST_PREFIX_SRC = tests/test_prefix.c src/gguf.c src/compat.c src/quants.c \
                  src/tokenizer.c src/model.c src/sample.c src/jsonmode.c \
                  src/schema.c src/json.c src/engine.c src/vramreg.c $(GPU_SRC)
$(TEST_PREFIX): $(TEST_PREFIX_SRC) src/runner.h
	$(CC) $(CFLAGS) -I src $(TEST_PREFIX_SRC) -o $@ $(LDFLAGS)

# the cross-process VRAM registry. Links only vramreg.c and compat.c: the
# free-VRAM figure arrives through a callback, so the whole module is drivable
# with synthetic numbers and the test needs no GPU, no model and no driver --
# which is what lets it run in CI.
$(TEST_VRAMREG): tests/test_vram_registry.c src/vramreg.c src/compat.c src/runner.h
	$(CC) $(CFLAGS) -I src tests/test_vram_registry.c src/vramreg.c src/compat.c -o $@ $(LDFLAGS)

# q8 KV tolerance gate: needs the tokenizer too, because it teacher-forces a
# fixed piece of real text rather than synthetic token ids
TEST_KV_TOL_SRC = tests/test_kv_tol.c src/gguf.c src/compat.c src/quants.c \
                  src/tokenizer.c src/model.c src/vramreg.c $(GPU_SRC)
$(TEST_KV_TOL): $(TEST_KV_TOL_SRC) src/runner.h
	$(CC) $(CFLAGS) -I src $(TEST_KV_TOL_SRC) -o $@ $(LDFLAGS)

test.gguf: scripts/make-test-model.py
	$(PYTHON) scripts/make-test-model.py test.gguf

# Ornith/Qwen3.5 CPU tracer: a committed generator builds a tiny hybrid model
# with three recurrent DeltaNet blocks and one full-attention block.
test-ornith-cpu: runner
	$(PYTHON) -m pytest -q tests/test_ornith_cpu.py

test: $(TEST_JSON_SCHEMA) $(TEST_JSON_OOM) $(TEST_SCHEMA_OOM) $(TEST_SAMPLER) \
      $(TEST_TOKENIZER) $(TEST_TOKENIZER_OOM) $(TEST_TEMPLATE) \
      $(TEST_TOOLS) $(TEST_SHARED) $(TEST_BATCH) $(TEST_BIND) \
      $(TEST_PREFIX) $(TEST_VRAMREG) $(TEST_KV_TOL) runner test.gguf
	./$(TEST_BIND)
	./$(TEST_VRAMREG)
	./$(TEST_JSON_SCHEMA)
	./$(TEST_JSON_OOM)
	./$(TEST_SCHEMA_OOM)
	./$(TEST_SAMPLER)
	./$(TEST_TOKENIZER)
	./$(TEST_TOKENIZER_OOM)
	./$(TEST_TEMPLATE)
	./$(TEST_TOOLS)
	./$(TEST_SHARED)
	./$(TEST_BATCH)
	./$(TEST_PREFIX)
	./$(TEST_KV_TOL)
	@if $(PYTHON) -c "import pytest" >/dev/null 2>&1; then \
		PYTHONPATH=python/src $(PYTHON) -m pytest python/tests/test_client.py; \
		$(PYTHON) -m pytest -q tests/test_ornith_cpu.py; \
	else \
		echo "Python client tests skipped: pytest is not installed; install it with '$(PYTHON) -m pip install pytest'"; \
	fi

smoke: runner test.gguf
	./$(RUNNER_EXE) --version
	./$(RUNNER_EXE) --caps
	./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; p=json.load(sys.stdin)['sampling_presets']; assert {x['name'] for x in p} >= {'generic','qwen3','llama3','gemma3','phi3'} and all(x['source'] for x in p); print('preset table ok')"
	./$(RUNNER_EXE) -m test.gguf -p "hello" -n 8 --temp 0 --gpu off
	./$(RUNNER_EXE) -m test.gguf -p "hi" -n 24 --temp 0 --json --gpu off 2>/dev/null | $(PYTHON) -c "import json,sys; json.load(sys.stdin); print('valid json')"
	./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; c=json.load(sys.stdin); assert c['kv_types'] == ['f16','q8'], c['kv_types']; assert c['kv_type_default'] == 'f16', 'q8 KV is lossy: f16 must stay the default'; print('kv cache types ok')"
	./$(RUNNER_EXE) -m test.gguf -p "hello" -n 8 --temp 0 --gpu off --kv q8 2>&1 | grep -q "head_dim not a multiple of 32" && echo "kv q8 fallback ok"

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
# first -- and tokenizer.c was rewritten substantially after the design in
# HANDOVER.md sec.2 was written, so it needs re-reading before a harness is
# worth trusting. The committed tests/fixtures/vocab-*.gguf are the natural
# fixture when someone picks this up.

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

fuzz-%: tests/fuzz/fuzz_%.c $(wildcard src/*.c) src/runner.h
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
	rm -f runner runner-debug $(TEST_JSON_SCHEMA) $(TEST_JSON_OOM) \
	      $(TEST_SCHEMA_OOM) $(TEST_SAMPLER) $(TEST_TOKENIZER) \
	      $(TEST_TOKENIZER_OOM) $(TEST_TEMPLATE) $(TEST_SHARED) \
	      $(TEST_BATCH) $(TEST_BIND) $(TEST_VRAMREG) test-shared-asan-bin
	rm -f $(addprefix fuzz-,$(FUZZ_TARGETS))
	rm -rf fuzz-corpus

# regenerate the committed PTX header (dev machines only: needs nvcc + a host
# compiler). Normal builds and CI use the committed src/kernels_ptx.h.
NVCC ?= nvcc
ptx: src/kernels.cu
	$(NVCC) -ptx -arch=compute_75 -O3 -o src/kernels.ptx src/kernels.cu
	python3 scripts/embed-ptx.py || python scripts/embed-ptx.py

.PHONY: clean debug ptx test smoke fuzz fuzz-build fuzz-run test-shared-asan
