CC      ?= cc
# Probe by RUNNING the interpreter, not by looking it up: on Windows a
# python3 "app execution alias" stub exists on PATH but exits non-zero with
# a Store advert, so `command -v python3` picks an interpreter that cannot run
PYTHON  ?= $(shell python3 -c "" >/dev/null 2>&1 && echo python3 || echo python)
# gnu11 (not c11): strict ISO mode hides M_PI and POSIX symbols on glibc/MinGW
# -march=native unlocks the AVX2/FMA/F16C dot kernels in quants.c on x86;
# other ISAs (ARM macs) compile the scalar fallbacks
CFLAGS  ?= -O3 -ffast-math -std=gnu11 -Wall -Wextra -Wno-unused-parameter -march=native
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
TEST_JSON_OOM = test-json-oom.exe
TEST_SAMPLER = test-sampler.exe
else ifeq ($(shell uname -s),Darwin)
GPU_SRC  = src/metal.m
LDFLAGS += -framework Metal -framework Foundation
RUNNER_EXE = runner
TEST_JSON_SCHEMA = test-json-schema
TEST_TOKENIZER = test-tokenizer
TEST_TEMPLATE = test-template
TEST_JSON_OOM = test-json-oom
TEST_SAMPLER = test-sampler
else
GPU_SRC  = src/cuda.c
LDFLAGS += -ldl
RUNNER_EXE = runner
TEST_JSON_SCHEMA = test-json-schema
TEST_TOKENIZER = test-tokenizer
TEST_TEMPLATE = test-template
TEST_JSON_OOM = test-json-oom
TEST_SAMPLER = test-sampler
endif

SRC = src/gguf.c src/compat.c src/quants.c src/tokenizer.c src/model.c src/sample.c \
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

TEST_TMPL_SRC = tests/test_template.c src/gguf.c src/tokenizer.c src/template.c \
                src/json.c src/compat.c src/quants.c
$(TEST_TEMPLATE): $(TEST_TMPL_SRC) src/runner.h
	$(CC) $(CFLAGS) -I src $(TEST_TMPL_SRC) -o $@ -lm

# sampler presets and the greedy/penalty contract need no model, so the test
# links src/sample.c alone
$(TEST_SAMPLER): tests/test_sampler.c src/sample.c src/runner.h
	$(CC) $(CFLAGS) -I src tests/test_sampler.c src/sample.c -o $@ -lm

# compiles src/json.c directly into the test with instrumented allocators
$(TEST_JSON_OOM): tests/test_json_oom.c src/json.c src/json.h
	$(CC) $(CFLAGS) -I src tests/test_json_oom.c -o $@ -lm

test.gguf: scripts/make-test-model.py
	$(PYTHON) scripts/make-test-model.py test.gguf

test: $(TEST_JSON_SCHEMA) $(TEST_JSON_OOM) $(TEST_TOKENIZER) $(TEST_TEMPLATE) $(TEST_SAMPLER) test.gguf
	./$(TEST_JSON_SCHEMA)
	./$(TEST_JSON_OOM)
	./$(TEST_SAMPLER)
	./$(TEST_TOKENIZER)
	./$(TEST_TEMPLATE)
	@if $(PYTHON) -c "import pytest" >/dev/null 2>&1; then \
		PYTHONPATH=python/src $(PYTHON) -m pytest python/tests/test_client.py; \
	else \
		echo "Python client tests skipped: pytest is not installed; install it with '$(PYTHON) -m pip install pytest'"; \
	fi

smoke: runner test.gguf
	./$(RUNNER_EXE) --version
	./$(RUNNER_EXE) --caps
	./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; p=json.load(sys.stdin)['sampling_presets']; assert {x['name'] for x in p} >= {'generic','qwen3','llama3','gemma3','phi3'} and all(x['source'] for x in p); print('preset table ok')"
	./$(RUNNER_EXE) -m test.gguf -p "hello" -n 8 --temp 0 --gpu off
	./$(RUNNER_EXE) -m test.gguf -p "hi" -n 24 --temp 0 --json --gpu off 2>/dev/null | $(PYTHON) -c "import json,sys; json.load(sys.stdin); print('valid json')"

clean:
	rm -f runner runner-debug $(TEST_JSON_SCHEMA) $(TEST_JSON_OOM) $(TEST_TOKENIZER) $(TEST_TEMPLATE) $(TEST_SAMPLER)

# regenerate the committed PTX header (dev machines only: needs nvcc + a host
# compiler). Normal builds and CI use the committed src/kernels_ptx.h.
NVCC ?= nvcc
ptx: src/kernels.cu
	$(NVCC) -ptx -arch=compute_75 -O3 -o src/kernels.ptx src/kernels.cu
	python3 scripts/embed-ptx.py || python scripts/embed-ptx.py

.PHONY: clean debug ptx test smoke
