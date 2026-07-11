CC      ?= cc
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
else ifeq ($(shell uname -s),Darwin)
GPU_SRC  = src/metal.m
LDFLAGS += -framework Metal -framework Foundation
else
GPU_SRC  = src/cuda.c
LDFLAGS += -ldl
endif

SRC = src/gguf.c src/compat.c src/quants.c src/tokenizer.c src/model.c src/sample.c \
      src/template.c src/jsonmode.c src/schema.c src/quantize.c src/engine.c src/json.c src/server.c \
      src/main.c $(GPU_SRC)

runner: $(SRC) src/runner.h
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDFLAGS)

debug: $(SRC) src/runner.h
	$(CC) -O0 -g -fsanitize=address,undefined -std=gnu11 -Wall $(SRC) -o runner-debug $(LDFLAGS)

clean:
	rm -f runner runner-debug

# regenerate the committed PTX header (dev machines only: needs nvcc + a host
# compiler). Normal builds and CI use the committed src/kernels_ptx.h.
NVCC ?= nvcc
ptx: src/kernels.cu
	$(NVCC) -ptx -arch=compute_75 -O3 -o src/kernels.ptx src/kernels.cu
	python3 scripts/embed-ptx.py || python scripts/embed-ptx.py

.PHONY: clean debug ptx
