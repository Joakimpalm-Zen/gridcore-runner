CC      ?= cc
# gnu11 (not c11): strict ISO mode hides M_PI and POSIX symbols on glibc/MinGW
CFLAGS  ?= -O3 -ffast-math -std=gnu11 -Wall -Wextra -Wno-unused-parameter
LDFLAGS  = -lm -lpthread
ifeq ($(OS),Windows_NT)
LDFLAGS += -lws2_32
GPU_SRC  = src/gpu_none.c
else ifeq ($(shell uname -s),Darwin)
GPU_SRC  = src/metal.m
LDFLAGS += -framework Metal -framework Foundation
else
GPU_SRC  = src/gpu_none.c
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

.PHONY: clean debug
