CC      ?= cc
CFLAGS  ?= -O3 -ffast-math -std=c11 -Wall -Wextra -Wno-unused-parameter
LDFLAGS  = -lm -lpthread

SRC = src/gguf.c src/quants.c src/tokenizer.c src/model.c src/main.c

runner: $(SRC) src/runner.h
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDFLAGS)

debug: $(SRC) src/runner.h
	$(CC) -O0 -g -fsanitize=address,undefined -std=c11 -Wall $(SRC) -o runner-debug $(LDFLAGS)

clean:
	rm -f runner runner-debug

.PHONY: clean debug
