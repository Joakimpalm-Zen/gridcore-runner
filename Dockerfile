# Runner as a container image: the same single C binary the release ships, on a
# distroless glibc base. This is packaging, not a new dependency — the image
# contains the binary and nothing else. CPU inference; a CUDA image would be a
# separate, heavier variant and is deliberately not this one.
#
# Runner binds the server to LOOPBACK ONLY by design (there is no --host /
# 0.0.0.0 flag; see src/server.c) — so it never exposes itself to a network,
# even in a container. That shapes how you run it:
#
#   One-shot inference (no networking needed):
#     docker run --rm -v "$PWD/models:/models" \
#       ghcr.io/joakimpalm-zen/xyntetik-runner:latest \
#       -m /models/your.gguf -p "hello" -n 128 --gpu off
#
#   Serve on the HOST's localhost (Linux; --network host shares the host
#   loopback, so the loopback-only server is reachable at 127.0.0.1:8080 on the
#   host and nowhere else):
#     docker run --rm --network host -v "$PWD/models:/models" \
#       ghcr.io/joakimpalm-zen/xyntetik-runner:latest \
#       -m /models/your.gguf --serve --port 8080
#
# `-p 8080:8080` does NOT work: the port-proxy cannot reach a server bound to
# the container's own loopback. Use --network host, and keep deployment on
# trusted hosts — there is no auth boundary.

FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
# Match the release's Linux build (x86-64-v3 / AVX2). The CUDA backend compiles
# in but is dlopen'd at runtime, so it needs no CUDA toolkit here and falls back
# to CPU when no GPU is present.
RUN make CFLAGS="-O3 -ffast-math -std=gnu11 -Wall -Wextra -Wno-unused-parameter -march=x86-64-v3"
# No unverified artifact: the binary must build a fixture and run before it ships.
RUN python3 scripts/make-test-model.py /tmp/test.gguf \
    && ./runner -m /tmp/test.gguf -p hi -n 4 --temp 0 --gpu off \
    && ./runner --version

FROM gcr.io/distroless/cc-debian12
LABEL org.opencontainers.image.title="Xyntetik Runner" \
      org.opencontainers.image.description="Single-binary local LLM inference engine (CPU); tool calls survive the token limit." \
      org.opencontainers.image.source="https://github.com/Joakimpalm-Zen/xyntetik-runner"
COPY --from=build /src/runner /usr/local/bin/runner
EXPOSE 8080
ENTRYPOINT ["runner"]
CMD ["--help"]
