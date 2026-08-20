// runner — a compact GGUF inference engine (CPU, CUDA, Metal).
#ifndef RUNNER_H
#define RUNNER_H

#define RUNNER_VERSION "0.1.20-alpha"
#define RUNNER_CUDA_NVCC_ARCH "compute_75"
#define RUNNER_CUDA_PTX_TARGET "sm_75"
#define RUNNER_CUDA_MIN_CC "7.5"
#define RUNNER_CUDA_MIN_GPU "NVIDIA Turing / compute capability 7.5 or newer"

// Fixed size of the -m swap registry (server.c SV.reg). Published in --caps as
// "max_models" so a controller can bound the working set it launches instead of
// overflowing this cap. Keep one source of truth for the array and the report.
#define RUNNER_MAX_MODELS 16

// The umbrella header. Everything below used to live in this file; it is now
// thirteen module headers, and this includes all of them so that any consumer
// which wants the whole engine — the CLI, the server, the GPU backends, the
// tests — keeps one include and sees exactly what it saw before.
//
// A translation unit that belongs to ONE module should include that module's
// header instead. That is the point of the split: model.c can no longer reach
// the schema validator or the HTTP-facing tool envelope by accident, and the
// header that declares a module's interface is now the file you read to learn
// what the module is.
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <math.h>
#include "fp16.h"
#include "quants.h"
#include "gguf.h"
#include "tpool.h"
#include "tokenizer.h"
#include "model.h"
#include "vramreg.h"
#include "gpu.h"
#include "sample.h"
#include "jsonmode.h"
#include "schema.h"
#include "template.h"
#include "engine.h"

#endif // RUNNER_H
