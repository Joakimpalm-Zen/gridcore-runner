// Fixed worker pool: run one index range function over N items.
#ifndef RUNNER_TPOOL_H
#define RUNNER_TPOOL_H


typedef void (*tp_fn)(void *ctx, int i0, int i1); // process items [i0, i1)
typedef struct tpool tpool;
tpool *tpool_create(int n_threads);
void   tpool_run(tpool *tp, tp_fn fn, void *ctx, int n_items);
void   tpool_destroy(tpool *tp);
int    tpool_size(const tpool *tp);  // workers incl. the calling thread

#endif // RUNNER_TPOOL_H
