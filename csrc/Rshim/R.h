// Rshim/R.h — minimal stand-in for R's <R.h>: pulls in the shim Rinternals plus
// the handful of R.h-level helpers SeqArray uses (Rprintf, R_alloc, Calloc/Free).
#ifndef PYSEQARRAY_RSHIM_R_H_
#define PYSEQARRAY_RSHIM_R_H_

#include "Rinternals.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Rprintf(...)   fprintf(stdout, __VA_ARGS__)
#define REprintf(...)  fprintf(stderr, __VA_ARGS__)
#define Rf_print(x)    ((void)0)

// R memory helpers — back with plain malloc/free.  (R_alloc is transient in R;
// here we leak until process exit, acceptable for the bounded uses in SeqArray.)
#define R_alloc(n, size)  malloc((size_t)(n) * (size_t)(size))
#define R_chk_calloc(n, size)  calloc((size_t)(n), (size_t)(size))

// R's Calloc/Free macros (R_ext/RS.h) used by some SeqArray code
#ifndef Calloc
#define Calloc(n, type)  ((type*)calloc((size_t)(n), sizeof(type)))
#endif
#ifndef Free
#define Free(p)  do { free(p); (p) = NULL; } while (0)
#endif
#ifndef Realloc
#define Realloc(p, n, type)  ((type*)realloc((p), (size_t)(n) * sizeof(type)))
#endif

#ifdef __cplusplus
}
#endif

#endif  // PYSEQARRAY_RSHIM_R_H_
