// Rshim/R_ext/Utils.h — minimal subset of R's <R_ext/Utils.h>.
#ifndef PYSEQARRAY_RSHIM_REXT_UTILS_H_
#define PYSEQARRAY_RSHIM_REXT_UTILS_H_

#include "../Rinternals.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// R sort helpers occasionally used; back with C stdlib qsort wrappers.
static inline void R_qsort_int_I(int *v, int *I, int i, int j) { (void)v;(void)I;(void)i;(void)j; }
static inline void R_isort(int *x, int n) { (void)x;(void)n; }
static inline void R_rsort(double *x, int n) { (void)x;(void)n; }
void R_CheckUserInterrupt(void);

#ifdef __cplusplus
}
#endif

#endif
