// Rshim/R_ext/Rdynload.h — stub for R's dynamic-registration header.  The shim
// module init (cclib.cpp) does not use R_registerRoutines; these are only here
// so SeqArray.cpp's R_init_SeqArray text compiles if ever included.
#ifndef PYSEQARRAY_RSHIM_REXT_RDYNLOAD_H_
#define PYSEQARRAY_RSHIM_REXT_RDYNLOAD_H_

#include "../Rinternals.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void *DllInfo;
typedef SEXP (*DL_FUNC)();
typedef struct { const char *name; DL_FUNC fun; int numArgs; } R_CallMethodDef;

static inline void R_registerRoutines(DllInfo, void*, R_CallMethodDef*, void*, void*) {}
static inline void R_useDynamicSymbols(DllInfo, Rboolean) {}
static inline void R_RegisterCCallable(const char*, const char*, DL_FUNC) {}
static inline DL_FUNC R_FindSymbol(const char*, const char*, void*) { return 0; }

#ifdef __cplusplus
}
#endif

#endif
