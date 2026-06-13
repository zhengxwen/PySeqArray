// Rshim/R_GDS_CPP.h — forwards SeqArray's <R_GDS_CPP.h> to pygds's <PyGDS_CPP.h>
// (the C++ GDS class interface), then overrides the COREARRAY_TRY/CATCH macros
// with gdsfmt's R-style semantics (declare `SEXP rv_ans`, `return rv_ans`),
// because pygds's versions are Python-oriented (no rv_ans, return NULL).
#ifndef PYSEQARRAY_RSHIM_R_GDS_CPP_H_
#define PYSEQARRAY_RSHIM_R_GDS_CPP_H_

#include "R_GDS.h"
#include <PyGDS_CPP.h>
#include "Rshim_error.h"   // Rsh_error

// --- R-style entry-point try/catch (overrides pygds's PyObject* versions) ----
#ifdef COREARRAY_TRY
#  undef COREARRAY_TRY
#endif
#ifdef COREARRAY_CATCH
#  undef COREARRAY_CATCH
#endif
#ifdef COREARRAY_CATCH_NONE
#  undef COREARRAY_CATCH_NONE
#endif

// Declares the SEXP return accumulator SeqArray's bodies assign to.
#define COREARRAY_TRY \
    SEXP rv_ans = R_NilValue; \
    int has_error = 0; (void)has_error; \
    try {

// On any C++/CoreArray exception, rethrow as Rsh_error so cclib.cpp's boundary
// converts it to a Python exception; otherwise return the accumulated SEXP.
#define COREARRAY_CATCH \
    } \
    catch (Rsh_error&) { throw; } \
    catch (std::exception &E) { throw Rsh_error(E.what()); } \
    catch (const char *E) { throw Rsh_error(E); } \
    catch (...) { throw Rsh_error("unknown C++ exception"); } \
    return rv_ans;

#define COREARRAY_CATCH_NONE \
    } \
    catch (Rsh_error&) { throw; } \
    catch (std::exception &E) { throw Rsh_error(E.what()); } \
    catch (const char *E) { throw Rsh_error(E); } \
    catch (...) { throw Rsh_error("unknown C++ exception"); }

#endif  // PYSEQARRAY_RSHIM_R_GDS_CPP_H_
