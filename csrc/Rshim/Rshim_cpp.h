// Rshim/Rshim_cpp.h — C++-only internals of the R shim: the variant cell layout
// and the boundary error type.  Included by Rshim.cpp and cclib.cpp, never by
// the SeqArray engine TUs (they only see the C API in Rinternals.h).
#ifndef PYSEQARRAY_RSHIM_CPP_H_
#define PYSEQARRAY_RSHIM_CPP_H_

#include <string>
#include <vector>
#include <map>
#include <stdexcept>

#include "Rinternals.h"
#include "Rshim_error.h"   // Rsh_error

// The variant cell a SEXP points to.  Only the field(s) matching `type` are used.
struct PYSEXPREC {
    int       type;     // SEXPTYPE
    R_xlen_t  length;   // logical length (mirrors the active container's size)
    std::vector<int>         i;     // INTSXP / LGLSXP
    std::vector<double>      d;     // REALSXP
    std::vector<Rbyte>       raw;   // RAWSXP
    std::vector<SEXP>        list;  // STRSXP (CHARSXP elems) / VECSXP
    std::vector<std::string> str;   // CHARSXP payload [0] / SYMSXP name [0]
    std::map<std::string, SEXP> attrib;  // attributes keyed by symbol name
    bool  na;           // CHARSXP NA marker
    void *ptr;          // EXTPTRSXP payload

    PYSEXPREC() : type(0), length(0), na(false), ptr(0) {}
};

// init the shim global singletons (R_NilValue, symbols, NA_REAL bit pattern)
extern "C" void Rsh_init_globals(void);

#endif  // PYSEQARRAY_RSHIM_CPP_H_
