// Rshim/Rshim_error.h — the boundary exception type, shared by the shim, the GDS
// bridge, and the R-style COREARRAY_TRY/CATCH override.  C++ only.
#ifndef PYSEQARRAY_RSHIM_ERROR_H_
#define PYSEQARRAY_RSHIM_ERROR_H_

#include <string>
#include <stdexcept>

// Rf_error throws this; cclib.cpp converts it to a Python exception at the
// entry-point boundary.
struct Rsh_error : public std::runtime_error {
    explicit Rsh_error(const std::string &m) : std::runtime_error(m) {}
};

#endif  // PYSEQARRAY_RSHIM_ERROR_H_
