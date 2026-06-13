// Rshim/Rdefines.h — R's legacy Rdefines.h macro layer over the shim Rinternals.
#ifndef PYSEQARRAY_RSHIM_RDEFINES_H_
#define PYSEQARRAY_RSHIM_RDEFINES_H_

#include "Rinternals.h"

// ---- NEW_* allocation aliases ----------------------------------------------
#define NEW_INTEGER(n)    Rf_allocVector(INTSXP,  (n))
#define NEW_NUMERIC(n)    Rf_allocVector(REALSXP, (n))
#define NEW_LOGICAL(n)    Rf_allocVector(LGLSXP,  (n))
#define NEW_CHARACTER(n)  Rf_allocVector(STRSXP,  (n))
#define NEW_RAW(n)        Rf_allocVector(RAWSXP,  (n))
#define NEW_LIST(n)       Rf_allocVector(VECSXP,  (n))
#define NEW_STRING(n)     Rf_allocVector(STRSXP,  (n))

// ---- dim / dimnames ---------------------------------------------------------
#define GET_DIM(x)            Rf_getAttrib((x), R_DimSymbol)
#define SET_DIM(x, d)         Rf_setAttrib((x), R_DimSymbol, (d))
#define GET_DIMNAMES(x)       Rf_getAttrib((x), R_DimNamesSymbol)
#define SET_DIMNAMES(x, d)    Rf_setAttrib((x), R_DimNamesSymbol, (d))
#define GET_NAMES(x)          Rf_getAttrib((x), R_NamesSymbol)
#define SET_NAMES(x, n)       Rf_setAttrib((x), R_NamesSymbol, (n))
#define GET_CLASS(x)          Rf_getAttrib((x), R_ClassSymbol)
#define SET_CLASS(x, c)       Rf_setAttrib((x), R_ClassSymbol, (c))
#define GET_LEVELS(x)         Rf_getAttrib((x), R_LevelsSymbol)
#define SET_LEVELS(x, l)      Rf_setAttrib((x), R_LevelsSymbol, (l))

// ---- element get/set legacy spellings --------------------------------------
#define INTEGER_POINTER(x)    INTEGER(x)
#define NUMERIC_POINTER(x)    REAL(x)
#define INTEGER_ELT(x, i)     (INTEGER(x)[(i)])
#define REAL_ELT(x, i)        (REAL(x)[(i)])
#define LOGICAL_ELT(x, i)     (LOGICAL(x)[(i)])
#define RAW_ELT(x, i)         (RAW(x)[(i)])
#define SET_INTEGER_ELT(x, i, v)  (INTEGER(x)[(i)] = (v))
#define SET_REAL_ELT(x, i, v)     (REAL(x)[(i)] = (v))
#define SET_LOGICAL_ELT(x, i, v)  (LOGICAL(x)[(i)] = (v))
#define CHARACTER_POINTER(x)  (x)
#define GET_LENGTH(x)         Rf_length(x)
#define SET_ELEMENT(x, i, v)  SET_VECTOR_ELT((x), (i), (v))
#define VECTOR_ELT_(x, i)     VECTOR_ELT((x), (i))

// ---- AS_* coercion aliases --------------------------------------------------
#define AS_INTEGER(x)     Rf_coerceVector((x), INTSXP)
#define AS_NUMERIC(x)     Rf_coerceVector((x), REALSXP)
#define AS_LOGICAL(x)     Rf_coerceVector((x), LGLSXP)
#define AS_CHARACTER(x)   Rf_coerceVector((x), STRSXP)
#define AS_LIST(x)        Rf_coerceVector((x), VECSXP)
#define IS_INTEGER(x)     Rf_isInteger(x)
#define IS_NUMERIC(x)     Rf_isReal(x)
#define IS_CHARACTER(x)   Rf_isString(x)
#define IS_LOGICAL(x)     Rf_isLogical(x)
#define IS_RAW(x)         (TYPEOF(x) == RAWSXP)
#define IS_LIST(x)        Rf_isVectorList(x)
#define IS_VECTOR(x)      Rf_isVector(x)

// ---- pairlist / language constructors (only reached by the R-eval apply path,
//      which the shim does not execute; no-op stubs so the text compiles) -----
#define SETCADR(x, y)         (y)
#define SETCADDR(x, y)        (y)
#define SETCADDDR(x, y)       (y)
#define SETCAD4R(x, y)        (y)
#define LCONS(a, b)           (b)
#define CONS(a, b)            (b)
#define list1(a)              (a)
#define list2(a, b)           (b)

#endif  // PYSEQARRAY_RSHIM_RDEFINES_H_
