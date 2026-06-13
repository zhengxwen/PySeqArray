// Rshim/Rinternals.h — bounded subset of R's C API used to compile the SeqArray
// C++ engine outside of R, backed by an arena of variant cells.  See DESIGN.md.
//
// Memory model: within one entry-point call nothing is freed (SeqArray relies on
// PROTECT to keep SEXPs alive, never frees mid-.Call), so PROTECT/UNPROTECT are
// no-ops and the whole arena is released at the call boundary (Rsh_arena_reset).
#ifndef PYSEQARRAY_RSHIM_RINTERNALS_H_
#define PYSEQARRAY_RSHIM_RINTERNALS_H_

#include <stddef.h>
#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef ptrdiff_t R_xlen_t;
typedef int       R_len_t;
typedef unsigned char Rbyte;
typedef enum { FALSE = 0, TRUE = 1 } Rboolean;

typedef struct PYSEXPREC *SEXP;

typedef enum {
    NILSXP    = 0,
    SYMSXP    = 1,
    LISTSXP   = 2,
    CLOSXP    = 3,
    ENVSXP    = 4,
    LGLSXP    = 10,
    INTSXP    = 13,
    REALSXP   = 14,
    CPLXSXP   = 15,
    STRSXP    = 16,
    DOTSXP    = 17,
    VECSXP    = 19,
    EXPRSXP   = 20,
    RAWSXP    = 24,
    CHARSXP   = 9,
    EXTPTRSXP = 22,
} SEXPTYPE;

// ---- NA / special values ----------------------------------------------------
extern double R_NaReal;
extern double R_NaN;
extern double R_PosInf;
extern double R_NegInf;
extern int    R_NaInt;
#define NA_INTEGER   R_NaInt
#define NA_LOGICAL   R_NaInt
#define NA_REAL      R_NaReal
extern SEXP R_NaString;   // CHARSXP NA
#define NA_STRING    R_NaString

#define R_FINITE(x)  (isfinite(x) != 0)
#define R_finite(x)  (isfinite(x) != 0)
#define ISNAN(x)     (isnan(x) != 0)
#define ISNA(x)      (Rsh_is_na_real(x) != 0)
int Rsh_is_na_real(double x);

// ---- well-known global SEXPs ------------------------------------------------
extern SEXP R_NilValue;
extern SEXP R_GlobalEnv;
extern SEXP R_EmptyEnv;
extern SEXP R_DotsSymbol;
extern SEXP R_NamesSymbol;
extern SEXP R_ClassSymbol;
extern SEXP R_LevelsSymbol;
extern SEXP R_DimSymbol;
extern SEXP R_DimNamesSymbol;
extern SEXP R_RowNamesSymbol;

// ---- allocation -------------------------------------------------------------
SEXP Rf_allocVector(SEXPTYPE type, R_xlen_t n);
SEXP Rf_allocMatrix(SEXPTYPE type, int nrow, int ncol);
SEXP Rf_ScalarInteger(int x);
SEXP Rf_ScalarReal(double x);
SEXP Rf_ScalarLogical(int x);
SEXP Rf_ScalarString(SEXP x);
SEXP Rf_mkChar(const char *s);
SEXP Rf_mkCharLen(const char *s, int len);
SEXP Rf_mkString(const char *s);
SEXP Rf_duplicate(SEXP x);
SEXP Rf_coerceVector(SEXP x, SEXPTYPE type);
SEXP Rf_asCharacterFactor(SEXP x);

// ---- length / type ----------------------------------------------------------
R_xlen_t Rf_xlength(SEXP x);
R_len_t  Rf_length(SEXP x);
SEXPTYPE TYPEOF_(SEXP x);
#define TYPEOF(x)  TYPEOF_(x)
#define XLENGTH(x) Rf_xlength(x)
#define LENGTH(x)  Rf_length(x)

// ---- element access (atomic) ------------------------------------------------
int    *INTEGER(SEXP x);
double *REAL(SEXP x);
int    *LOGICAL(SEXP x);
Rbyte  *RAW(SEXP x);

// ---- string / list ----------------------------------------------------------
const char *R_CHAR(SEXP charsxp);
#define CHAR(x) R_CHAR(x)
SEXP STRING_ELT(SEXP x, R_xlen_t i);
SEXP VECTOR_ELT(SEXP x, R_xlen_t i);
void SET_STRING_ELT(SEXP x, R_xlen_t i, SEXP v);
SEXP SET_VECTOR_ELT(SEXP x, R_xlen_t i, SEXP v);
const char *Rf_translateChar(SEXP x);
const char *Rf_translateCharUTF8(SEXP x);

// ---- coercion scalars -------------------------------------------------------
int    Rf_asInteger(SEXP x);
double Rf_asReal(SEXP x);
int    Rf_asLogical(SEXP x);
SEXP   Rf_asChar(SEXP x);

// ---- predicates -------------------------------------------------------------
Rboolean Rf_isNull(SEXP x);
Rboolean Rf_isString(SEXP x);
Rboolean Rf_isReal(SEXP x);
Rboolean Rf_isInteger(SEXP x);
Rboolean Rf_isLogical(SEXP x);
Rboolean Rf_isNumeric(SEXP x);
Rboolean Rf_isVector(SEXP x);
Rboolean Rf_isVectorList(SEXP x);
Rboolean Rf_isList(SEXP x);
Rboolean Rf_isFactor(SEXP x);
Rboolean Rf_isEnvironment(SEXP x);
Rboolean Rf_isObject(SEXP x);
Rboolean Rf_inherits(SEXP x, const char *name);

// ---- attributes / symbols ---------------------------------------------------
SEXP Rf_getAttrib(SEXP x, SEXP sym);
SEXP Rf_setAttrib(SEXP x, SEXP sym, SEXP val);
SEXP Rf_install(const char *name);
SEXP Rf_GetRowNames(SEXP x);

// ---- environments / eval (stubbed: PySeqArray drives apply from Python) -----
SEXP Rf_eval(SEXP expr, SEXP env);
SEXP Rf_lang1(SEXP);
SEXP Rf_lang2(SEXP, SEXP);
SEXP Rf_lang3(SEXP, SEXP, SEXP);
void Rf_defineVar(SEXP sym, SEXP val, SEXP env);
SEXP Rf_findVarInFrame(SEXP env, SEXP sym);
Rboolean R_existsVarInFrame(SEXP env, SEXP sym);
SEXP Rf_findFun(SEXP sym, SEXP env);
SEXP R_do_slot(SEXP obj, SEXP name);
SEXP R_do_slot_assign(SEXP obj, SEXP name, SEXP value);
Rboolean R_has_slot(SEXP obj, SEXP name);

// ---- external pointers ------------------------------------------------------
SEXP  R_MakeExternalPtr(void *p, SEXP tag, SEXP prot);
void *R_ExternalPtrAddr(SEXP s);
void  R_SetExternalPtrAddr(SEXP s, void *p);
typedef void (*R_CFinalizer_t)(SEXP);
void  R_RegisterCFinalizerEx(SEXP s, R_CFinalizer_t fun, Rboolean onexit);

// ---- protection (no-ops; arena freed at call boundary) ----------------------
SEXP Rf_protect(SEXP x);
void Rf_unprotect(int n);
void Rf_unprotect_ptr(SEXP x);
void R_ProtectWithIndex(SEXP x, int *out);
void R_Reprotect(SEXP x, int idx);
#define PROTECT(x)               Rf_protect(x)
#define UNPROTECT(n)             Rf_unprotect(n)
#define UNPROTECT_PTR(x)         Rf_unprotect_ptr(x)
#define PROTECT_WITH_INDEX(x, i) R_ProtectWithIndex(x, i)
#define REPROTECT(x, i)          R_Reprotect(x, i)
typedef int PROTECT_INDEX;

// ---- misc -------------------------------------------------------------------
void Rf_error(const char *fmt, ...);
void Rf_warning(const char *fmt, ...);
void R_CheckUserInterrupt(void);
Rboolean R_ToplevelExec(void (*fun)(void *), void *data);

// NOTE: we deliberately do NOT define lowercase Rf_-less aliases (length, error,
// install, allocVector, ...) — they would clobber STL/Python symbols (e.g.
// std::string::length).  Modern SeqArray uses Rf_-prefixed names; any genuinely
// bare usage is given a targeted alias below, added as the compiler demands.

// ---- shim-internal arena management (called from cclib.cpp boundary) --------
void Rsh_arena_reset(void);   // free all cells allocated since last reset

#ifdef __cplusplus
}
#endif

#endif  // PYSEQARRAY_RSHIM_RINTERNALS_H_
