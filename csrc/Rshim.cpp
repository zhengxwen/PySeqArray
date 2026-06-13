// Rshim.cpp — implementation of the bounded R C-API subset (see Rshim/Rinternals.h).
//
// A SEXP is a heap-allocated PYSEXPREC tracked in a global arena.  Nothing is
// freed during an entry-point call (PROTECT is a no-op); cclib.cpp calls
// Rsh_arena_reset() after converting a result to a PyObject, freeing everything.
//
// Errors: Rf_error throws Rsh_error (a C++ exception); cclib.cpp catches it at
// the boundary and converts to a Python exception.

#include "Rshim/Rinternals.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <map>

#include "Rshim/Rshim_cpp.h"   // Rsh_error, PYSEXPREC layout (C++ only)

// ---- the arena --------------------------------------------------------------
static std::vector<PYSEXPREC*> g_arena;

static SEXP alloc_cell(SEXPTYPE t, R_xlen_t n)
{
    PYSEXPREC *c = new PYSEXPREC();
    c->type = t;
    c->length = n;
    g_arena.push_back(c);
    return c;
}

extern "C" void Rsh_arena_reset(void)
{
    for (size_t i = 0; i < g_arena.size(); i++) delete g_arena[i];
    g_arena.clear();
}

// ---- globals ----------------------------------------------------------------
double R_NaReal;
double R_NaN;
double R_PosInf;
double R_NegInf;
int    R_NaInt = (-2147483647 - 1);   // R's NA_INTEGER == INT_MIN
SEXP R_NaString = NULL;
SEXP R_NilValue = NULL;
SEXP R_GlobalEnv = NULL;
SEXP R_EmptyEnv = NULL;
SEXP R_DotsSymbol = NULL;
SEXP R_NamesSymbol = NULL;
SEXP R_ClassSymbol = NULL;
SEXP R_LevelsSymbol = NULL;
SEXP R_DimSymbol = NULL;
SEXP R_DimNamesSymbol = NULL;
SEXP R_RowNamesSymbol = NULL;

// interned symbols (never freed; live for process lifetime, not in arena)
static std::map<std::string, SEXP> g_symbols;

static SEXP intern_symbol(const char *name)
{
    std::string key(name);
    std::map<std::string, SEXP>::iterator it = g_symbols.find(key);
    if (it != g_symbols.end()) return it->second;
    PYSEXPREC *c = new PYSEXPREC();   // persistent, not arena-tracked
    c->type = SYMSXP;
    c->str.assign(1, key);
    g_symbols[key] = c;
    return c;
}

// one-time init of the shim globals (called from cclib module init)
extern "C" void Rsh_init_globals(void)
{
    static bool done = false;
    if (done) return;
    done = true;
    // NaN / NA
    R_NaN = nan("");
    R_PosInf = HUGE_VAL;
    R_NegInf = -HUGE_VAL;
    // R's NA_real_ is a specific quiet-NaN bit pattern; for the shim any NaN with
    // a recognizable payload works as long as ISNA round-trips.
    {
        volatile double x; unsigned int *p = (unsigned int*)&x;
        p[0] = 1954; p[1] = 0x7FF00000;   // R's NA_REAL low-word marker (1954)
        R_NaReal = x;
    }
    R_NaInt = (-2147483647 - 1);
    // persistent singletons
    PYSEXPREC *nil = new PYSEXPREC(); nil->type = NILSXP; nil->length = 0;
    R_NilValue = nil;
    PYSEXPREC *ge = new PYSEXPREC(); ge->type = ENVSXP; R_GlobalEnv = ge;
    PYSEXPREC *ee = new PYSEXPREC(); ee->type = ENVSXP; R_EmptyEnv = ee;
    R_DotsSymbol    = intern_symbol("...");
    R_NamesSymbol   = intern_symbol("names");
    R_ClassSymbol   = intern_symbol("class");
    R_LevelsSymbol  = intern_symbol("levels");
    R_DimSymbol     = intern_symbol("dim");
    R_DimNamesSymbol= intern_symbol("dimnames");
    R_RowNamesSymbol= intern_symbol("row.names");
    PYSEXPREC *nastr = new PYSEXPREC(); nastr->type = CHARSXP; nastr->na = true;
    R_NaString = nastr;
}

int Rsh_is_na_real(double x)
{
    if (!isnan(x)) return 0;
    unsigned int *p = (unsigned int*)&x;
    return (p[0] == 1954);
}

// ---- allocation -------------------------------------------------------------
SEXP Rf_allocVector(SEXPTYPE type, R_xlen_t n)
{
    SEXP c = alloc_cell(type, n);
    switch (type) {
        case INTSXP: case LGLSXP: c->i.assign(n, 0); break;
        case REALSXP:             c->d.assign(n, 0.0); break;
        case RAWSXP:              c->raw.assign(n, 0); break;
        case STRSXP:              c->list.assign(n, R_NilValue); break;
        case VECSXP:              c->list.assign(n, R_NilValue); break;
        case CHARSXP:             c->str.assign(1, std::string((size_t)n, '\0')); break;
        default:                  break;
    }
    return c;
}

SEXP Rf_allocMatrix(SEXPTYPE type, int nrow, int ncol)
{
    SEXP m = Rf_allocVector(type, (R_xlen_t)nrow * ncol);
    SEXP dim = Rf_allocVector(INTSXP, 2);
    dim->i[0] = nrow; dim->i[1] = ncol;
    Rf_setAttrib(m, R_DimSymbol, dim);
    return m;
}

SEXP Rf_ScalarInteger(int x) { SEXP s = Rf_allocVector(INTSXP, 1); s->i[0] = x; return s; }
SEXP Rf_ScalarReal(double x) { SEXP s = Rf_allocVector(REALSXP, 1); s->d[0] = x; return s; }
SEXP Rf_ScalarLogical(int x) { SEXP s = Rf_allocVector(LGLSXP, 1); s->i[0] = x; return s; }
SEXP Rf_ScalarString(SEXP x) { SEXP s = Rf_allocVector(STRSXP, 1); s->list[0] = x; return s; }

SEXP Rf_mkCharLen(const char *s, int len)
{
    SEXP c = alloc_cell(CHARSXP, len);
    c->str.assign(1, std::string(s, s + len));
    return c;
}
SEXP Rf_mkChar(const char *s) { return Rf_mkCharLen(s, s ? (int)strlen(s) : 0); }
SEXP Rf_mkString(const char *s) { return Rf_ScalarString(Rf_mkChar(s)); }

// ---- length / type ----------------------------------------------------------
R_xlen_t Rf_xlength(SEXP x)
{
    if (x == NULL || x == R_NilValue) return 0;
    switch (x->type) {
        case INTSXP: case LGLSXP: return (R_xlen_t)x->i.size();
        case REALSXP:             return (R_xlen_t)x->d.size();
        case RAWSXP:              return (R_xlen_t)x->raw.size();
        case STRSXP: case VECSXP: return (R_xlen_t)x->list.size();
        case CHARSXP:             return x->str.empty() ? 0 : (R_xlen_t)x->str[0].size();
        default: return 0;
    }
}
R_len_t  Rf_length(SEXP x) { return (R_len_t)Rf_xlength(x); }
SEXPTYPE TYPEOF_(SEXP x) { return (x == NULL) ? NILSXP : (SEXPTYPE)x->type; }

// ---- element access ---------------------------------------------------------
int    *INTEGER(SEXP x) { return x->i.empty() ? NULL : &x->i[0]; }
double *REAL(SEXP x)    { return x->d.empty() ? NULL : &x->d[0]; }
int    *LOGICAL(SEXP x) { return x->i.empty() ? NULL : &x->i[0]; }
Rbyte  *RAW(SEXP x)     { return x->raw.empty() ? NULL : &x->raw[0]; }

const char *R_CHAR(SEXP c)
{
    if (c == NULL || c == R_NilValue || c->str.empty()) return "";
    return c->str[0].c_str();
}
SEXP STRING_ELT(SEXP x, R_xlen_t i) { return x->list[i]; }
SEXP VECTOR_ELT(SEXP x, R_xlen_t i) { return x->list[i]; }
void SET_STRING_ELT(SEXP x, R_xlen_t i, SEXP v) { x->list[i] = v; }
SEXP SET_VECTOR_ELT(SEXP x, R_xlen_t i, SEXP v) { x->list[i] = v; return v; }
const char *Rf_translateChar(SEXP x)     { return R_CHAR(x); }
const char *Rf_translateCharUTF8(SEXP x) { return R_CHAR(x); }

// ---- coercion scalars -------------------------------------------------------
int Rf_asInteger(SEXP x)
{
    if (x == NULL || x == R_NilValue || Rf_xlength(x) == 0) return NA_INTEGER;
    switch (x->type) {
        case INTSXP: case LGLSXP: return x->i[0];
        case REALSXP: return Rsh_is_na_real(x->d[0]) || isnan(x->d[0]) ? NA_INTEGER : (int)x->d[0];
        case RAWSXP:  return (int)x->raw[0];
        case STRSXP:  { const char *s = R_CHAR(x->list[0]); return s[0] ? atoi(s) : NA_INTEGER; }
        default: return NA_INTEGER;
    }
}
double Rf_asReal(SEXP x)
{
    if (x == NULL || x == R_NilValue || Rf_xlength(x) == 0) return NA_REAL;
    switch (x->type) {
        case INTSXP: case LGLSXP: return x->i[0] == NA_INTEGER ? NA_REAL : (double)x->i[0];
        case REALSXP: return x->d[0];
        case RAWSXP:  return (double)x->raw[0];
        case STRSXP:  { const char *s = R_CHAR(x->list[0]); return s[0] ? atof(s) : NA_REAL; }
        default: return NA_REAL;
    }
}
int Rf_asLogical(SEXP x)
{
    if (x == NULL || x == R_NilValue || Rf_xlength(x) == 0) return NA_LOGICAL;
    switch (x->type) {
        case INTSXP: case LGLSXP: return x->i[0];
        case REALSXP: return isnan(x->d[0]) ? NA_LOGICAL : (x->d[0] != 0.0);
        case RAWSXP:  return x->raw[0] != 0;
        default: return NA_LOGICAL;
    }
}
SEXP Rf_asChar(SEXP x)
{
    if (x == NULL || Rf_xlength(x) == 0) return R_NaString;
    if (x->type == STRSXP) return x->list[0];
    if (x->type == CHARSXP) return x;
    char buf[64];
    if (x->type == REALSXP) snprintf(buf, sizeof buf, "%g", x->d[0]);
    else snprintf(buf, sizeof buf, "%d", x->i[0]);
    return Rf_mkChar(buf);
}

// ---- predicates -------------------------------------------------------------
Rboolean Rf_isNull(SEXP x)   { return (x == NULL || x->type == NILSXP) ? TRUE : FALSE; }
Rboolean Rf_isString(SEXP x) { return (x && x->type == STRSXP) ? TRUE : FALSE; }
Rboolean Rf_isReal(SEXP x)   { return (x && x->type == REALSXP) ? TRUE : FALSE; }
Rboolean Rf_isInteger(SEXP x){ return (x && x->type == INTSXP && !Rf_isFactor(x)) ? TRUE : FALSE; }
Rboolean Rf_isLogical(SEXP x){ return (x && x->type == LGLSXP) ? TRUE : FALSE; }
Rboolean Rf_isNumeric(SEXP x){ return (x && (x->type == INTSXP || x->type == REALSXP) && !Rf_isFactor(x)) ? TRUE : FALSE; }
Rboolean Rf_isVector(SEXP x) { if(!x) return FALSE; switch(x->type){case INTSXP:case LGLSXP:case REALSXP:case RAWSXP:case STRSXP:case VECSXP:return TRUE;default:return FALSE;} }
Rboolean Rf_isVectorList(SEXP x){ return (x && x->type == VECSXP) ? TRUE : FALSE; }
Rboolean Rf_isList(SEXP x)   { return (x && (x->type == LISTSXP || x->type == NILSXP)) ? TRUE : FALSE; }
Rboolean Rf_isEnvironment(SEXP x){ return (x && x->type == ENVSXP) ? TRUE : FALSE; }
Rboolean Rf_isObject(SEXP x) { return (x && x->attrib.count("class")) ? TRUE : FALSE; }
Rboolean Rf_isFactor(SEXP x)
{
    if (!x || x->type != INTSXP) return FALSE;
    return x->attrib.count("levels") ? TRUE : FALSE;
}
Rboolean Rf_inherits(SEXP x, const char *name)
{
    if (!x) return FALSE;
    std::map<std::string,SEXP>::iterator it = x->attrib.find("class");
    if (it == x->attrib.end()) return FALSE;
    SEXP cls = it->second;
    for (R_xlen_t i = 0; i < Rf_xlength(cls); i++)
        if (strcmp(R_CHAR(STRING_ELT(cls, i)), name) == 0) return TRUE;
    return FALSE;
}

// ---- attributes -------------------------------------------------------------
SEXP Rf_getAttrib(SEXP x, SEXP sym)
{
    if (!x || sym == NULL || sym->str.empty()) return R_NilValue;
    std::map<std::string,SEXP>::iterator it = x->attrib.find(sym->str[0]);
    return it == x->attrib.end() ? R_NilValue : it->second;
}
SEXP Rf_setAttrib(SEXP x, SEXP sym, SEXP val)
{
    if (x && sym && !sym->str.empty()) x->attrib[sym->str[0]] = val;
    return val;
}
SEXP Rf_install(const char *name) { return intern_symbol(name); }
SEXP Rf_GetRowNames(SEXP x) { return Rf_getAttrib(x, R_RowNamesSymbol); }

// ---- duplicate / coerce -----------------------------------------------------
SEXP Rf_duplicate(SEXP x)
{
    if (x == NULL || x == R_NilValue) return x;
    SEXP c = alloc_cell((SEXPTYPE)x->type, x->length);
    c->i = x->i; c->d = x->d; c->raw = x->raw; c->list = x->list;
    c->str = x->str; c->attrib = x->attrib; c->na = x->na; c->ptr = x->ptr;
    return c;
}
SEXP Rf_coerceVector(SEXP x, SEXPTYPE type)
{
    if (x && x->type == type) return x;
    R_xlen_t n = Rf_xlength(x);
    SEXP out = Rf_allocVector(type, n);
    for (R_xlen_t i = 0; i < n; i++) {
        double dv; int iv;
        switch (x->type) {
            case INTSXP: case LGLSXP: iv = x->i[i]; dv = (iv==NA_INTEGER)?NA_REAL:iv; break;
            case REALSXP: dv = x->d[i]; iv = isnan(dv)?NA_INTEGER:(int)dv; break;
            case RAWSXP: iv = x->raw[i]; dv = iv; break;
            default: iv = NA_INTEGER; dv = NA_REAL; break;
        }
        switch (type) {
            case INTSXP: case LGLSXP: out->i[i] = iv; break;
            case REALSXP: out->d[i] = dv; break;
            case STRSXP: { char b[64]; snprintf(b,sizeof b,"%g",dv); out->list[i] = Rf_mkChar(b); } break;
            default: break;
        }
    }
    return out;
}
SEXP Rf_asCharacterFactor(SEXP x)
{
    SEXP lv = Rf_getAttrib(x, R_LevelsSymbol);
    R_xlen_t n = Rf_xlength(x);
    SEXP out = Rf_allocVector(STRSXP, n);
    for (R_xlen_t i = 0; i < n; i++) {
        int code = x->i[i];
        if (code == NA_INTEGER || code < 1 || code > Rf_xlength(lv))
            out->list[i] = R_NaString;
        else
            out->list[i] = STRING_ELT(lv, code - 1);
    }
    return out;
}

// ---- protection (no-ops) ----------------------------------------------------
SEXP Rf_protect(SEXP x) { return x; }
void Rf_unprotect(int) {}
void Rf_unprotect_ptr(SEXP) {}
void R_ProtectWithIndex(SEXP, int *out) { if (out) *out = 0; }
void R_Reprotect(SEXP, int) {}

// ---- errors -----------------------------------------------------------------
void Rf_error(const char *fmt, ...)
{
    char buf[1024];
    va_list ap; va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    throw Rsh_error(buf);
}
void Rf_warning(const char *fmt, ...)
{
    char buf[1024];
    va_list ap; va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    fprintf(stderr, "PySeqArray warning: %s\n", buf);
}
void R_CheckUserInterrupt(void) {}
Rboolean R_ToplevelExec(void (*fun)(void *), void *data) { fun(data); return TRUE; }

// ---- environments / eval / slots: stubbed (apply driven from Python) --------
SEXP Rf_eval(SEXP, SEXP) { throw Rsh_error("Rf_eval: R evaluation is not available in PySeqArray (apply is driven from Python)"); }
SEXP Rf_lang1(SEXP) { return R_NilValue; }
SEXP Rf_lang2(SEXP, SEXP) { return R_NilValue; }
SEXP Rf_lang3(SEXP, SEXP, SEXP) { return R_NilValue; }
void Rf_defineVar(SEXP, SEXP, SEXP) {}
SEXP Rf_findVarInFrame(SEXP, SEXP) { return R_NilValue; }
Rboolean R_existsVarInFrame(SEXP, SEXP) { return FALSE; }
SEXP Rf_findFun(SEXP, SEXP) { return R_NilValue; }
SEXP R_do_slot(SEXP, SEXP) { return R_NilValue; }
SEXP R_do_slot_assign(SEXP obj, SEXP, SEXP) { return obj; }
Rboolean R_has_slot(SEXP, SEXP) { return FALSE; }

// ---- external pointers ------------------------------------------------------
SEXP R_MakeExternalPtr(void *p, SEXP, SEXP)
{
    SEXP c = alloc_cell(EXTPTRSXP, 0); c->ptr = p; return c;
}
void *R_ExternalPtrAddr(SEXP s) { return s ? s->ptr : NULL; }
void  R_SetExternalPtrAddr(SEXP s, void *p) { if (s) s->ptr = p; }
void  R_RegisterCFinalizerEx(SEXP, R_CFinalizer_t, Rboolean) {}
// R connection layer is implemented in csrc/conn.cpp (zlib gzFile-backed).
