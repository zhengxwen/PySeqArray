// vcf_api.cpp — VCF import/export entry points.
//
// VCF is the one place SEXP genuinely stays: SeqArray's SEQ_VCF_Parse / SEQ_ToVCF_*
// take/return SEXP (header/param lists, the GDS root, the connection).  This file
// owns the Python<->SEXP marshalling for those calls; the rest of PySeqArray is
// SEXP-free.  The GDS data is written through the engine's GDS_Array_AppendData
// (via the gds_bridge), reading the input VCF through the gzFile connection layer
// (conn.cpp).

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PYSEQARRAY_ARRAY_API
#include <numpy/arrayobject.h>

#include <string>
#include <vector>

#include <PyGDS.h>                 // GDS_ID2FileRoot
#include "Rshim/Rshim_cpp.h"      // shim SEXP + arena
#include "Rshim/Rshim_error.h"
#include "vcf_api.h"

// SeqArray VCF entry points (SEXP signatures)
extern "C" {
SEXP SEQ_VCF_Parse(SEXP vcf_fn, SEXP header, SEXP gds_root, SEXP param,
                   SEXP line_cnt, SEXP rho);
}

// ===========================================================================
// Python value  ->  shim SEXP  (recursive)
// ===========================================================================

static SEXP py_to_sexp(PyObject *o);

static SEXP list_to_sexp(PyObject *seq)
{
    Py_ssize_t n = PySequence_Size(seq);
    // homogeneity probe
    bool all_int = n > 0, all_float = n > 0, all_str = n > 0, all_bool = n > 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *e = PySequence_GetItem(seq, i);
        if (!PyBool_Check(e)) all_bool = false;
        if (!PyLong_Check(e) || PyBool_Check(e)) all_int = false;
        if (!PyFloat_Check(e)) all_float = false;
        if (!(PyUnicode_Check(e) || PyBytes_Check(e))) all_str = false;
        Py_DECREF(e);
    }
    SEXP out;
    if (all_bool) {
        out = Rf_allocVector(LGLSXP, n);
        for (Py_ssize_t i = 0; i < n; i++) { PyObject *e = PySequence_GetItem(seq, i); LOGICAL(out)[i] = (e == Py_True); Py_DECREF(e); }
    } else if (all_int) {
        out = Rf_allocVector(INTSXP, n);
        for (Py_ssize_t i = 0; i < n; i++) { PyObject *e = PySequence_GetItem(seq, i); INTEGER(out)[i] = (int)PyLong_AsLong(e); Py_DECREF(e); }
    } else if (all_float) {
        out = Rf_allocVector(REALSXP, n);
        for (Py_ssize_t i = 0; i < n; i++) { PyObject *e = PySequence_GetItem(seq, i); REAL(out)[i] = PyFloat_AsDouble(e); Py_DECREF(e); }
    } else if (all_str) {
        out = Rf_allocVector(STRSXP, n);
        for (Py_ssize_t i = 0; i < n; i++) {
            PyObject *e = PySequence_GetItem(seq, i);
            PyObject *s = PyObject_Str(e);
            SET_STRING_ELT(out, i, Rf_mkChar(s ? PyUnicode_AsUTF8(s) : ""));
            Py_XDECREF(s); Py_DECREF(e);
        }
    } else {
        out = Rf_allocVector(VECSXP, n);
        for (Py_ssize_t i = 0; i < n; i++) { PyObject *e = PySequence_GetItem(seq, i); SET_VECTOR_ELT(out, i, py_to_sexp(e)); Py_DECREF(e); }
    }
    return out;
}

static SEXP dict_to_sexp(PyObject *d)
{
    PyObject *keys = PyDict_Keys(d);
    Py_ssize_t n = PyList_Size(keys);
    SEXP out = Rf_allocVector(VECSXP, n);
    SEXP names = Rf_allocVector(STRSXP, n);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *k = PyList_GetItem(keys, i);            // borrowed
        PyObject *ks = PyObject_Str(k);
        SET_STRING_ELT(names, i, Rf_mkChar(ks ? PyUnicode_AsUTF8(ks) : ""));
        Py_XDECREF(ks);
        SET_VECTOR_ELT(out, i, py_to_sexp(PyDict_GetItem(d, k)));  // borrowed val
    }
    Rf_setAttrib(out, R_NamesSymbol, names);
    Py_DECREF(keys);
    return out;
}

static SEXP py_to_sexp(PyObject *o)
{
    if (o == NULL || o == Py_None) return R_NilValue;
    if (PyBool_Check(o)) return Rf_ScalarLogical(o == Py_True);
    if (PyLong_Check(o)) return Rf_ScalarInteger((int)PyLong_AsLong(o));
    if (PyFloat_Check(o)) return Rf_ScalarReal(PyFloat_AsDouble(o));
    if (PyUnicode_Check(o)) return Rf_mkString(PyUnicode_AsUTF8(o));
    if (PyBytes_Check(o)) return Rf_mkString(PyBytes_AsString(o));
    if (PyDict_Check(o)) return dict_to_sexp(o);
    if (PyArray_Check(o)) {
        PyObject *lst = PySequence_List(o);
        SEXP r = list_to_sexp(lst);
        Py_DECREF(lst);
        return r;
    }
    if (PyList_Check(o) || PyTuple_Check(o)) return list_to_sexp(o);
    // fallback: stringify
    PyObject *s = PyObject_Str(o);
    SEXP r = Rf_mkString(s ? PyUnicode_AsUTF8(s) : "");
    Py_XDECREF(s);
    return r;
}

// ===========================================================================
// debug: round-trip a Python value through the shim SEXP (validates the marshaller)
// ===========================================================================

static PyObject *sexp_to_py(SEXP x);

static PyObject *sexp_to_py(SEXP x)
{
    if (x == NULL || TYPEOF(x) == NILSXP) Py_RETURN_NONE;
    R_xlen_t n = Rf_xlength(x);
    switch (TYPEOF(x)) {
        case LGLSXP: { PyObject *l = PyList_New(n); for (R_xlen_t i=0;i<n;i++) PyList_SetItem(l,i,PyBool_FromLong(LOGICAL(x)[i])); return l; }
        case INTSXP: { PyObject *l = PyList_New(n); for (R_xlen_t i=0;i<n;i++) PyList_SetItem(l,i,PyLong_FromLong(INTEGER(x)[i])); return l; }
        case REALSXP:{ PyObject *l = PyList_New(n); for (R_xlen_t i=0;i<n;i++) PyList_SetItem(l,i,PyFloat_FromDouble(REAL(x)[i])); return l; }
        case STRSXP: { PyObject *l = PyList_New(n); for (R_xlen_t i=0;i<n;i++) PyList_SetItem(l,i,PyUnicode_FromString(CHAR(STRING_ELT(x,i)))); return l; }
        case VECSXP: {
            SEXP names = Rf_getAttrib(x, R_NamesSymbol);
            if (names != NULL && TYPEOF(names) == STRSXP) {
                PyObject *d = PyDict_New();
                for (R_xlen_t i=0;i<n;i++) { PyObject *v = sexp_to_py(VECTOR_ELT(x,i)); PyDict_SetItemString(d, CHAR(STRING_ELT(names,i)), v); Py_DECREF(v); }
                return d;
            }
            PyObject *l = PyList_New(n);
            for (R_xlen_t i=0;i<n;i++) PyList_SetItem(l,i,sexp_to_py(VECTOR_ELT(x,i)));
            return l;
        }
        default: Py_RETURN_NONE;
    }
}

PyObject *PySeq_sexp_roundtrip(PyObject *, PyObject *args)
{
    PyObject *o;
    if (!PyArg_ParseTuple(args, "O", &o)) return NULL;
    PyObject *res = NULL;
    try {
        SEXP s = py_to_sexp(o);
        res = sexp_to_py(s);
    } catch (std::exception &e) {
        Rsh_arena_reset();
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    Rsh_arena_reset();
    return res;
}

// ===========================================================================
// seqVCF2GDS: parse the VCF body into the (already-skeletoned) output GDS
// ===========================================================================

PyObject *PySeq_vcf_parse(PyObject *, PyObject *args)
{
    const char *vcf_path;
    PyObject *header, *param;
    int out_fileid;
    if (!PyArg_ParseTuple(args, "sOiO", &vcf_path, &header, &out_fileid, &param))
        return NULL;
    PyObject *result = NULL;
    try {
        SEXP vcf_fn = Rf_mkString(vcf_path);
        SEXP hdr = py_to_sexp(header);
        SEXP par = py_to_sexp(param);
        // GDS root of the output file as an external pointer (GDS_R_SEXP2Obj reads it)
        PdGDSFolder root = GDS_ID2FileRoot(out_fileid);
        if (root == NULL) throw Rsh_error("invalid output GDS file id");
        SEXP gds_root = R_MakeExternalPtr((void *)root, R_NilValue, R_NilValue);
        SEXP line_cnt = Rf_ScalarReal(0);
        SEXP r = SEQ_VCF_Parse(vcf_fn, hdr, gds_root, par, line_cnt, R_NilValue);
        // r is the number of parsed variant lines (a scalar)
        double nline = (TYPEOF(r) == REALSXP || TYPEOF(r) == INTSXP) ? Rf_asReal(r) : 0;
        result = PyLong_FromDouble(nline);
    } catch (Rsh_error &e) {
        Rsh_arena_reset();
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    } catch (std::exception &e) {
        Rsh_arena_reset();
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    Rsh_arena_reset();
    return result;
}
