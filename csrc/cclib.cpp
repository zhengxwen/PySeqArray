// cclib.cpp — CPython C-API module for PySeqArray.
//
// Replaces SeqArray's R_init_SeqArray glue: binds the pygds GDS C-API (via the
// pygds.ccall._GDS_C_API PyCapsule, imported by Init_GDS_Routines) and exposes
// the SeqArray SEQ_* engine entry points as Python callables.
//
// Each wrapper builds shim SEXP arguments from Python, calls SEQ_*, converts the
// result SEXP to a numpy/Python object, then releases the shim arena.  Rf_error
// inside the engine throws Rsh_error, caught here and raised as a Python error.

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL PYSEQARRAY_ARRAY_API
#include <numpy/arrayobject.h>

#include <PyGDS2.h>                 // GDS_* capsule wrappers + Init_GDS_Routines
#include "Rshim/Rshim_cpp.h"       // SEXP cell, Rsh_init_globals, Rsh_arena_reset
#include "Rshim/Rshim_error.h"     // Rsh_error
#include "native_api.h"            // SEXP-free native entry layer (option B)

// ---- SeqArray engine entry points (subset wired in B1) ----------------------
extern "C" {
SEXP SEQ_File_Init(SEXP gdsfile);
SEXP SEQ_GetData(SEXP gdsfile, SEXP var_name, SEXP UseRaw, SEXP PadNA, SEXP ToList, SEXP Env);
SEXP SEQ_SetSpaceSample(SEXP gdsfile, SEXP samp_id, SEXP intersect, SEXP verbose);
SEXP SEQ_SetSpaceVariant(SEXP gdsfile, SEXP var_id, SEXP intersect, SEXP verbose);
SEXP SEQ_SetSpaceSample2(SEXP gdsfile, SEXP samp_sel, SEXP intersect, SEXP warn, SEXP verbose);
SEXP SEQ_SetSpaceVariant2(SEXP gdsfile, SEXP var_sel, SEXP intersect, SEXP warn, SEXP verbose);
SEXP SEQ_GetSpaceSample(SEXP gdsfile);
SEXP SEQ_GetSpaceVariant(SEXP gdsfile);
SEXP SEQ_Summary(SEXP gdsfile, SEXP varname);
}

// ===========================================================================
// Python  ->  shim SEXP
// ===========================================================================

// gds.class shape: VECSXP{ id=ScalarInteger(fileid) } with names = {"id"}.
static SEXP make_gdsfile(int fileid)
{
    SEXP lst = Rf_allocVector(VECSXP, 1);
    SET_VECTOR_ELT(lst, 0, Rf_ScalarInteger(fileid));
    SEXP nm = Rf_allocVector(STRSXP, 1);
    SET_STRING_ELT(nm, 0, Rf_mkChar("id"));
    Rf_setAttrib(lst, R_NamesSymbol, nm);
    return lst;
}

// A selection argument: None -> R_NilValue; a numpy bool array -> LGLSXP.
static SEXP py_to_selection(PyObject *o)
{
    if (o == NULL || o == Py_None) return R_NilValue;
    PyArrayObject *a = (PyArrayObject*)PyArray_FROM_OTF(o, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    if (a == NULL) throw Rsh_error("selection must be a 1-D boolean array or None");
    R_xlen_t n = (R_xlen_t)PyArray_SIZE(a);
    SEXP s = Rf_allocVector(LGLSXP, n);
    const npy_bool *p = (const npy_bool*)PyArray_DATA(a);
    for (R_xlen_t i = 0; i < n; i++) LOGICAL(s)[i] = p[i] ? 1 : 0;
    Py_DECREF(a);
    return s;
}

static SEXP py_bool(int v) { return Rf_ScalarLogical(v ? 1 : 0); }

// ===========================================================================
// shim SEXP  ->  Python / numpy
// ===========================================================================

static PyObject* sexp_dims_tuple(SEXP x, int *out_nd)
{
    // returns a numpy shape (reversed R dim -> C-order) or NULL for 1-D
    SEXP dim = Rf_getAttrib(x, R_DimSymbol);
    int nd = (int)Rf_xlength(dim);
    *out_nd = nd;
    if (nd <= 1) return NULL;
    PyObject *shp = PyTuple_New(nd);
    for (int k = 0; k < nd; k++)
        PyTuple_SetItem(shp, k, PyLong_FromLong(INTEGER(dim)[nd - 1 - k]));
    return shp;
}

static PyObject* reshape(PyObject *flat, SEXP x)
{
    int nd; PyObject *shp = sexp_dims_tuple(x, &nd);
    if (shp == NULL) return flat;
    PyObject *r = PyArray_Reshape((PyArrayObject*)flat, shp);
    Py_DECREF(shp); Py_DECREF(flat);
    return r;
}

static PyObject* sexp_to_py(SEXP x)
{
    if (x == NULL || TYPEOF(x) == NILSXP) Py_RETURN_NONE;
    R_xlen_t n = Rf_xlength(x);
    switch (TYPEOF(x)) {
        case INTSXP: case LGLSXP: {
            npy_intp dn = (npy_intp)n;
            PyObject *arr = PyArray_SimpleNew(1, &dn, NPY_INT32);
            memcpy(PyArray_DATA((PyArrayObject*)arr), INTEGER(x), n * sizeof(int));
            return reshape(arr, x);
        }
        case REALSXP: {
            npy_intp dn = (npy_intp)n;
            PyObject *arr = PyArray_SimpleNew(1, &dn, NPY_FLOAT64);
            memcpy(PyArray_DATA((PyArrayObject*)arr), REAL(x), n * sizeof(double));
            return reshape(arr, x);
        }
        case RAWSXP: {
            npy_intp dn = (npy_intp)n;
            PyObject *arr = PyArray_SimpleNew(1, &dn, NPY_UINT8);
            memcpy(PyArray_DATA((PyArrayObject*)arr), RAW(x), n);
            return reshape(arr, x);
        }
        case STRSXP: {
            PyObject *lst = PyList_New(n);
            for (R_xlen_t i = 0; i < n; i++) {
                SEXP e = STRING_ELT(x, i);
                if (e == NULL || TYPEOF(e) == NILSXP)
                    { Py_INCREF(Py_None); PyList_SetItem(lst, i, Py_None); }
                else
                    PyList_SetItem(lst, i, PyUnicode_FromString(CHAR(e)));
            }
            return lst;
        }
        case VECSXP: {
            PyObject *lst = PyList_New(n);
            for (R_xlen_t i = 0; i < n; i++)
                PyList_SetItem(lst, i, sexp_to_py(VECTOR_ELT(x, i)));
            return lst;
        }
        default:
            Py_RETURN_NONE;
    }
}

// ===========================================================================
// boundary: run a SEXP-returning lambda, convert + reset arena + map errors
// ===========================================================================

template <class F>
static PyObject* boundary(F fn)
{
    PyObject *res = NULL;
    try {
        SEXP r = fn();
        res = sexp_to_py(r);
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
    return res;
}

// ---- wrappers ---------------------------------------------------------------

static PyObject* w_file_init(PyObject*, PyObject *args)
{
    int fid;
    if (!PyArg_ParseTuple(args, "i", &fid)) return NULL;
    return boundary([&]{ return SEQ_File_Init(make_gdsfile(fid)); });
}

static PyObject* w_get_data(PyObject*, PyObject *args)
{
    int fid, use_raw = 0;
    const char *name;
    if (!PyArg_ParseTuple(args, "is|i", &fid, &name, &use_raw)) return NULL;
    return boundary([&]{
        return SEQ_GetData(make_gdsfile(fid), Rf_mkString(name),
                           py_bool(use_raw), py_bool(0), py_bool(0), R_NilValue);
    });
}

static PyObject* w_set_sample(PyObject*, PyObject *args)
{
    int fid, intersect = 0;
    PyObject *sel;
    if (!PyArg_ParseTuple(args, "iO|i", &fid, &sel, &intersect)) return NULL;
    return boundary([&]{
        return SEQ_SetSpaceSample2(make_gdsfile(fid), py_to_selection(sel),
                                   py_bool(intersect), py_bool(0), py_bool(0));
    });
}

static PyObject* w_set_variant(PyObject*, PyObject *args)
{
    int fid, intersect = 0;
    PyObject *sel;
    if (!PyArg_ParseTuple(args, "iO|i", &fid, &sel, &intersect)) return NULL;
    return boundary([&]{
        return SEQ_SetSpaceVariant2(make_gdsfile(fid), py_to_selection(sel),
                                    py_bool(intersect), py_bool(0), py_bool(0));
    });
}

// reset selection to all: the id-version with NULL selects everything
static PyObject* w_reset_filter(PyObject*, PyObject *args)
{
    int fid, sample = 1, variant = 1;
    if (!PyArg_ParseTuple(args, "i|ii", &fid, &sample, &variant)) return NULL;
    return boundary([&]() -> SEXP {
        if (sample)
            SEQ_SetSpaceSample(make_gdsfile(fid), R_NilValue, py_bool(0), py_bool(0));
        if (variant)
            SEQ_SetSpaceVariant(make_gdsfile(fid), R_NilValue, py_bool(0), py_bool(0));
        return R_NilValue;
    });
}

static PyObject* w_get_sample_sel(PyObject*, PyObject *args)
{
    int fid;
    if (!PyArg_ParseTuple(args, "i", &fid)) return NULL;
    return boundary([&]{ return SEQ_GetSpaceSample(make_gdsfile(fid)); });
}

static PyObject* w_get_variant_sel(PyObject*, PyObject *args)
{
    int fid;
    if (!PyArg_ParseTuple(args, "i", &fid)) return NULL;
    return boundary([&]{ return SEQ_GetSpaceVariant(make_gdsfile(fid)); });
}

static PyObject* py_init_gds(PyObject*, PyObject*)
{
    if (Init_GDS_Routines() != 0) return NULL;
    Py_RETURN_TRUE;
}

static PyMethodDef cclib_methods[] = {
    {"init_gds",        py_init_gds,       METH_NOARGS,  "Bind the pygds GDS C-API capsule."},
    {"file_init",       w_file_init,       METH_VARARGS, "SEQ_File_Init(fileid): init selection to all."},
    {"get_data",        w_get_data,        METH_VARARGS, "SEQ_GetData(fileid, name, use_raw=0)."},
    {"set_sample",      w_set_sample,      METH_VARARGS, "SEQ_SetSpaceSample(fileid, bool_sel|None, intersect=0)."},
    {"set_variant",     w_set_variant,     METH_VARARGS, "SEQ_SetSpaceVariant(fileid, bool_sel|None, intersect=0)."},
    {"reset_filter",    w_reset_filter,    METH_VARARGS, "Reset selection to all (fileid, sample=1, variant=1)."},
    {"get_sample_sel",  w_get_sample_sel,  METH_VARARGS, "SEQ_GetSpaceSample(fileid) -> logical selection."},
    {"get_variant_sel", w_get_variant_sel, METH_VARARGS, "SEQ_GetSpaceVariant(fileid) -> logical selection."},
    // SEXP-free native entry layer (option B)
    {"n_dims",          PySeq_native_dims,           METH_VARARGS, "Native: (nSamp,nVar,ploidy,nSampSel,nVarSel) via CFileInfo, no SEXP."},
    {"native_sample_sel", PySeq_native_get_sample_sel, METH_VARARGS, "Native sample selection (bool) via TSelection, no SEXP."},
    {"native_variant_sel", PySeq_native_get_variant_sel, METH_VARARGS, "Native variant selection (bool) via TSelection, no SEXP."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cclib_module = {
    PyModuleDef_HEAD_INIT, "cclib",
    "PySeqArray C++ engine (SeqArray) bound to the pygds GDS C-API.",
    -1, cclib_methods, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_cclib(void)
{
    import_array();
    Rsh_init_globals();
    PyObject *m = PyModule_Create(&cclib_module);
    if (!m) return NULL;
    if (Init_GDS_Routines() != 0) { Py_DECREF(m); return NULL; }
    return m;
}
