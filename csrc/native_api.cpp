// native_api.cpp — PySeqArray's SEXP-free native entry layer (option B).
//
// Instead of calling SeqArray's R-style SEQ_* entry points (which take/return
// SEXP and need the Rshim marshalling boundary), these functions drive the
// engine's C++ classes directly (CFileInfo / TSelection) and build numpy arrays
// with zero SEXP.  This is the template the remaining entry points follow as the
// engine is migrated off the R object model.  PySeqArray owns this layer, so it
// is freely optimizable independently of upstream SeqArray.

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PYSEQARRAY_ARRAY_API
#include <numpy/arrayobject.h>

#include "Rshim/Rshim_error.h"
#include "native_api.h"
#include "Index.h"            // CFileInfo, TSelection, GetFileInfo(int)

using namespace SeqArray;

// run a native lambda, mapping engine C++ exceptions to a Python error.
template <class F>
static PyObject *native_guard(F fn)
{
    try {
        return fn();
    } catch (Rsh_error &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "unknown native error");
    }
    return NULL;
}

// the current sample/variant selection as a numpy bool array — read straight
// from TSelection::pSample / pVariant, no SEXP.
static PyObject *get_selection(int fileid, bool sample)
{
    CFileInfo &File = GetFileInfo(fileid);
    TSelection &sel = File.Selection();
    int n = sample ? File.SampleNum() : File.VariantNum();
    const C_BOOL *p = sample ? sel.pSample : sel.pVariant;
    npy_intp dn = n;
    PyObject *arr = PyArray_SimpleNew(1, &dn, NPY_BOOL);
    npy_bool *o = (npy_bool *)PyArray_DATA((PyArrayObject *)arr);
    for (int i = 0; i < n; i++) o[i] = p[i] ? 1 : 0;
    return arr;
}

PyObject *PySeq_native_dims(PyObject *, PyObject *args)
{
    int fid;
    if (!PyArg_ParseTuple(args, "i", &fid)) return NULL;
    return native_guard([&]() -> PyObject * {
        CFileInfo &F = GetFileInfo(fid);
        return Py_BuildValue("(iiiii)", F.SampleNum(), F.VariantNum(),
                             F.Ploidy(), F.SampleSelNum(), F.VariantSelNum());
    });
}

PyObject *PySeq_native_get_sample_sel(PyObject *, PyObject *args)
{
    int fid;
    if (!PyArg_ParseTuple(args, "i", &fid)) return NULL;
    return native_guard([&] { return get_selection(fid, true); });
}

PyObject *PySeq_native_get_variant_sel(PyObject *, PyObject *args)
{
    int fid;
    if (!PyArg_ParseTuple(args, "i", &fid)) return NULL;
    return native_guard([&] { return get_selection(fid, false); });
}
