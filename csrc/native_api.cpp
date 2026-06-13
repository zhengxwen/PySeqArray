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
#include "ReadByVariant.h"    // CApply_Variant_Geno / _Dosage

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

// Native dosage read: CApply_Variant_Dosage over the current selection, filling
// numpy int32 (nVarSel, nSampSel).  alt=false -> reference-allele dosage; true ->
// alternate.  Missing -> NA_INTEGER.  No SEXP.
static PyObject *read_dosage(int fid, bool alt)
{
    CFileInfo &File = GetFileInfo(fid);
    const int nVar = File.VariantSelNum();
    const int nSamp = File.SampleSelNum();
    npy_intp dims[2] = { nVar, nSamp };
    PyObject *arr = PyArray_SimpleNew(2, dims, NPY_INT32);
    if (nVar > 0) {
        CApply_Variant_Dosage geno(File, false, alt, false);
        int *base = (int *)PyArray_DATA((PyArrayObject *)arr);
        do {
            if (alt) geno.ReadDosageAlt(base); else geno.ReadDosage(base);
            base += nSamp;
        } while (geno.Next());
    }
    return arr;
}

PyObject *PySeq_native_dosage(PyObject *, PyObject *args)
{
    int fid, alt = 0;
    if (!PyArg_ParseTuple(args, "i|i", &fid, &alt)) return NULL;
    return native_guard([&] { return read_dosage(fid, alt != 0); });
}

// Native genotype read: drive the engine's CApply_Variant_Geno over the current
// selection, filling a numpy int32 array (nVarSel, nSampSel, ploidy) directly.
// Missing alleles are NA_INTEGER (-2147483648), matching R seqGetData.  No SEXP.
PyObject *PySeq_native_genotype(PyObject *, PyObject *args)
{
    int fid;
    if (!PyArg_ParseTuple(args, "i", &fid)) return NULL;
    return native_guard([&]() -> PyObject * {
        CFileInfo &File = GetFileInfo(fid);
        const int nVar = File.VariantSelNum();
        const int nSamp = File.SampleSelNum();
        const int ploidy = File.Ploidy();
        npy_intp dims[3] = { nVar, nSamp, ploidy };
        PyObject *arr = PyArray_SimpleNew(3, dims, NPY_INT32);
        if (nVar > 0) {
            CApply_Variant_Geno geno(File, 0 /*use_raw=int*/);
            const ssize_t SIZE = (ssize_t)nSamp * ploidy;
            int *base = (int *)PyArray_DATA((PyArrayObject *)arr);
            do {
                geno.ReadGenoData(base);
                base += SIZE;
            } while (geno.Next());
        }
        return arr;
    });
}
