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

#include <cstring>
#include <map>

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

// ---- selection writes (no SEXP) --------------------------------------------
// Write the engine's TSelection.pSample/pVariant directly, with the same
// invalidation the engine's SEQ_SetSpace* would do: ClearStructSample drops the
// cached genotype-selection (pFlagGenoSel) so genotype reads recompute against the
// new sample set; ClearStructVariant resets varTrueNum=-1 so VariantSelNum recounts.
static void apply_mask(int fid, PyObject *maskobj, bool sample, bool intersect)
{
    CFileInfo &File = GetFileInfo(fid);
    TSelection &Sel = File.Selection();
    int n = sample ? File.SampleNum() : File.VariantNum();
    C_BOOL *p = sample ? Sel.pSample : Sel.pVariant;
    PyArrayObject *a = (PyArrayObject *)PyArray_FROM_OTF(maskobj, NPY_BOOL,
                                                         NPY_ARRAY_IN_ARRAY);
    if (!a) throw Rsh_error("selection must be a 1-D boolean array");
    if ((int)PyArray_SIZE(a) != n) {
        Py_DECREF(a);
        throw Rsh_error("selection length does not match the number of samples/variants");
    }
    const npy_bool *m = (const npy_bool *)PyArray_DATA(a);
    if (sample) Sel.ClearStructSample(); else Sel.ClearStructVariant();
    for (int i = 0; i < n; i++)
        p[i] = intersect ? (p[i] && m[i]) : (m[i] ? 1 : 0);
    Py_DECREF(a);
}

PyObject *PySeq_native_file_init(PyObject *, PyObject *args)
{
    int fid;
    if (!PyArg_ParseTuple(args, "i", &fid)) return NULL;
    return native_guard([&]() -> PyObject * {
        GetFileInfo(fid).Selection();   // force selection to initialize (all TRUE)
        Py_RETURN_NONE;
    });
}

// Drop the engine's cached CFileInfo for this file id (the SEXP-free analog of
// SEQ_File_Done).  MUST be called when closing a file: pygds recycles file-id
// integers, and GetFileInfo(int) skips ResetRoot when a stale entry's Root()
// pointer happens to match a reused root address -> it would otherwise return the
// previous file's variant count and selection.
PyObject *PySeq_native_file_done(PyObject *, PyObject *args)
{
    int fid;
    if (!PyArg_ParseTuple(args, "i", &fid)) return NULL;
    return native_guard([&]() -> PyObject * {
        std::map<int, CFileInfo>::iterator it = GDSFile_ID_Info.find(fid);
        if (it != GDSFile_ID_Info.end())
            GDSFile_ID_Info.erase(it);
        Py_RETURN_NONE;
    });
}

PyObject *PySeq_native_set_sample(PyObject *, PyObject *args)
{
    int fid, intersect = 0; PyObject *mask;
    if (!PyArg_ParseTuple(args, "iO|i", &fid, &mask, &intersect)) return NULL;
    return native_guard([&]() -> PyObject * {
        apply_mask(fid, mask, true, intersect != 0); Py_RETURN_NONE;
    });
}

PyObject *PySeq_native_set_variant(PyObject *, PyObject *args)
{
    int fid, intersect = 0; PyObject *mask;
    if (!PyArg_ParseTuple(args, "iO|i", &fid, &mask, &intersect)) return NULL;
    return native_guard([&]() -> PyObject * {
        apply_mask(fid, mask, false, intersect != 0); Py_RETURN_NONE;
    });
}

PyObject *PySeq_native_reset(PyObject *, PyObject *args)
{
    int fid, sample = 1, variant = 1;
    if (!PyArg_ParseTuple(args, "i|ii", &fid, &sample, &variant)) return NULL;
    return native_guard([&]() -> PyObject * {
        CFileInfo &File = GetFileInfo(fid);
        TSelection &Sel = File.Selection();
        if (sample) { Sel.ClearStructSample(); memset(Sel.pSample, 1, File.SampleNum()); }
        if (variant) { Sel.ClearStructVariant(); memset(Sel.pVariant, 1, File.VariantNum()); }
        Py_RETURN_NONE;
    });
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
