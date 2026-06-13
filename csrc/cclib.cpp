// cclib.cpp — CPython C-API module for PySeqArray.
//
// PySeqArray's entry layer is SEXP-free: every Python-facing function lives in
// csrc/native_api.cpp and drives the SeqArray engine's C++ classes (CFileInfo /
// CApply_Variant_*) straight into numpy.  This file is just the module shell:
// it binds the pygds GDS C-API capsule (Init_GDS_Routines) and initialises the
// engine's shim globals (Rsh_init_globals — the vendored SeqArray .cpp still use
// R_NilValue/NA internally, which is intrinsic to reusing the C++ engine and is
// never exposed at the PySeqArray boundary).

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL PYSEQARRAY_ARRAY_API
#include <numpy/arrayobject.h>

#include <PyGDS2.h>                 // GDS_* capsule wrappers + Init_GDS_Routines
#include "Rshim/Rshim_cpp.h"       // Rsh_init_globals (engine-internal shim)
#include "native_api.h"            // the SEXP-free native entry layer
#include "vcf_api.h"               // VCF import/export (SEXP marshalling lives here)

static PyObject *py_init_gds(PyObject *, PyObject *)
{
    if (Init_GDS_Routines() != 0) return NULL;
    Py_RETURN_TRUE;
}

static PyMethodDef cclib_methods[] = {
    {"init_gds",        py_init_gds,                 METH_NOARGS,  "Bind the pygds GDS C-API capsule."},
    // selection / file
    {"file_init",       PySeq_native_file_init,      METH_VARARGS, "Initialise selection to all (fileid)."},
    {"set_sample",      PySeq_native_set_sample,     METH_VARARGS, "Set sample selection (fileid, bool_mask, intersect=0)."},
    {"set_variant",     PySeq_native_set_variant,    METH_VARARGS, "Set variant selection (fileid, bool_mask, intersect=0)."},
    {"reset_filter",    PySeq_native_reset,          METH_VARARGS, "Reset selection to all (fileid, sample=1, variant=1)."},
    {"get_sample_sel",  PySeq_native_get_sample_sel, METH_VARARGS, "Current sample selection (bool)."},
    {"get_variant_sel", PySeq_native_get_variant_sel,METH_VARARGS, "Current variant selection (bool)."},
    {"n_dims",          PySeq_native_dims,           METH_VARARGS, "(nSamp,nVar,ploidy,nSampSel,nVarSel)."},
    // data
    {"genotype",        PySeq_native_genotype,       METH_VARARGS, "Genotype (nVar,nSamp,ploidy) int32, NA_INTEGER=missing."},
    {"dosage",          PySeq_native_dosage,         METH_VARARGS, "Dosage (nVar,nSamp) int32; arg alt=0 ref/1 alt."},
    // VCF (SEXP marshalling confined here)
    {"sexp_roundtrip",  PySeq_sexp_roundtrip,        METH_VARARGS, "Debug: round-trip a Python value through the shim SEXP marshaller."},
    {"vcf_parse",       PySeq_vcf_parse,             METH_VARARGS, "seqVCF2GDS body parse: (vcf_path, header, out_fileid, param) -> nlines."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cclib_module = {
    PyModuleDef_HEAD_INIT, "cclib",
    "PySeqArray native engine entry layer (SEXP-free) bound to the pygds GDS C-API.",
    -1, cclib_methods, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_cclib(void)
{
    import_array();
    Rsh_init_globals();          // engine-internal shim globals (R_NilValue, NA, ...)
    PyObject *m = PyModule_Create(&cclib_module);
    if (!m) return NULL;
    if (Init_GDS_Routines() != 0) { Py_DECREF(m); return NULL; }
    return m;
}
