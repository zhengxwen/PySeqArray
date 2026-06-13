// native_api.h — declarations for PySeqArray's SEXP-free native entry layer.
#ifndef PYSEQARRAY_NATIVE_API_H_
#define PYSEQARRAY_NATIVE_API_H_

#include <Python.h>

PyObject *PySeq_native_file_init(PyObject *, PyObject *args);
PyObject *PySeq_native_file_done(PyObject *, PyObject *args);
PyObject *PySeq_native_set_sample(PyObject *, PyObject *args);
PyObject *PySeq_native_set_variant(PyObject *, PyObject *args);
PyObject *PySeq_native_reset(PyObject *, PyObject *args);
PyObject *PySeq_native_dims(PyObject *, PyObject *args);
PyObject *PySeq_native_get_sample_sel(PyObject *, PyObject *args);
PyObject *PySeq_native_get_variant_sel(PyObject *, PyObject *args);
PyObject *PySeq_native_genotype(PyObject *, PyObject *args);
PyObject *PySeq_native_dosage(PyObject *, PyObject *args);

#endif  // PYSEQARRAY_NATIVE_API_H_
