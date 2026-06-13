// native_api.h — declarations for PySeqArray's SEXP-free native entry layer.
#ifndef PYSEQARRAY_NATIVE_API_H_
#define PYSEQARRAY_NATIVE_API_H_

#include <Python.h>

PyObject *PySeq_native_dims(PyObject *, PyObject *args);
PyObject *PySeq_native_get_sample_sel(PyObject *, PyObject *args);
PyObject *PySeq_native_get_variant_sel(PyObject *, PyObject *args);

#endif  // PYSEQARRAY_NATIVE_API_H_
