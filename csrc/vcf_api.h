// vcf_api.h — VCF import/export entry points (the SEXP-marshalling lives here).
#ifndef PYSEQARRAY_VCF_API_H_
#define PYSEQARRAY_VCF_API_H_

#include <Python.h>

PyObject *PySeq_sexp_roundtrip(PyObject *, PyObject *args);  // marshaller self-test
PyObject *PySeq_vcf_parse(PyObject *, PyObject *args);       // seqVCF2GDS body parse

#endif  // PYSEQARRAY_VCF_API_H_
