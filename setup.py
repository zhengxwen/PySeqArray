import os
import sys

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy
import pygds


class build_ext_cppstd(build_ext):
    """Apply -std=c++11 to C++ translation units only (CoreArray ships C too)."""
    CXX_STD = '-std=c++11'

    def build_extension(self, ext):
        orig = self.compiler._compile

        def _compile(obj, src, ext_, cc_args, extra_postargs, pp_opts):
            postargs = list(extra_postargs)
            if src.endswith(('.cpp', '.cxx', '.cc', '.C')):
                postargs = [self.CXX_STD] + postargs
            return orig(obj, src, ext_, cc_args, postargs, pp_opts)

        self.compiler._compile = _compile
        try:
            super().build_extension(ext)
        finally:
            self.compiler._compile = orig


SEQ_SRC = os.environ.get('SEQARRAY_SRC', os.path.join('..', 'SeqArray', 'src'))

# B1 read path: SeqArray engine TUs that the Rinternals shim must support.
SEQARRAY_TUS = [os.path.join(SEQ_SRC, fn) for fn in [
    'vectorization.cpp',
    'Index.cpp',
    'ReadByVariant.cpp',
    'ReadBySample.cpp',
    'ReadByUnit.cpp',
    'GetData.cpp',
    'SeqArray.cpp',
    # B3: conversion + merge
    'ConvToGDS.cpp',
    'ConvVCF2GDS.cpp',
    'ConvGDS2VCF.cpp',
    'FileMerge.cpp',
]]

SHIM_TUS = ['csrc/cclib.cpp', 'csrc/Rshim.cpp', 'csrc/gds_bridge.cpp', 'csrc/conn.cpp']

ext = Extension(
    'PySeqArray.cclib',
    SHIM_TUS + SEQARRAY_TUS,
    include_dirs=[
        'csrc/Rshim',            # mini-Rinternals shim (takes priority)
        SEQ_SRC,                 # Index.h, vectorization.h, ...
        pygds.get_include(),     # PyGDS*.h + CoreDEF.h + dType/dTrait.h
        numpy.get_include(),
    ],
    define_macros=[
        ('USING_PYTHON', None),
        ('_FILE_OFFSET_BITS', 64),
        ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
    ],
    libraries=['z'],            # system zlib for the gzFile connection layer
    extra_compile_args=['-O2'],
)

setup(ext_modules=[ext], cmdclass={'build_ext': build_ext_cppstd})
