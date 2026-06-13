# PySeqArray architecture (C++ engine reuse)

PySeqArray reuses **SeqArray's own C++ engine** (`SeqArray/src/*.cpp`, ~16.5K LoC)
unchanged, swapping only the R `.Call` glue for a CPython C-API module — exactly
the way **pygds** reuses the CoreArray C++ engine of gdsfmt.

```
SeqArray (R)                          PySeqArray (Python)
─────────────                         ───────────────────
R_init_SeqArray + callMethods[]  →    PyInit_cclib + PyMethodDef[]   (cclib.cpp)
SeqArray.cpp / GetData.cpp / ...  →   SAME .cpp, compiled unchanged
  ↓ #include <Rinternals.h>             ↓ #include shim <Rinternals.h>  (Rshim)
  ↓ SEXP / Rf_* / INTEGER() ...         ↓ shim SEXP backed by numpy/PyObject
  ↓ GDS C-API (GDS_Node_Path, ...)      ↓ SAME GDS C-API  ← pygds capsule
R_GDS.h / R_GDS_CPP.h            →     PyGDS.h / PyGDS_CPP.h  (pygds.get_include())
gdsfmt.so                        →     pygds.ccall   (Init_GDS_Routines capsule)
```

## Two pillars

### 1. GDS C-API via the pygds capsule (DONE in pygds, no work)

SeqArray's C++ touches GDS only through gdsfmt's exported C functions
(`GDS_Node_Path`, `GDS_Array_ReadData`/`ReadDataEx`, `GDS_Array_AppendData`,
`GDS_Iter_*`, the buffered `GDS_ArrayRead_*`). pygds re-exports that whole surface:
`pygds.get_include()` ships `PyGDS{,2,_CPP}.h`; `Init_GDS_Routines()` imports the
`pygds.ccall._GDS_C_API` PyCapsule and binds the table. The numpy bridge
`GDS_Py_Array_Read` replaces `GDS_R_Array_Read`.

File-handle bridge: SeqArray's `GetFileInfo(gdsfile)` needs `$id` (int) and the
root folder via `GDS_R_SEXP2FileRoot`. pygds already exposes
`GDS_ID2FileRoot(int file_id)` — so passing pygds's integer `fileid` suffices; no
R object is reconstructed. The shim's `GDS_R_SEXP2FileRoot(sexp)` is just
`GDS_ID2FileRoot(asInteger(sexp))`.

If pygds needs extra `GDS_Py_*` bridges (e.g. `GDS_Py_Append`, an `IfFactor` /
`Is_Logical` helper), they are added directly in `pygds/pygds/include/PyGDS*.h` +
`pygds/src/PyCoreArray.cpp` (user-approved to modify pygds).

### 2. Mini-Rinternals shim (the build work — `Rshim`)

SeqArray's `.cpp` are saturated with `SEXP` (~500 boundary lines). Rather than
rewrite each of the 59 entry points + bodies, we provide a **bounded subset of
R's C API** so the bodies compile verbatim. This keeps SeqArray byte-identical
(correctness + trivial re-sync with upstream SeqArray) and centralizes all the
glue in one place.

`SEXP` = pointer to a small variant cell that holds an int/real/logical/string
vector or a VECSXP list, backed by malloc or by a borrowed numpy array. The shim
implements only what SeqArray uses — surfaced incrementally by the compiler:

- types/alloc: `allocVector`, `allocMatrix`, `Rf_Scalar{Integer,Real,Logical,String}`,
  `mkChar`, `mkString`, `R_NilValue`, `PROTECT/UNPROTECT` (no-op or simple stack)
- accessors: `INTEGER`, `REAL`, `LOGICAL`, `RAW`, `STRING_ELT`, `VECTOR_ELT`,
  `SET_*`, `LENGTH`/`Rf_length`, `XLENGTH`
- coercion/predicate: `Rf_asInteger`, `asReal`, `asLogical`, `isNull`, `isString`,
  `isReal`, `isLogical`, `coerceVector`
- attributes/list: `getAttrib`/`setAttrib`, `install`, names lookup (`RGetListElement`)
- errors: `Rf_error` → set Python exception + longjmp/throw; `Rf_warning`
- NA sentinels: `NA_INTEGER`, `NA_REAL`, `R_NaInt`, `ISNAN`

Boundary conversion (`cclib.cpp`): each Python entry converts PyObject args →
shim SEXP, calls `SEQ_x(...)`, converts the shim SEXP result → numpy/PyObject.
The 59-row `callMethods[]` table maps 1:1 to `PyMethodDef[]` (test_/Progress/
ExternalName helpers can be stubbed/skipped initially).

VCF I/O uses R's connection API (`R_ext/Connections`); replaced with
stdio/zlib/bgzf or Python file objects in B3.

## Milestones (tasks B0–B4)

- **B0** build skeleton: setup.py compiles SeqArray/src + shim against pygds
  includes, `Init_GDS_Routines()` in module init, importable. Spike: compile the
  zero-SEXP `vectorization.cpp` first to validate toolchain + capsule.
- **B1** read path: shim grown enough that SeqArray/GetData/ReadByVariant/Index.cpp
  compile; wire `SEQ_File_Init/SetSpace*/GetSpace*/GetData/Apply_*`. Validate vs R
  oracle (CEU_Exon: non-missing geno sum 32683, missing 17216).
- **B2** Methods.cpp summary (`FC_AF_*`/`FC_AC_*`/`FC_Missing_*`) + FileMerge.cpp.
- **B3** Conv*.cpp VCF/BED + connection IO → stdio/zlib.
- **B4** thin Python API (`seqOpen/seqGetData/...`) over pygds + cclib.

The pure-Python M1 (`_genotype.py` etc.) is kept as a cross-validation oracle /
fallback, superseded as the public engine by cclib.
