// gds_bridge.cpp — implements the R-style GDS bridges (GDS_R_*) that gdsfmt's
// R_GDS.h provides but pygds does not, in terms of pygds's GDS_Py_*/GDS_ID2* C-API
// and the shim SEXP.  This is the only engine-side TU that touches numpy (to
// convert pygds's array results into shim vectors).
//
// Dimension order: pygds returns numpy C-order arrays whose flat buffer equals
// CoreArray's native storage order — identical bytes to what gdsfmt hands R.  R
// (and hence SeqArray's C++) labels the dims reversed (column-major), so we set
// the shim `dim` attribute to the reversed numpy shape; the flat data is copied
// as-is and indexes identically.

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PYSEQARRAY_ARRAY_API
#include <numpy/arrayobject.h>

#include "Rshim/Rshim_cpp.h"
#include "Rshim/R_GDS.h"

// SeqArray.cpp's R_init_SeqArray (unused in PySeqArray — we have our own module
// init) references this SNPRelate hook; stub it so the symbol resolves.  The
// SNPRelate bridge (LinkSNPRelate.cpp) is out of scope for the core engine.
extern "C" void Register_SNPRelate_Functions() {}

// ---- File / object bridges --------------------------------------------------

// The "gdsfile" SEXP is the R gds.class shape: a VECSXP with a "names" attribute
// containing "id" -> ScalarInteger(pygds file id).  (A bare integer is also
// accepted.)  Extract that pygds file id.
static int file_id_of(SEXP File)
{
    if (File && TYPEOF(File) == VECSXP) {
        SEXP names = Rf_getAttrib(File, R_NamesSymbol);
        for (R_xlen_t i = 0; i < Rf_xlength(names); i++)
            if (strcmp(R_CHAR(STRING_ELT(names, i)), "id") == 0)
                return Rf_asInteger(VECTOR_ELT(File, i));
    }
    return Rf_asInteger(File);
}

extern "C" PdGDSFolder GDS_R_SEXP2FileRoot(SEXP File)
{
    PdGDSFolder root = GDS_ID2FileRoot(file_id_of(File));
    if (root == NULL) throw Rsh_error("invalid GDS file id");
    return root;
}

extern "C" PdGDSFile GDS_R_SEXP2File(SEXP File)
{
    PdGDSFile f = GDS_ID2File(file_id_of(File));
    if (f == NULL) throw Rsh_error("invalid GDS file id");
    return f;
}

// A GDS node passed from Python is a boxed EXTPTR holding the PdGDSObj.
extern "C" PdGDSObj GDS_R_SEXP2Obj(SEXP Obj, C_BOOL /*ReadOnly*/)
{
    if (Obj && TYPEOF(Obj) == EXTPTRSXP) return (PdGDSObj)R_ExternalPtrAddr(Obj);
    throw Rsh_error("GDS_R_SEXP2Obj: expected a GDS node external pointer");
}

extern "C" SEXP GDS_R_Obj2SEXP(PdGDSObj Obj)
{
    return R_MakeExternalPtr(Obj, R_NilValue, R_NilValue);
}

extern "C" C_BOOL GDS_R_Is_Logical(PdGDSObj Obj)
{
    return GDS_Is_RLogical(Obj);
}

extern "C" int GDS_R_Set_IfFactor(PdGDSObj Obj, SEXP Val)
{
    // If the node carries R factor metadata, mark the shim vector as a factor by
    // reading its "R.levels" attribute and attaching class/levels.  (B1: detect
    // only; level attachment is wired in B2 alongside annotation/filter tests.)
    return GDS_Is_RFactor(Obj) ? 1 : 0;
}

// ---- numpy -> shim SEXP -----------------------------------------------------

static SEXP numpy_to_sexp(PyObject *a)
{
    if (a == NULL) throw Rsh_error("GDS_Py_Array_Read returned NULL");
    if (a == Py_None) { Py_DECREF(a); return R_NilValue; }
    PyArrayObject *arr = (PyArrayObject*)a;
    int nd = PyArray_NDIM(arr);
    R_xlen_t n = (R_xlen_t)PyArray_SIZE(arr);
    int t = PyArray_TYPE(arr);

    // ensure C-contiguous for a flat copy
    PyArrayObject *c = (PyArrayObject*)PyArray_GETCONTIGUOUS(arr);
    SEXP out;
    switch (t) {
        case NPY_BOOL: {
            out = Rf_allocVector(LGLSXP, n);
            const npy_bool *p = (const npy_bool*)PyArray_DATA(c);
            for (R_xlen_t i = 0; i < n; i++) INTEGER(out)[i] = p[i] ? 1 : 0;
            break;
        }
        case NPY_INT8: case NPY_UINT8: case NPY_INT16: case NPY_UINT16:
        case NPY_INT32: {
            out = Rf_allocVector(INTSXP, n);
            // copy element-by-element widening to int
            for (R_xlen_t i = 0; i < n; i++) {
                void *ip = PyArray_GETPTR1((PyArrayObject*)PyArray_Ravel(c, NPY_CORDER), i);
                (void)ip;
            }
            // faster: use a typed copy via PyArray_CastToType to int32
            { PyArrayObject *ci = (PyArrayObject*)PyArray_Cast(c, NPY_INT32);
              const int32_t *p = (const int32_t*)PyArray_DATA(ci);
              for (R_xlen_t i = 0; i < n; i++) INTEGER(out)[i] = p[i];
              Py_DECREF(ci); }
            break;
        }
        case NPY_INT64: case NPY_UINT64: case NPY_UINT32: {
            out = Rf_allocVector(REALSXP, n);
            PyArrayObject *cd = (PyArrayObject*)PyArray_Cast(c, NPY_FLOAT64);
            const double *p = (const double*)PyArray_DATA(cd);
            for (R_xlen_t i = 0; i < n; i++) REAL(out)[i] = p[i];
            Py_DECREF(cd);
            break;
        }
        case NPY_FLOAT32: case NPY_FLOAT64: {
            out = Rf_allocVector(REALSXP, n);
            PyArrayObject *cd = (PyArrayObject*)PyArray_Cast(c, NPY_FLOAT64);
            const double *p = (const double*)PyArray_DATA(cd);
            for (R_xlen_t i = 0; i < n; i++) REAL(out)[i] = p[i];
            Py_DECREF(cd);
            break;
        }
        default: {
            // object / string arrays -> STRSXP
            out = Rf_allocVector(STRSXP, n);
            PyObject *flat = PyArray_Ravel(c, NPY_CORDER);
            for (R_xlen_t i = 0; i < n; i++) {
                PyObject *o = PySequence_GetItem(flat, (Py_ssize_t)i);
                if (o) {
                    PyObject *s = PyObject_Str(o);
                    const char *cs = s ? PyUnicode_AsUTF8(s) : "";
                    SET_STRING_ELT(out, i, Rf_mkChar(cs ? cs : ""));
                    Py_XDECREF(s); Py_DECREF(o);
                }
            }
            Py_XDECREF(flat);
            break;
        }
    }

    // dim attribute (reverse numpy shape -> R/CoreArray column-major order)
    if (nd > 1) {
        SEXP dim = Rf_allocVector(INTSXP, nd);
        const npy_intp *shp = PyArray_DIMS(arr);
        for (int k = 0; k < nd; k++) INTEGER(dim)[k] = (int)shp[nd - 1 - k];
        Rf_setAttrib(out, R_DimSymbol, dim);
    }
    Py_DECREF(c);
    Py_DECREF(a);
    return out;
}

extern "C" SEXP GDS_R_Array_Read(PdAbstractArray Obj, const C_Int32 *Start,
    const C_Int32 *Length, const C_BOOL *const Selection[], C_UInt32 /*UseMode*/)
{
    enum C_SVType sv = GDS_Array_GetSVType(Obj);
    PyObject *a = GDS_Py_Array_Read(Obj, Start, Length, Selection, sv);
    if (a == NULL) {
        PyErr_Clear();
        throw Rsh_error("GDS_Py_Array_Read failed");
    }
    return numpy_to_sexp(a);
}

// ---- GDS functions absent from pygds (built on GDS_Node_Path/GDS_File_Root) -
extern "C" C_BOOL GDS_Node_Load(PdGDSObj /*Node*/, int NodeID, const char *Path,
    PdGDSFile File, PdGDSObj *OutNode, int *OutNodeID)
{
    PdGDSFolder root = GDS_File_Root(File);
    PdGDSObj obj = GDS_Node_Path(root, Path, 1 /*MustExist*/);
    if (OutNode) *OutNode = obj;
    if (OutNodeID) *OutNodeID = NodeID;   // internal IDs are not tracked here
    return 1;
}
extern "C" void GDS_Node_Unload(PdGDSObj /*Node*/) { /* pygds keeps nodes loaded */ }
extern "C" SEXP GDS_New_SpCMatrix2(SEXP, SEXP, SEXP, int, int)
{ throw Rsh_error("GDS_New_SpCMatrix2: sparse-matrix output not supported in PySeqArray"); }

// ---- write bridges: shim SEXP -> GDS_Array_AppendData -----------------------
// C_SVType values (dType.h): svInt8=5, svInt32=9, svFloat64=14, svStrUTF8=15.
extern "C" void GDS_R_AppendEx(PdAbstractArray Obj, SEXP Val, size_t Start, size_t Count)
{
    switch (TYPEOF(Val)) {
        case INTSXP: case LGLSXP:
            GDS_Array_AppendData(Obj, (ssize_t)Count, INTEGER(Val) + Start,
                                 (enum C_SVType)9 /*svInt32*/);
            break;
        case REALSXP:
            GDS_Array_AppendData(Obj, (ssize_t)Count, REAL(Val) + Start,
                                 (enum C_SVType)14 /*svFloat64*/);
            break;
        case RAWSXP:
            GDS_Array_AppendData(Obj, (ssize_t)Count, RAW(Val) + Start,
                                 (enum C_SVType)5 /*svInt8*/);
            break;
        case STRSXP:
            for (size_t i = 0; i < Count; i++) {
                SEXP e = STRING_ELT(Val, (R_xlen_t)(Start + i));
                GDS_Array_AppendString(Obj, (e && TYPEOF(e) != NILSXP) ? R_CHAR(e) : "");
            }
            break;
        default:
            throw Rsh_error("GDS_R_Append: unsupported SEXP type");
    }
}
extern "C" void GDS_R_Append(PdAbstractArray Obj, SEXP Val)
{
    GDS_R_AppendEx(Obj, Val, 0, (size_t)Rf_xlength(Val));
}
