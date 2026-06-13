// Rshim/R_GDS.h — forwards SeqArray's <R_GDS.h> include to pygds's <PyGDS.h>,
// and declares the R-style GDS bridges (GDS_R_*) that gdsfmt's R_GDS.h provides
// but pygds does not.  These are implemented in csrc/gds_bridge.cpp in terms of
// pygds's GDS_Py_*/GDS_ID2* and the shim SEXP.
#ifndef PYSEQARRAY_RSHIM_R_GDS_H_
#define PYSEQARRAY_RSHIM_R_GDS_H_

// gdsfmt's real R_GDS.h transitively pulls R's headers, so any SeqArray TU that
// includes <R_GDS.h>/<R_GDS_CPP.h> gets the full R macro layer.  pygds's PyGDS.h
// does not, so we pull the shim R headers here to reproduce that.
#include "R.h"            // shim <R.h> -> Rinternals + Rprintf/Calloc
#include "Rdefines.h"     // NEW_INTEGER, SET_DIM, ...
#include <PyGDS.h>        // pygds GDS_* C-API, PdGDS* types, C_Int32/C_SVType (dType.h)

#ifdef __cplusplus
extern "C" {
#endif

// read-mode flags (match gdsfmt R_GDS.h)
#define GDS_R_READ_DEFAULT_MODE      0x00
#define GDS_R_READ_ALLOW_RAW_TYPE    0x01
#define GDS_R_READ_ALLOW_SP_MATRIX   0x10

// SEXP <-> GDS object bridges (PySeqArray/gds_bridge.cpp)
extern PdGDSFile   GDS_R_SEXP2File(SEXP File);
extern PdGDSFolder GDS_R_SEXP2FileRoot(SEXP File);
extern PdGDSObj    GDS_R_SEXP2Obj(SEXP Obj, C_BOOL ReadOnly);
extern SEXP        GDS_R_Obj2SEXP(PdGDSObj Obj);

extern C_BOOL GDS_R_Is_Logical(PdGDSObj Obj);
extern int    GDS_R_Set_IfFactor(PdGDSObj Obj, SEXP Val);

extern SEXP GDS_R_Array_Read(PdAbstractArray Obj, const C_Int32 *Start,
    const C_Int32 *Length, const C_BOOL *const Selection[], C_UInt32 UseMode);

extern void GDS_R_Append(PdAbstractArray Obj, SEXP Val);
extern void GDS_R_AppendEx(PdAbstractArray Obj, SEXP Val, size_t Start, size_t Count);

// GDS C-API functions present in gdsfmt's R_GDS.h but not (yet) in pygds —
// implemented in gds_bridge.cpp on top of pygds's GDS_Node_Path/GDS_File_Root.
extern C_BOOL GDS_Node_Load(PdGDSObj Node, int NodeID, const char *Path,
    PdGDSFile File, PdGDSObj *OutNode, int *OutNodeID);
extern void GDS_Node_Unload(PdGDSObj Node);
extern SEXP GDS_New_SpCMatrix2(SEXP x, SEXP i, SEXP p, int nrow, int ncol);

#ifdef __cplusplus
}
#endif

#endif  // PYSEQARRAY_RSHIM_R_GDS_H_
