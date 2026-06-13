// ===========================================================
//
// ReadByVariant.cpp: Read data variant by variant
//
// Copyright (C) 2013-2026    Xiuwen Zheng
//
// This file is part of PySeqArray.
//
// PySeqArray is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License Version 3 as
// published by the Free Software Foundation.
//
// PySeqArray is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with PySeqArray.
// If not, see <http://www.gnu.org/licenses/>.

// Native (Python/NumPy) port of SeqArray's src/ReadByVariant.cpp. The R
// SEXP allocation/access (NEW_*/INTEGER/REAL/SET_STRING_ELT/PROTECT) becomes
// NumPy arrays; NeedRData(int&)->NeedArray(); the per-variant apply calls a
// Python callable via PyObject_CallObject instead of R's Rf_eval.

#include "ReadByVariant.h"
#include <numpy/arrayobject.h>

#ifndef NUMPY_IMPORT_ARRAY_RETVAL
#  define NUMPY_IMPORT_ARRAY_RETVAL  NULL
#endif
#define PyArray_DATA(o)        PyArray_DATA((PyArrayObject*)(o))
#define PyArray_TYPE(o)        PyArray_TYPE((PyArrayObject*)(o))


namespace PySeqArray
{

using namespace Vectorization;

static const char *ERR_DIM = "Invalid dimension of '%s'.";

// variable list
static const string VAR_POSITION("position");
static const string VAR_CHROM("chromosome");
static const string VAR_ID("variant.id");
static const string VAR_ALLELE("allele");
static const string VAR_ANNOT_ID("annotation/id");
static const string VAR_ANNOT_QUAL("annotation/qual");
static const string VAR_ANNOT_FILTER("annotation/filter");
static const string VAR_GENOTYPE("genotype");
static const string VAR_PHASE("phase");

// variable list: internally generated
static const string VAR_DOSAGE("$dosage");
static const string VAR_DOSAGE_ALT("$dosage_alt");
static const string VAR_DOSAGE_ALT2("$dosage_alt2");
static const string VAR_NUM_ALLELE("$num_allele");
static const string VAR_REF_ALLELE("$ref");
static const string VAR_ALT_ALLELE("$alt");
static const string VAR_CHROM_POS("$chrom_pos");
static const string VAR_CHROM_POS_ALLELE("$chrom_pos_allele");
static const string VAR_VARIANT_INDEX("$variant_index");


// ---------------------------------------------------------------------------
// helpers to get/set a string element of a NumPy object array (SET_STRING_ELT)
static inline void set_str(PyObject *val, size_t i, const char *s)
{
	PyObject **p = (PyObject**)PyArray_DATA(val);
	PyObject *o = PyUnicode_FromString(s);
	PyObject *old = p[i];
	p[i] = o;
	Py_XDECREF(old);
}
static inline void set_strn(PyObject *val, size_t i, const char *s, size_t n)
{
	PyObject **p = (PyObject**)PyArray_DATA(val);
	PyObject *o = PyUnicode_FromStringAndSize(s, n);
	PyObject *old = p[i];
	p[i] = o;
	Py_XDECREF(old);
}

// allocate a NumPy array (1- or 2-D) given by the SVType of a GDS node
static PyObject *gds_alloc(PdAbstractArray Node, const npy_intp *dims, int nd,
	bool bit1_logical)
{
	C_SVType sv = GDS_Array_GetSVType(Node);
	int npt;
	if (COREARRAY_SV_FLOAT(sv))
		npt = NPY_FLOAT64;
	else if (COREARRAY_SV_STRING(sv))
		npt = NPY_OBJECT;
	else {  // integer
		char cn[128] = { 0 };
		GDS_Node_GetClassName(Node, cn, sizeof(cn));
		if ((strcmp(cn, "dBit1")==0) && bit1_logical)
			npt = NPY_BOOL;
		else if (GDS_Is_RLogical(Node))
			npt = NPY_BOOL;
		else
			npt = NPY_INT32;
	}
	PyObject *rv = PyArray_SimpleNew(nd, (npy_intp*)dims, npt);
	if (!rv) throw ErrSeqArray("Fails to allocate a NumPy array.");
	return rv;
}


// =====================================================================
// Object for reading basic variables variant by variant

CApply_Variant_Basic::CApply_Variant_Basic(CFileInfo &File,
	const char *var_name): CApply_Variant(File)
{
	fVarType = ctBasic;
	Node = File.GetObj(var_name, TRUE);
	SVType = GDS_Array_GetSVType(Node);
	VarNode = NULL;
	Reset();
}

void CApply_Variant_Basic::ReadData(PyObject *val)
{
	C_Int32 st = Position, one = 1;
	if (COREARRAY_SV_INTEGER(SVType))
	{
		GDS_Array_ReadData(Node, &st, &one, (int*)PyArray_DATA(val), svInt32);
	} else if (COREARRAY_SV_FLOAT(SVType))
	{
		GDS_Array_ReadData(Node, &st, &one, (double*)PyArray_DATA(val), svFloat64);
	} else if (COREARRAY_SV_STRING(SVType))
	{
		string s;
		GDS_Array_ReadData(Node, &st, &one, &s, svStrUTF8);
		set_str(val, 0, s.c_str());
	}
}

PyObject* CApply_Variant_Basic::NeedArray()
{
	if (VarNode == NULL)
	{
		npy_intp d[1] = { 1 };
		VarNode = gds_alloc(Node, d, 1, false);
	}
	return VarNode;
}

// ====

CApply_Variant_Pos::CApply_Variant_Pos(CFileInfo &File):
	CApply_Variant(File)
{
	fVarType = ctBasic;
	Node = File.GetObj("position", TRUE);
	PtrPos = &File.Position()[0];
	VarNode = NULL;
	Reset();
}

void CApply_Variant_Pos::ReadData(PyObject *val)
{
	((int*)PyArray_DATA(val))[0] = PtrPos[Position];
}

PyObject* CApply_Variant_Pos::NeedArray()
{
	if (VarNode == NULL) VarNode = numpy_new_int32(1);
	return VarNode;
}

// ====

CApply_Variant_Chrom::CApply_Variant_Chrom(CFileInfo &File):
	CApply_Variant(File)
{
	fVarType = ctBasic;
	Node = File.GetObj("chromosome", TRUE);
	ChromIndex = &File.Chromosome();
	VarNode = NULL;
	Reset();
}

void CApply_Variant_Chrom::ReadData(PyObject *val)
{
	const string &s1 = (*ChromIndex)[Position];
	set_str(val, 0, s1.c_str());
}

PyObject* CApply_Variant_Chrom::NeedArray()
{
	if (VarNode == NULL)
	{
		VarNode = numpy_new_string(1);
		set_str(VarNode, 0, "");
	}
	return VarNode;
}



// =====================================================================
// Object for reading genotypes variant by variant

static inline void read_geno(CdIterator &it, int *out, TSelection::TSampStruct *p)
{
	C_Int64 base = it.Ptr;
	for (; p->length > 0; p++)
	{
		it.Ptr = base + p->offset;
		if (!p->sel)
			out = (int*)GDS_Iter_RData(&it, out, p->length, svInt32);
		else
			out = (int*)GDS_Iter_RDataEx(&it, out, p->length, svInt32, p->sel);
	}
}

static inline void read_geno(CdIterator &it, C_UInt8 *out, TSelection::TSampStruct *p)
{
	C_Int64 base = it.Ptr;
	for (; p->length > 0; p++)
	{
		it.Ptr = base + p->offset;
		if (!p->sel)
			out = (C_UInt8*)GDS_Iter_RData(&it, out, p->length, svUInt8);
		else
			out = (C_UInt8*)GDS_Iter_RDataEx(&it, out, p->length, svUInt8, p->sel);
	}
}


CApply_Variant_Geno::CApply_Variant_Geno(): CApply_Variant()
{
	fVarType = ctGenotype;
	SiteCount = CellCount = 0;
	SampNum = 0; Ploidy = 0;
	UseRaw = FALSE;
	pSampSel = NULL;
	VarIntGeno = VarRawGeno = NULL;
}

CApply_Variant_Geno::CApply_Variant_Geno(CFileInfo &File, int use_raw):
	CApply_Variant()
{
	fVarType = ctGenotype;
	Init(File, use_raw);
}

void CApply_Variant_Geno::Init(CFileInfo &File, int use_raw)
{
	static const char *VAR_NAME = "genotype/data";

	// initialize
	Node = File.GetObj(VAR_NAME, TRUE);

	// check
	if (GDS_Array_DimCnt(Node) != 3)
		throw ErrSeqArray(ERR_DIM, VAR_NAME);
	C_Int32 DLen[3];
	GDS_Array_GetDim(Node, DLen, 3);
	if ((DLen[0] < File.VariantNum()) || (DLen[1] != File.SampleNum()))
		throw ErrSeqArray(ERR_DIM, VAR_NAME);

	// initialize
	InitMarginal(File);
	GenoIndex = &File.GenoIndex();
	SiteCount = ssize_t(DLen[1]) * DLen[2];
	SampNum = File.SampleSelNum();
	CellCount = SampNum * DLen[2];
	Ploidy = File.Ploidy();
	UseRaw = use_raw;

	// initialize selection
	pSampSel = File.Selection().GetStructSample();

	ExtPtr.reset(SiteCount);
	VarIntGeno = VarRawGeno = NULL;
	Reset();
}

int CApply_Variant_Geno::_ReadGenoData(int *Base)
{
	C_UInt8 NumIndexRaw;
	C_Int64 Index;
	GenoIndex->GetInfo(Position, Index, NumIndexRaw);

	if (NumIndexRaw >= 1)
	{
		CdIterator it;
		GDS_Iter_Position(Node, &it, Index*SiteCount);
		read_geno(it, Base, pSampSel);

		const int bit_mask = 0x03;
		int missing = bit_mask;
		for (C_UInt8 i=1; i < NumIndexRaw; i++)
		{
			GDS_Iter_Position(Node, &it, (Index+i)*SiteCount);
			C_UInt8 *buf = (C_UInt8*)ExtPtr.get();
			read_geno(it, buf, pSampSel);
			vec_i32_or_shl2(Base, CellCount, buf, i*2); // shift left = i*2
			missing = (missing << 2) | bit_mask;
		}

		return missing;
	} else {
		memset(Base, 0, sizeof(int)*CellCount);
		return 0;
	}
}

C_UInt8 CApply_Variant_Geno::_ReadGenoData(C_UInt8 *Base)
{
	C_UInt8 NumIndexRaw;
	C_Int64 Index;
	GenoIndex->GetInfo(Position, Index, NumIndexRaw);

	if (NumIndexRaw >= 1)
	{
		CdIterator it;
		GDS_Iter_Position(Node, &it, Index*SiteCount);
		read_geno(it, Base, pSampSel);

		const C_UInt8 bit_mask = 0x03;
		C_UInt8 missing = bit_mask;
		if (NumIndexRaw > 4) NumIndexRaw = 4;

		for (C_UInt8 i=1; i < NumIndexRaw; i++)
		{
			GDS_Iter_Position(Node, &it, (Index+i)*SiteCount);
			C_UInt8 *buf = (C_UInt8*)ExtPtr.get();
			read_geno(it, buf, pSampSel);
			vec_u8_or_shl(Base, CellCount, buf, i*2); // shift left = i*2
			missing = (missing << 2) | bit_mask;
		}

		return missing;
	} else {
		memset(Base, 0, CellCount);
		return 0;
	}
}

void CApply_Variant_Geno::ReadData(PyObject *val)
{
	if (numpy_is_uint8(val))
		ReadGenoData((C_UInt8*)PyArray_DATA(val));
	else
		ReadGenoData((int*)PyArray_DATA(val));
}

PyObject* CApply_Variant_Geno::NeedArray()
{
	bool int_type;
	if (UseRaw == NA_INTEGER)
	{
		C_UInt8 NumIndexRaw;
		C_Int64 Index;
		GenoIndex->GetInfo(Position, Index, NumIndexRaw);
		int_type = (NumIndexRaw > 4);
	} else if (UseRaw == FALSE)
		int_type = true;
	else
		int_type = false;

	if (int_type)
	{
		if (VarIntGeno == NULL)
			VarIntGeno = numpy_new_int32_mat(SampNum, Ploidy);
		return VarIntGeno;
	} else {
		if (VarRawGeno == NULL)
			VarRawGeno = numpy_new_uint8_mat(SampNum, Ploidy);
		return VarRawGeno;
	}
}

bool CApply_Variant_Geno::NeedIntType()
{
	const C_BOOL *b = MarginalSelect;
	C_Int32 P = Position;
	while (P < MarginalEnd)
	{
		C_UInt8 NumIndexRaw;
		C_Int64 Index;
		GenoIndex->GetInfo(P, Index, NumIndexRaw);
		if (NumIndexRaw > 4) return true;
		// find next
		P = VEC_BOOL_FIND_TRUE(b+P+1, b+MarginalEnd) - b;
	}
	return false;
}

void CApply_Variant_Geno::ReadGenoData(int *Base)
{
	int missing = _ReadGenoData(Base);
	vec_i32_replace(Base, CellCount, missing, NA_INTEGER);
}

void CApply_Variant_Geno::ReadGenoData(C_UInt8 *Base)
{
	C_UInt8 missing = _ReadGenoData(Base);
	vec_i8_replace((C_Int8*)Base, CellCount, missing, (C_Int8)NA_RAW);
}



// =====================================================================
// Object for reading genotypes (dosages) variant by variant

CApply_Variant_Dosage::CApply_Variant_Dosage(CFileInfo &File, int use_raw,
	bool alt, bool alt2): CApply_Variant_Geno(File, use_raw)
{
	fVarType = ctDosage;
	IsAlt = alt; IsAlt2 = alt2;
	ExtPtr2.reset(sizeof(int)*CellCount);
	VarDosage = NULL;
}

void CApply_Variant_Dosage::ReadData(PyObject *val)
{
	if (numpy_is_uint8(val))
	{
		C_UInt8 *p = (C_UInt8*)PyArray_DATA(val);
		if (IsAlt) { if (IsAlt2) ReadDosageAlt_p(p); else ReadDosageAlt(p); }
		else ReadDosage(p);
	} else {
		int *p = (int*)PyArray_DATA(val);
		if (IsAlt) { if (IsAlt2) ReadDosageAlt_p(p); else ReadDosageAlt(p); }
		else ReadDosage(p);
	}
}

PyObject* CApply_Variant_Dosage::NeedArray()
{
	if (VarDosage == NULL)
		VarDosage = UseRaw ? numpy_new_uint8(SampNum) : numpy_new_int32(SampNum);
	return VarDosage;
}

void CApply_Variant_Dosage::ReadDosage(int *Base)
{
	int *p = (int *)ExtPtr2.get();
	int missing = _ReadGenoData(p);
	if (Ploidy == 2)
		vec_i32_cnt_dosage2(p, Base, SampNum, 0, missing, NA_INTEGER);
	else {
		for (int n=SampNum; n > 0; n--)
		{
			int cnt = 0;
			for (int m=Ploidy; m > 0; m--, p++)
			{
				if (*p == 0) { if (cnt != NA_INTEGER) cnt ++; }
				else if (*p == missing) cnt = NA_INTEGER;
			}
			*Base ++ = cnt;
		}
	}
}

void CApply_Variant_Dosage::ReadDosageAlt(int *Base)
{
	int *p = (int *)ExtPtr2.get();
	int missing = _ReadGenoData(p);
	if (Ploidy == 2)
		vec_i32_cnt_dosage_alt2(p, Base, SampNum, 0, missing, NA_INTEGER);
	else {
		for (int n=SampNum; n > 0; n--)
		{
			int cnt = 0;
			for (int m=Ploidy; m > 0; m--, p++)
			{
				if (*p == missing) cnt = NA_INTEGER;
				else if (*p != 0) { if (cnt != NA_INTEGER) cnt ++; }
			}
			*Base ++ = cnt;
		}
	}
}

void CApply_Variant_Dosage::ReadDosageAlt_p(int *Base)
{
	int *p = (int *)ExtPtr2.get();
	int missing = _ReadGenoData(p);
	if (Ploidy == 2)
		vec_i32_cnt_dosage_alt2_p(p, Base, SampNum, 0, missing, NA_INTEGER);
	else {
		for (int n=SampNum; n > 0; n--)
		{
			int cnt = 0, non_miss = Ploidy;
			for (int m=Ploidy; m > 0; m--, p++)
			{
				if (*p == missing) non_miss --;
				else if (*p != 0) cnt ++;
			}
			*Base ++ = (non_miss > 0) ? cnt : NA_INTEGER;
		}
	}
}

void CApply_Variant_Dosage::ReadDosage(C_UInt8 *Base)
{
	C_UInt8 *p = (C_UInt8 *)ExtPtr2.get();
	C_UInt8 missing = _ReadGenoData(p);
	if (Ploidy == 2)
		vec_i8_cnt_dosage2((int8_t *)p, (int8_t *)Base, SampNum, 0,
			missing, (C_Int8)NA_RAW);
	else {
		for (int n=SampNum; n > 0; n--)
		{
			C_UInt8 cnt = 0;
			for (int m=Ploidy; m > 0; m--, p++)
			{
				if (*p == 0) { if (cnt != NA_RAW) cnt ++; }
				else if (*p == missing) cnt = NA_RAW;
			}
			*Base ++ = cnt;
		}
	}
}

void CApply_Variant_Dosage::ReadDosageAlt(C_UInt8 *Base)
{
	C_UInt8 *p = (C_UInt8 *)ExtPtr2.get();
	C_UInt8 missing = _ReadGenoData(p);
	if (Ploidy == 2)
		vec_i8_cnt_dosage_alt2((int8_t *)p, (int8_t *)Base, SampNum, 0,
			missing, (C_Int8)NA_RAW);
	else {
		for (int n=SampNum; n > 0; n--)
		{
			C_UInt8 cnt = 0;
			for (int m=Ploidy; m > 0; m--, p++)
			{
				if (*p == missing) cnt = NA_RAW;
				else if (*p != 0) { if (cnt != NA_RAW) cnt ++; }
			}
			*Base ++ = cnt;
		}
	}
}

void CApply_Variant_Dosage::ReadDosageAlt_p(C_UInt8 *Base)
{
	C_UInt8 *p = (C_UInt8 *)ExtPtr2.get();
	C_UInt8 missing = _ReadGenoData(p);
	if (Ploidy == 2)
		vec_i8_cnt_dosage_alt2_p((int8_t *)p, (int8_t *)Base, SampNum, 0,
			missing, (C_Int8)NA_RAW);
	else {
		for (int n=SampNum; n > 0; n--)
		{
			C_UInt8 cnt = 0, non_miss = Ploidy;
			for (int m=Ploidy; m > 0; m--, p++)
			{
				if (*p == missing) non_miss --;
				else if (*p != 0) cnt ++;
			}
			*Base ++ = (non_miss > 0) ? cnt : NA_RAW;
		}
	}
}



// =====================================================================
// Object for reading phasing information variant by variant

CApply_Variant_Phase::CApply_Variant_Phase():
	CApply_Variant()
{
	fVarType = ctPhase;
	SiteCount = CellCount = 0;
	SampNum = 0; Ploidy = 0;
	UseRaw = FALSE;
	VarPhase = NULL;
}

CApply_Variant_Phase::CApply_Variant_Phase(CFileInfo &File, bool use_raw):
	CApply_Variant()
{
	fVarType = ctPhase;
	Init(File, use_raw);
}

void CApply_Variant_Phase::Init(CFileInfo &File, bool use_raw)
{
	static const char *VAR_NAME = "phase/data";

	// initialize
	Node = File.GetObj(VAR_NAME, TRUE);

	// check
	int DimCnt = GDS_Array_DimCnt(Node);
	if ((DimCnt != 2) && (DimCnt != 3))
		throw ErrSeqArray(ERR_DIM, VAR_NAME);
	C_Int32 DLen[3] = { 0, 0, 1 };
	GDS_Array_GetDim(Node, DLen, 3);
	if ((DLen[0] != File.VariantNum()) || (DLen[1] != File.SampleNum()))
		throw ErrSeqArray(ERR_DIM, VAR_NAME);

	// initialize
	InitMarginal(File);
	SiteCount = ssize_t(DLen[1]) * DLen[2];
	SampNum = File.SampleSelNum();
	CellCount = SampNum * DLen[2];
	Ploidy = File.Ploidy();
	UseRaw = use_raw;

	// initialize selection
	Selection.resize(SiteCount);
	C_BOOL *p = &Selection[0];
	memset(p, TRUE, SiteCount);
	C_BOOL *s = File.Selection().pSample;
	for (int n=DLen[1]; n > 0; n--)
	{
		if (*s++ == FALSE)
		{
			for (int m=DLen[2]; m > 0; m--) *p ++ = FALSE;
		} else {
			p += DLen[2];
		}
	}

	VarPhase = NULL;
	Reset();
}

void CApply_Variant_Phase::ReadData(PyObject *val)
{
	CdIterator it;
	GDS_Iter_Position(Node, &it, ssize_t(Position)*SiteCount);
	if (UseRaw)
		GDS_Iter_RDataEx(&it, (C_UInt8*)PyArray_DATA(val), SiteCount, svInt8, &Selection[0]);
	else
		GDS_Iter_RDataEx(&it, (int*)PyArray_DATA(val), SiteCount, svInt32, &Selection[0]);
}

PyObject* CApply_Variant_Phase::NeedArray()
{
	if (VarPhase == NULL)
	{
		if (Ploidy > 2)
			VarPhase = UseRaw ?
				numpy_new_uint8_mat(SampNum, Ploidy-1) :
				numpy_new_int32_mat(SampNum, Ploidy-1);
		else
			VarPhase = UseRaw ? numpy_new_uint8(CellCount) :
				numpy_new_int32(CellCount);
	}
	return VarPhase;
}



// =====================================================================
// Object for reading info variables variant by variant

CApply_Variant_Info::CApply_Variant_Info(CFileInfo &File,
	const char *var_name): CApply_Variant(File)
{
	// initialize
	fVarType = ctInfo;
	Node = File.GetObj(var_name, TRUE);

	// check
	int DimCnt = GDS_Array_DimCnt(Node);
	if ((DimCnt != 1) && (DimCnt != 2))
		throw ErrSeqArray(ERR_DIM, var_name);

	// initialize
	C_Int32 DLen[2];
	GDS_Array_GetDim(Node, DLen, 2);
	BaseNum = (DimCnt == 2) ? DLen[1] : 1;
	VarIndex = &VarGetStruct(File, var_name).Index;
	SVType = GDS_Array_GetSVType(Node);

	Reset();
}

void CApply_Variant_Info::ReadData(PyObject *val)
{
	C_Int64 IndexRaw;
	int NumIndexRaw;
	VarIndex->GetInfo(Position, IndexRaw, NumIndexRaw);

	if (NumIndexRaw > 0)
	{
		C_Int32 st[2]  = { (C_Int32)IndexRaw, 0 };
		C_Int32 cnt[2] = { NumIndexRaw, BaseNum };

		if (COREARRAY_SV_INTEGER(SVType))
		{
			GDS_Array_ReadData(Node, st, cnt, (int*)PyArray_DATA(val), svInt32);
		} else if (COREARRAY_SV_FLOAT(SVType))
		{
			GDS_Array_ReadData(Node, st, cnt, (double*)PyArray_DATA(val), svFloat64);
		} else if (COREARRAY_SV_STRING(SVType))
		{
			size_t N = numpy_size(val);
			vector<string> buffer(N);
			GDS_Array_ReadData(Node, st, cnt, &buffer[0], svStrUTF8);
			for (size_t i=0; i < N; i++)
				set_str(val, i, buffer[i].c_str());
		}
	}
}

PyObject* CApply_Variant_Info::NeedArray()
{
	C_Int64 IndexRaw;
	int NumIndexRaw;
	VarIndex->GetInfo(Position, IndexRaw, NumIndexRaw);
	if (NumIndexRaw <= 0) return Py_None;

	map<int, PyObject*>::iterator it = VarList.find(NumIndexRaw);
	if (it == VarList.end())
	{
		PyObject *ans;
		if (BaseNum > 1)
		{
			npy_intp d[2] = { NumIndexRaw, BaseNum };
			ans = gds_alloc(Node, d, 2, true);
		} else {
			npy_intp d[1] = { NumIndexRaw };
			ans = gds_alloc(Node, d, 1, true);
		}
		VarList.insert(pair<int, PyObject*>(NumIndexRaw, ans));
		return ans;
	} else
		return it->second;
}



// =====================================================================
// Object for reading format variables variant by variant

CApply_Variant_Format::CApply_Variant_Format(): CApply_Variant()
{
	fVarType = ctFormat;
}

CApply_Variant_Format::CApply_Variant_Format(CFileInfo &File,
	const char *var_name): CApply_Variant()
{
	fVarType = ctFormat;
	Init(File, var_name);
}

void CApply_Variant_Format::Init(CFileInfo &File, const char *var_name)
{
	// initialize
	Node = File.GetObj((string(var_name)+"/data").c_str(), TRUE);

	// check
	int DimCnt = GDS_Array_DimCnt(Node);
	if (DimCnt != 2)
		throw ErrSeqArray(ERR_DIM, var_name);
	C_Int32 DLen[2];
	GDS_Array_GetDim(Node, DLen, 2);
	if (DLen[1] != File.SampleNum())
		throw ErrSeqArray(ERR_DIM, var_name);

	// initialize
	InitMarginal(File);
	SVType = GDS_Array_GetSVType(Node);
	VarIndex = &VarGetStruct(File, var_name).Index;
	SampNum = File.SampleSelNum();
	_TotalSampNum = File.SampleNum();

	// initialize selection
	SelPtr[0] = NULL;
	SelPtr[1] = File.Selection().pSample;

	Reset();
}

void CApply_Variant_Format::ReadData(PyObject *val)
{
	C_Int64 IndexRaw;
	int NumIndexRaw;
	VarIndex->GetInfo(Position, IndexRaw, NumIndexRaw);

	if (NumIndexRaw > 0)
	{
		C_Int32 st[2]  = { (C_Int32)IndexRaw, 0 };
		C_Int32 cnt[2] = { NumIndexRaw, (C_Int32)_TotalSampNum };
		SelPtr[0] = NeedTRUEs(NumIndexRaw);

		if (COREARRAY_SV_INTEGER(SVType))
		{
			GDS_Array_ReadDataEx(Node, st, cnt, SelPtr, (int*)PyArray_DATA(val), svInt32);
		} else if (COREARRAY_SV_FLOAT(SVType))
		{
			GDS_Array_ReadDataEx(Node, st, cnt, SelPtr, (double*)PyArray_DATA(val), svFloat64);
		} else if (COREARRAY_SV_STRING(SVType))
		{
			size_t N = numpy_size(val);
			vector<string> buffer(N);
			GDS_Array_ReadDataEx(Node, st, cnt, SelPtr, &buffer[0], svStrUTF8);
			for (size_t i=0; i < N; i++)
				set_str(val, i, buffer[i].c_str());
		}
	}
}

PyObject* CApply_Variant_Format::NeedArray()
{
	C_Int64 IndexRaw;
	int NumIndexRaw;
	VarIndex->GetInfo(Position, IndexRaw, NumIndexRaw);
	if (NumIndexRaw <= 0) return Py_None;

	map<int, PyObject*>::iterator it = VarList.find(NumIndexRaw);
	if (it == VarList.end())
	{
		npy_intp d[2] = { NumIndexRaw, (npy_intp)SampNum };
		PyObject *ans = gds_alloc(Node, d, 2, false);
		VarList.insert(pair<int, PyObject*>(NumIndexRaw, ans));
		return ans;
	} else
		return it->second;
}



// =====================================================================
// Object for the number of distinct alleles variant by variant

CApply_Variant_NumAllele::CApply_Variant_NumAllele(CFileInfo &File):
	CApply_Variant(File)
{
	strbuf.reserve(128);
	fVarType = ctBasic;
	Node = File.GetObj("allele", TRUE);
	VarNode = NULL;
	Reset();
}

void CApply_Variant_NumAllele::ReadData(PyObject *val)
{
	((int*)PyArray_DATA(val))[0] = GetNumAllele();
}

PyObject* CApply_Variant_NumAllele::NeedArray()
{
	if (VarNode == NULL) VarNode = numpy_new_int32(1);
	return VarNode;
}

int CApply_Variant_NumAllele::GetNumAllele()
{
	C_Int32 st = Position, one = 1;
	GDS_Array_ReadData(Node, &st, &one, &strbuf, svStrUTF8);
	return GetNumOfAllele(strbuf.c_str());
}


// =====================================================================
// Object for reading the reference allele variant by variant

CApply_Variant_RefAllele::CApply_Variant_RefAllele(CFileInfo &File):
	CApply_Variant(File)
{
	strbuf.reserve(128);
	fVarType = ctBasic;
	Node = File.GetObj("allele", TRUE);
	VarNode = NULL;
	Reset();
}

void CApply_Variant_RefAllele::ReadData(PyObject *val)
{
	C_Int32 st = Position, one = 1;
	GDS_Array_ReadData(Node, &st, &one, &strbuf, svStrUTF8);
	const char *p = strbuf.c_str();
	size_t m = 0;
	for (const char *s=p; *s!=',' && *s!=0; s++) m++;
	set_strn(val, 0, p, m);
}

PyObject* CApply_Variant_RefAllele::NeedArray()
{
	if (VarNode == NULL) VarNode = numpy_new_string(1);
	return VarNode;
}


// =====================================================================
// Object for reading alternative allele(s) variant by variant

CApply_Variant_AltAllele::CApply_Variant_AltAllele(CFileInfo &File):
	CApply_Variant(File)
{
	strbuf.reserve(128);
	fVarType = ctBasic;
	Node = File.GetObj("allele", TRUE);
	VarNode = NULL;
	Reset();
}

void CApply_Variant_AltAllele::ReadData(PyObject *val)
{
	C_Int32 st = Position, one = 1;
	GDS_Array_ReadData(Node, &st, &one, &strbuf, svStrUTF8);
	const char *p = strbuf.c_str();
	for (; *p!=',' && *p!=0;) p++;
	if (*p == ',') p++;
	set_str(val, 0, p);
}

PyObject* CApply_Variant_AltAllele::NeedArray()
{
	if (VarNode == NULL) VarNode = numpy_new_string(1);
	return VarNode;
}


// =====================================================================
// Object for reading chromosome:position variant by variant

CApply_Variant_ChromPos::CApply_Variant_ChromPos(CFileInfo &File):
	CApply_Variant(File)
{
	fVarType = ctBasic;
	Node = File.GetObj("chromosome", TRUE);
	ChromIndex = &File.Chromosome();
	PtrPos = &File.Position()[0];
	VarNode = NULL;
	Reset();
}

void CApply_Variant_ChromPos::ReadData(PyObject *val)
{
	char buf[1024];
	snprintf(buf, sizeof(buf), "%s:%d",
		(*ChromIndex)[Position].c_str(), PtrPos[Position]);
	set_str(val, 0, buf);
}

PyObject* CApply_Variant_ChromPos::NeedArray()
{
	if (VarNode == NULL) VarNode = numpy_new_string(1);
	return VarNode;
}


// =====================================================================
// Object for reading chromosome:position_allele variant by variant

CApply_Variant_ChromPosAllele::CApply_Variant_ChromPosAllele(CFileInfo &File):
	CApply_Variant(File)
{
	strbuf.reserve(128);
	fVarType = ctBasic;
	Node = File.GetObj("allele", TRUE);
	ChromIndex = &File.Chromosome();
	PtrPos = &File.Position()[0];
	VarNode = NULL;
	Reset();
}

void CApply_Variant_ChromPosAllele::ReadData(PyObject *val)
{
	C_Int32 st = Position, one = 1;
	GDS_Array_ReadData(Node, &st, &one, &strbuf, svStrUTF8);
	for (size_t i=0; i < strbuf.size(); i++)
		if (strbuf[i] == ',') strbuf[i] = '_';
	char buf[8192];
	snprintf(buf, sizeof(buf), "%s:%d_%s",
		(*ChromIndex)[Position].c_str(), PtrPos[Position], strbuf.c_str());
	set_str(val, 0, buf);
}

PyObject* CApply_Variant_ChromPosAllele::NeedArray()
{
	if (VarNode == NULL) VarNode = numpy_new_string(1);
	return VarNode;
}


// =====================================================================
// Object for reading the 1-based variant index variant by variant

CApply_Variant_VariantIndex::CApply_Variant_VariantIndex(CFileInfo &File):
	CApply_Variant(File)
{
	fVarType = ctBasic;
	Node = NULL;
	VarNode = NULL;
	Reset();
}

void CApply_Variant_VariantIndex::ReadData(PyObject *val)
{
	((int*)PyArray_DATA(val))[0] = Position + 1;
}

PyObject* CApply_Variant_VariantIndex::NeedArray()
{
	if (VarNode == NULL) VarNode = numpy_new_int32(1);
	return VarNode;
}

}


extern "C"
{
using namespace PySeqArray;

// ===========================================================
// Apply functions over margins on a working space
// ===========================================================

COREARRAY_DLL_LOCAL const char *Txt_Apply_AsIs[] =
{
	"none", "list", "integer", "double", "character", "logical",
	"raw", NULL
};

COREARRAY_DLL_LOCAL const char *Txt_Apply_VarIdx[] =
{
	"none", "relative", "absolute", NULL
};


/// build a CApply_Variant_* reader from a variable name
static CVarApply *new_var_apply(CFileInfo &File, const string &s, int use_raw)
{
	if (s==VAR_ID || s==VAR_ALLELE || s==VAR_ANNOT_ID ||
		s==VAR_ANNOT_QUAL || s==VAR_ANNOT_FILTER)
		return new CApply_Variant_Basic(File, s.c_str());
	else if (s == VAR_POSITION)
		return new CApply_Variant_Pos(File);
	else if (s == VAR_CHROM)
		return new CApply_Variant_Chrom(File);
	else if (s == VAR_GENOTYPE)
		return new CApply_Variant_Geno(File, use_raw);
	else if (s == VAR_PHASE)
		return new CApply_Variant_Phase(File, use_raw!=FALSE);
	else if (strncmp(s.c_str(), "annotation/info/", 16) == 0)
		return new CApply_Variant_Info(File, s.c_str());
	else if (strncmp(s.c_str(), "annotation/format/", 18) == 0)
		return new CApply_Variant_Format(File, s.c_str());
	else if (s == VAR_DOSAGE)
		return new CApply_Variant_Dosage(File, use_raw, false, false);
	else if (s == VAR_DOSAGE_ALT)
		return new CApply_Variant_Dosage(File, use_raw, true, false);
	else if (s == VAR_DOSAGE_ALT2)
		return new CApply_Variant_Dosage(File, use_raw, true, true);
	else if (s == VAR_NUM_ALLELE)
		return new CApply_Variant_NumAllele(File);
	else if (s == VAR_REF_ALLELE)
		return new CApply_Variant_RefAllele(File);
	else if (s == VAR_ALT_ALLELE)
		return new CApply_Variant_AltAllele(File);
	else if (s == VAR_CHROM_POS)
		return new CApply_Variant_ChromPos(File);
	else if (s == VAR_CHROM_POS_ALLELE)
		return new CApply_Variant_ChromPosAllele(File);
	else if (s == VAR_VARIANT_INDEX)
		return new CApply_Variant_VariantIndex(File);
	else
		throw ErrSeqArray("'%s' is not a valid variable name.", s.c_str());
}


/// Apply a Python function variant by variant.
/// Python signature: apply_variant(file_id, var_name, FUN, as_is, var_index,
///                                  use_raw, verbose)
///   var_name : a string or a list/tuple of strings
///   as_is    : one of none/list/integer/double/character/logical/raw
///   var_index: one of none/relative/absolute
COREARRAY_DLL_EXPORT PyObject* SEQ_Apply_Variant(PyObject *self, PyObject *args)
{
	int file_id;
	PyObject *var_name, *FUN;
	const char *as_is, *var_index;
	int use_raw, verbose;
	if (!PyArg_ParseTuple(args, "iOOssii", &file_id, &var_name, &FUN,
			&as_is, &var_index, &use_raw, &verbose))
		return NULL;

	if (!PyCallable_Check(FUN))
	{
		PyErr_SetString(PyExc_TypeError, "'FUN' must be callable.");
		return NULL;
	}

	COREARRAY_TRY

		CFileInfo &File = GetFileInfo(file_id);

		int nVariant = File.VariantSelNum();
		int DatType = MatchText(as_is, Txt_Apply_AsIs);
		if (DatType < 0)
			throw ErrSeqArray("'as.is' is not valid!");
		int VarIdx = MatchText(var_index, Txt_Apply_VarIdx);
		if (VarIdx < 0)
			throw ErrSeqArray("'var.index' is not valid!");

		// variable names
		vector<string> names;
		numpy_to_string(var_name, names);
		if (names.empty())
			throw ErrSeqArray("'var.name' should be specified.");

		// empty selection
		if (nVariant <= 0)
		{
			switch (DatType)
			{
			case 1: return PyList_New(0);
			case 2: return numpy_new_int32(0);
			case 3: return numpy_new_double(0);
			case 4: return numpy_new_string(0);
			case 5: return numpy_new_bool(0);
			case 6: return numpy_new_uint8(0);
			default: Py_RETURN_NONE;
			}
		}

		// initialize the reader list
		CVarApplyList NodeList;
		for (size_t i=0; i < names.size(); i++)
			NodeList.push_back(new_var_apply(File, names[i], use_raw));

		// the output object
		PyObject *rv_ans = NULL;
		C_Int8 *rv_ptr = NULL;
		switch (DatType)
		{
		case 1: rv_ans = PyList_New(nVariant); break;
		case 2: rv_ans = numpy_new_int32(nVariant);
			rv_ptr = (C_Int8*)numpy_getptr(rv_ans); break;
		case 3: rv_ans = numpy_new_double(nVariant);
			rv_ptr = (C_Int8*)numpy_getptr(rv_ans); break;
		case 4: rv_ans = numpy_new_string(nVariant); break;
		case 5: rv_ans = numpy_new_bool(nVariant);
			rv_ptr = (C_Int8*)numpy_getptr(rv_ans); break;
		case 6: rv_ans = numpy_new_uint8(nVariant);
			rv_ptr = (C_Int8*)numpy_getptr(rv_ans); break;
		}

		const size_t nNode = NodeList.size();
		CProgressStdOut progress(nVariant, verbose!=0);

		int ans_index = 0;
		do {
			// build the argument for FUN
			PyObject *data;
			if (nNode <= 1)
			{
				data = NodeList[0]->NeedArray();
				NodeList[0]->ReadData(data);
				Py_INCREF(data);
			} else {
				data = PyList_New(nNode);
				for (size_t i=0; i < nNode; i++)
				{
					PyObject *t = NodeList[i]->NeedArray();
					NodeList[i]->ReadData(t);
					Py_INCREF(t);
					PyList_SetItem(data, i, t);
				}
			}

			PyObject *fargs;
			if (VarIdx > 0)
			{
				int idx = (VarIdx==1) ? (ans_index+1) : (NodeList[0]->Position+1);
				fargs = PyTuple_New(2);
				PyTuple_SetItem(fargs, 0, PyLong_FromLong(idx));
				PyTuple_SetItem(fargs, 1, data);  // steals data
			} else {
				fargs = PyTuple_New(1);
				PyTuple_SetItem(fargs, 0, data);  // steals data
			}

			PyObject *val = PyObject_CallObject(FUN, fargs);
			Py_DECREF(fargs);
			if (val == NULL)
			{
				if (rv_ans) Py_DECREF(rv_ans);
				return NULL;
			}

			// store
			switch (DatType)
			{
			case 1:  // list
				PyList_SetItem(rv_ans, ans_index, val);  // steals val
				break;
			case 2:  // integer
				*((int*)rv_ptr) = (int)PyLong_AsLong(val);
				rv_ptr += sizeof(int); Py_DECREF(val);
				break;
			case 3:  // double
				*((double*)rv_ptr) = PyFloat_AsDouble(val);
				rv_ptr += sizeof(double); Py_DECREF(val);
				break;
			case 4:  // character
				set_str(rv_ans, ans_index,
					PyUnicode_Check(val) ? PyUnicode_AsUTF8(val) : "");
				Py_DECREF(val);
				break;
			case 5:  // logical
				*((C_Int8*)rv_ptr) = (PyObject_IsTrue(val) ? 1 : 0);
				rv_ptr += 1; Py_DECREF(val);
				break;
			case 6:  // raw
				*rv_ptr = (C_UInt8)PyLong_AsLong(val);
				rv_ptr += 1; Py_DECREF(val);
				break;
			default: // none
				Py_DECREF(val);
				break;
			}
			ans_index ++;
			progress.Forward();

		} while (NodeList.CallNext());

		if (rv_ans) return rv_ans;

	COREARRAY_CATCH_NONE
}

} // extern "C"
