// ===========================================================
//
// ReadByVariant.h: Read data variant by variant
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

// Native (Python/NumPy) port of SeqArray's src/ReadByVariant.h. Each
// CApply_Variant_* reader replaces R's SEXP-based NeedRData(int&)/ReadData(SEXP)
// with NeedArray()/ReadData(PyObject*).

#include "Index.h"


namespace PySeqArray
{

using namespace Vectorization;


// =====================================================================

/// Object for reading basic variables variant by variant
class COREARRAY_DLL_LOCAL CApply_Variant_Basic: public CApply_Variant
{
protected:
	C_SVType SVType;
	PyObject *VarNode;  ///< Python object
public:
	/// constructor
	CApply_Variant_Basic(CFileInfo &File, const char *var_name);
	virtual void ReadData(PyObject *val);
	virtual PyObject *NeedArray();
};


/// Object for reading positions variant by variant
class COREARRAY_DLL_LOCAL CApply_Variant_Pos: public CApply_Variant
{
protected:
	int *PtrPos;
	PyObject *VarNode;  ///< Python object
public:
	/// constructor
	CApply_Variant_Pos(CFileInfo &File);
	virtual void ReadData(PyObject *val);
	virtual PyObject *NeedArray();
};


/// Object for reading chromosomes variant by variant
class COREARRAY_DLL_LOCAL CApply_Variant_Chrom: public CApply_Variant
{
protected:
	CChromIndex *ChromIndex;
	PyObject *VarNode;  ///< Python object
public:
	/// constructor
	CApply_Variant_Chrom(CFileInfo &File);
	virtual void ReadData(PyObject *val);
	virtual PyObject *NeedArray();
};


// =====================================================================

/// Object for reading genotypes variant by variant
class COREARRAY_DLL_LOCAL CApply_Variant_Geno: public CApply_Variant
{
protected:
	CGenoIndex *GenoIndex;  ///< indexing genotypes
	ssize_t SiteCount;  ///< the total number of entries at a site
	ssize_t CellCount;  ///< the selected number of entries at a site
	int UseRaw;  ///< whether use RAW type: FALSE, int; TRUE, raw; NA: auto
	TSelection::TSampStruct *pSampSel;   ///< the structure for selected samples
	VEC_AUTO_PTR ExtPtr;  ///< a pointer to the additional buffer
	PyObject *VarIntGeno;    ///< genotype integer (NumPy) object
	PyObject *VarRawGeno;    ///< genotype RAW (NumPy uint8) object

	inline int _ReadGenoData(int *Base);
	inline C_UInt8 _ReadGenoData(C_UInt8 *Base);

public:
	ssize_t SampNum;  ///< the number of selected samples
	int Ploidy;       ///< ploidy

	/// constructor
	CApply_Variant_Geno();
	CApply_Variant_Geno(CFileInfo &File, int use_raw);

	void Init(CFileInfo &File, int use_raw);

	virtual void ReadData(PyObject *val);
	virtual PyObject *NeedArray();
	bool NeedIntType();

	/// read genotypes in 32-bit integer
	void ReadGenoData(int *Base);
	/// read genotypes in unsigned 8-bit intetger
	void ReadGenoData(C_UInt8 *Base);
};


// =====================================================================

/// Object for reading genotypes (dosages) variant by variant
class COREARRAY_DLL_LOCAL CApply_Variant_Dosage: public CApply_Variant_Geno
{
protected:
	PyObject *VarDosage;   ///< dosage (NumPy) object
	VEC_AUTO_PTR ExtPtr2;  ///< a pointer to the additional buffer for dosages
	bool IsAlt;            ///< if true, ReadData() returns the dosage of alternative alleles
	bool IsAlt2;           ///< if IsAlt=true && IsAlt2==true, ReadData() returns
	                       ///  the dosage of alternative alleles allowing partially missing
public:
	/// constructor
	CApply_Variant_Dosage(CFileInfo &File, int use_raw, bool alt, bool alt2);

	virtual void ReadData(PyObject *val);
	virtual PyObject *NeedArray();

	/// read dosages in 32-bit integer
	void ReadDosage(int *Base);
	/// read dosages of alternative alleles in 32-bit integer
	void ReadDosageAlt(int *Base);
	/// read dosages of alternative alleles in 32-bit integer, allowing partial missing
	void ReadDosageAlt_p(int *Base);
	/// read dosages in unsigned 8-bit intetger
	void ReadDosage(C_UInt8 *Base);
	/// read dosages of alternative alleles in unsigned 8-bit intetger
	void ReadDosageAlt(C_UInt8 *Base);
	/// read dosages of alternative alleles in unsigned 8-bit intetger, allowing partial missing
	void ReadDosageAlt_p(C_UInt8 *Base);
};


// =====================================================================

/// Object for reading phasing information variant by variant
class COREARRAY_DLL_LOCAL CApply_Variant_Phase: public CApply_Variant
{
protected:
	ssize_t SiteCount;  ///< the total number of entries at a site
	ssize_t CellCount;  ///< the selected number of entries at a site
	bool UseRaw;  ///< whether use RAW type
	vector<C_BOOL> Selection;  ///< the buffer of selection
	PyObject *VarPhase;  ///< phase (NumPy) object

public:
	ssize_t SampNum;  ///< the number of selected samples
	int Ploidy;       ///< ploidy

	/// constructor
	CApply_Variant_Phase();
	CApply_Variant_Phase(CFileInfo &File, bool use_raw);

	void Init(CFileInfo &File, bool use_raw);

	virtual void ReadData(PyObject *val);
	virtual PyObject *NeedArray();
};


// =====================================================================

/// Object for reading info variables variant by variant
class COREARRAY_DLL_LOCAL CApply_Variant_Info: public CApply_Variant
{
protected:
	CIndex *VarIndex;  ///< indexing the format variable
	C_SVType SVType;        ///< data type for GDS reading
	C_Int32 BaseNum;        ///< if 2-dim, the size of the first dimension
	map<int, PyObject*> VarList;  ///< a list of PyObject variables

public:
	/// constructor
	CApply_Variant_Info(CFileInfo &File, const char *var_name);

	virtual void ReadData(PyObject *val);
	virtual PyObject *NeedArray();
};


// =====================================================================

/// Object for reading format variables variant by variant
class COREARRAY_DLL_LOCAL CApply_Variant_Format: public CApply_Variant
{
protected:
	CIndex *VarIndex;  ///< indexing the format variable
	ssize_t _TotalSampNum;  ///< the total number of samples

	C_SVType SVType;        ///< data type for GDS reading
	C_BOOL *SelPtr[2];      ///< pointers to selection
	map<int, PyObject*> VarList;  ///< a list of PyObject variables

public:
	ssize_t SampNum;  ///< the number of selected samples

	/// constructor
	CApply_Variant_Format();
	CApply_Variant_Format(CFileInfo &File, const char *var_name);

	void Init(CFileInfo &File, const char *var_name);

	virtual void ReadData(PyObject *val);
	virtual PyObject *NeedArray();
};


// =====================================================================

/// Object for calculating the number of distinct alleles variant by variant
class COREARRAY_DLL_LOCAL CApply_Variant_NumAllele: public CApply_Variant
{
private:
	string strbuf;
protected:
	PyObject *VarNode;  ///< Python object
public:
	/// constructor
	CApply_Variant_NumAllele(CFileInfo &File);

	virtual void ReadData(PyObject *val);
	virtual PyObject *NeedArray();
	int GetNumAllele();
};


/// Object for reading the reference allele variant by variant
class COREARRAY_DLL_LOCAL CApply_Variant_RefAllele: public CApply_Variant
{
private:
	string strbuf;
protected:
	PyObject *VarNode;  ///< Python object
public:
	/// constructor
	CApply_Variant_RefAllele(CFileInfo &File);

	virtual void ReadData(PyObject *val);
	virtual PyObject *NeedArray();
};


/// Object for reading the alternative allele(s) variant by variant
class COREARRAY_DLL_LOCAL CApply_Variant_AltAllele: public CApply_Variant
{
private:
	string strbuf;
protected:
	PyObject *VarNode;  ///< Python object
public:
	/// constructor
	CApply_Variant_AltAllele(CFileInfo &File);

	virtual void ReadData(PyObject *val);
	virtual PyObject *NeedArray();
};


/// Object for reading chromosome:position variant by variant
class COREARRAY_DLL_LOCAL CApply_Variant_ChromPos: public CApply_Variant
{
protected:
	CChromIndex *ChromIndex;
	int *PtrPos;
	PyObject *VarNode;  ///< Python object
public:
	/// constructor
	CApply_Variant_ChromPos(CFileInfo &File);

	virtual void ReadData(PyObject *val);
	virtual PyObject *NeedArray();
};


/// Object for reading chromosome:position_allele variant by variant
class COREARRAY_DLL_LOCAL CApply_Variant_ChromPosAllele: public CApply_Variant
{
private:
	string strbuf;
protected:
	CChromIndex *ChromIndex;
	int *PtrPos;
	PyObject *VarNode;  ///< Python object
public:
	/// constructor
	CApply_Variant_ChromPosAllele(CFileInfo &File);

	virtual void ReadData(PyObject *val);
	virtual PyObject *NeedArray();
};


/// Object for reading the 1-based variant index variant by variant
class COREARRAY_DLL_LOCAL CApply_Variant_VariantIndex: public CApply_Variant
{
protected:
	PyObject *VarNode;  ///< Python object
public:
	/// constructor
	CApply_Variant_VariantIndex(CFileInfo &File);

	virtual void ReadData(PyObject *val);
	virtual PyObject *NeedArray();
};

}
