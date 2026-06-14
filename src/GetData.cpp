// ===========================================================
//
// GetData.cpp: Get data from the GDS file
//
// Copyright (C) 2017-2026    Xiuwen Zheng
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

// Native (Python/NumPy) port of SeqArray's src/GetData.cpp, updated to the
// current engine. Data are returned as NumPy arrays (via pygds
// GDS_Py_Array_Read) or, for variable-length INFO/FORMAT, a dict
// {'index':..., 'data':...}. The new $-selectors ($dosage_alt[2], $ref, $alt,
// $num_allele, $chrom_pos[_allele], $sample_index, $variant_index) are
// supported. Sparse dosage ($dosage_sp/$dosage_sp2) is returned as a dict
// {'sparse':'csc','data','indices','indptr','shape'} that the Python layer
// wraps into scipy.sparse.csc_matrix. The Bioconductor-only paths of the R
// source (S4Vectors::Rle, IRanges CompressedList, R-environment variables)
// are intentionally not ported.

#include "Index.h"
#include "ReadByVariant.h"


using namespace PySeqArray;

extern "C"
{

static const char *ERR_DIM = "Invalid dimension of '%s'.";


// ---- sparse alt-allele dosage as a CSC dict (for scipy.sparse) ----
static PyObject* get_dosage_sparse(CFileInfo &File, bool partial_missing)
{
	const ssize_t nSample  = File.SampleSelNum();
	const ssize_t nVariant = File.VariantSelNum();
	// dgCMatrix-style CSC: column = variant, row = sample
	vector<int> indices;     // row (sample) indices of non-zeros
	vector<double> data;     // values
	vector<int> indptr(nVariant + 1, 0);

	if ((nSample > 0) && (nVariant > 0))
	{
		CApply_Variant_Dosage NodeVar(File, 0, true, partial_missing);
		vector<int> buf(nSample);
		int col = 0;
		do {
			if (partial_missing)
				NodeVar.ReadDosageAlt_p(&buf[0]);
			else
				NodeVar.ReadDosageAlt(&buf[0]);
			for (ssize_t j=0; j < nSample; j++)
			{
				int g = buf[j];
				if (g != 0)
				{
					indices.push_back((int)j);
					data.push_back((g != NA_INTEGER) ? (double)g : NAN);
				}
			}
			indptr[++col] = (int)indices.size();
		} while (NodeVar.Next());
	}

	const size_t nnz = data.size();
	PyObject *py_data = numpy_new_double(nnz);
	PyObject *py_ind  = numpy_new_int32(nnz);
	PyObject *py_ptr  = numpy_new_int32(nVariant + 1);
	if (nnz > 0)
	{
		memcpy(numpy_getptr(py_data), &data[0], sizeof(double)*nnz);
		memcpy(numpy_getptr(py_ind), &indices[0], sizeof(int)*nnz);
	}
	memcpy(numpy_getptr(py_ptr), &indptr[0], sizeof(int)*(nVariant+1));

	// {'sparse':'csc', 'data':.., 'indices':.., 'indptr':.., 'shape':(r,c)}
	return Py_BuildValue("{s:s,s:N,s:N,s:N,s:(ii)}",
		"sparse", "csc", "data", py_data, "indices", py_ind,
		"indptr", py_ptr, "shape", (int)nSample, (int)nVariant);
}


// get data
static PyObject* VarGetData(CFileInfo &File, const char *name)
{
	PyObject *rv_ans = NULL;
	TSelection &Sel = File.Selection();
	Sel.GetStructVariant();

	if (strcmp(name, "sample.id") == 0)
	{
		// ===========================================================
		// sample.id
		PdAbstractArray N = File.GetObj(name, TRUE);
		if ((GDS_Array_DimCnt(N) != 1) ||
				(GDS_Array_GetTotalCount(N) != File.SampleNum()))
			throw ErrSeqArray(ERR_DIM, name);
		C_BOOL *ss = Sel.pSample;
		rv_ans = GDS_Py_Array_Read(N, NULL, NULL, &ss, svCustom);

	} else if (strcmp(name, "position") == 0)
	{
		int n = File.VariantSelNum();
		rv_ans = numpy_new_int32(n);
		if (n > 0)
		{
			const int *base = &File.Position()[0];
			int *p = (int*)numpy_getptr(rv_ans);
			C_BOOL *s = Sel.pVariant;
			for (size_t m=File.VariantNum(); m > 0; m--)
			{
				if (*s++) *p++ = *base;
				base ++;
			}
		}

	} else if (strcmp(name, "chromosome") == 0)
	{
		int n = File.VariantSelNum();
		rv_ans = numpy_new_string(n);
		if (n > 0)
		{
			CChromIndex &Chrom = File.Chromosome();
			PyObject **p = (PyObject**)numpy_getptr(rv_ans);
			C_BOOL *s = Sel.pVariant;
			size_t m = File.VariantNum();
			string lastss;
			PyObject *last = NULL;
			for (size_t i=0; i < m; i++)
			{
				if (*s++)
				{
					const string &ss = Chrom[i];
					if (ss != lastss)
					{
						lastss = ss;
						last = NULL;
					}
					if (!last)
						last = PYSTR_SET2(&lastss[0], lastss.size());
					numpy_setval(rv_ans, p++, last);
				}
			}
		}

	} else if ( (strcmp(name, "variant.id")==0) ||
		(strcmp(name, "allele")==0) ||
		(strcmp(name, "annotation/id")==0) ||
		(strcmp(name, "annotation/qual")==0) ||
		(strcmp(name, "annotation/filter")==0) )
	{
		// ===========================================================
		// variant.id, allele, annotation/id, annotation/qual, annotation/filter
		PdAbstractArray N = File.GetObj(name, TRUE);
		if ((GDS_Array_DimCnt(N) != 1) ||
				(GDS_Array_GetTotalCount(N) != File.VariantNum()))
			throw ErrSeqArray(ERR_DIM, name);
		C_BOOL *ss = Sel.pVariant;
		rv_ans = GDS_Py_Array_Read(N, NULL, NULL, &ss, svCustom);

	} else if (strcmp(name, "genotype") == 0)
	{
		// ===========================================================
		// genotypic data (uint8, dim = variant x sample x ploidy)
		int nSample  = File.SampleSelNum();
		int nVariant = File.VariantSelNum();
		if ((nSample > 0) && (nVariant > 0))
		{
			CApply_Variant_Geno NodeVar(File, 1);  // use_raw -> uint8
			rv_ans = numpy_new_uint8_dim3(nVariant, nSample, File.Ploidy());
			C_UInt8 *base = (C_UInt8*)numpy_getptr(rv_ans);
			ssize_t SIZE = (ssize_t)nSample * File.Ploidy();
			do {
				NodeVar.ReadGenoData(base);
				base += SIZE;
			} while (NodeVar.Next());
		} else
			rv_ans = numpy_new_uint8(0);

	} else if (strcmp(name, "@genotype") == 0)
	{
		static const char *VarName = "genotype/@data";
		PdAbstractArray N = File.GetObj(VarName, TRUE);
		if ((GDS_Array_DimCnt(N) != 1) ||
				(GDS_Array_GetTotalCount(N) != File.VariantNum()))
			throw ErrSeqArray(ERR_DIM, VarName);
		C_BOOL *ss = Sel.pVariant;
		rv_ans = GDS_Py_Array_Read(N, NULL, NULL, &ss, svInt32);

	} else if (strcmp(name, "$dosage")==0)
	{
		// dosage of the reference allele (uint8, variant x sample)
		ssize_t nSample  = File.SampleSelNum();
		ssize_t nVariant = File.VariantSelNum();
		if ((nSample > 0) && (nVariant > 0))
		{
			CApply_Variant_Dosage NodeVar(File, 1, false, false);
			rv_ans = numpy_new_uint8_mat(nVariant, nSample);
			C_UInt8 *base = (C_UInt8*)numpy_getptr(rv_ans);
			do {
				NodeVar.ReadDosage(base);
				base += nSample;
			} while (NodeVar.Next());
		} else
			rv_ans = numpy_new_uint8(0);

	} else if (strcmp(name, "$dosage_alt")==0)
	{
		// dosage of the alternative allele(s) (int32, variant x sample)
		ssize_t nSample  = File.SampleSelNum();
		ssize_t nVariant = File.VariantSelNum();
		if ((nSample > 0) && (nVariant > 0))
		{
			CApply_Variant_Dosage NodeVar(File, 0, true, false);
			rv_ans = numpy_new_int32_mat(nVariant, nSample);
			int *base = (int*)numpy_getptr(rv_ans);
			do {
				NodeVar.ReadDosageAlt(base);
				base += nSample;
			} while (NodeVar.Next());
		} else
			rv_ans = numpy_new_int32(0);

	} else if (strcmp(name, "$dosage_alt2")==0)
	{
		// dosage of the alternative allele(s), partial missing allowed
		ssize_t nSample  = File.SampleSelNum();
		ssize_t nVariant = File.VariantSelNum();
		if ((nSample > 0) && (nVariant > 0))
		{
			CApply_Variant_Dosage NodeVar(File, 0, true, true);
			rv_ans = numpy_new_int32_mat(nVariant, nSample);
			int *base = (int*)numpy_getptr(rv_ans);
			do {
				NodeVar.ReadDosageAlt_p(base);
				base += nSample;
			} while (NodeVar.Next());
		} else
			rv_ans = numpy_new_int32(0);

	} else if (strcmp(name, "$dosage_sp")==0)
	{
		// sparse dosage of alternative allele(s) -> scipy.sparse CSC dict
		rv_ans = get_dosage_sparse(File, false);

	} else if (strcmp(name, "$dosage_sp2")==0)
	{
		// sparse dosage, partial missing allowed
		rv_ans = get_dosage_sparse(File, true);

	} else if (strcmp(name, "phase") == 0)
	{
		// ===========================================================
		// phase/
		PdAbstractArray N = File.GetObj("phase/data", TRUE);
		int ndim = GDS_Array_DimCnt(N);
		C_Int32 dim[4];
		GDS_Array_GetDim(N, dim, 3);
		if (ndim<2 || ndim>3 || dim[0]!= File.VariantNum() ||
				dim[1]!=File.SampleNum())
			throw ErrSeqArray(ERR_DIM, name);
		C_BOOL *ss[3] = { Sel.pVariant, Sel.pSample, NULL };
		if (ndim == 3)
			ss[2] = NeedArrayTRUEs(dim[2]);
		rv_ans = GDS_Py_Array_Read(N, NULL, NULL, ss, svCustom);

	} else if (strncmp(name, "annotation/info/@", 17) == 0)
	{
		string nm(name);
		nm.erase(16, 1);  // strip the '@' -> annotation/info/X
		if (File.GetObj(name, FALSE) != NULL)
		{
			CIndex &V = VarGetStruct(File, nm).Index;
			rv_ans = V.GetLen_Sel(Sel.pVariant);
		}

	} else if (strncmp(name, "annotation/info/", 16) == 0)
	{
		// ===========================================================
		// annotation/info
		GDS_PATH_PREFIX_CHECK(name);
		PdAbstractArray N = File.GetObj(name, TRUE);
		int ndim = GDS_Array_DimCnt(N);
		if ((ndim!=1) && (ndim!=2))
			throw ErrSeqArray(ERR_DIM, name);

		string name2 = GDS_PATH_PREFIX(name, '@');
		PdAbstractArray N_idx = File.GetObj(name2.c_str(), FALSE);
		if (N_idx == NULL)
		{
			// no index
			C_Int32 dim[4];
			GDS_Array_GetDim(N, dim, 2);
			C_BOOL *ss[2] = { Sel.pVariant, NULL };
			if (ndim == 2)
				ss[1] = NeedArrayTRUEs(dim[1]);
			rv_ans = GDS_Py_Array_Read(N, NULL, NULL, ss, svCustom);
		} else {
			// with index
			CIndex &V = VarGetStruct(File, name).Index;
			int var_start, var_count;
			vector<C_BOOL> var_sel;
			PyObject *Index = V.GetLen_Sel(Sel.pVariant, var_start, var_count, var_sel);

			C_BOOL *ss[2] = { &var_sel[0], NULL };
			C_Int32 dimst[2]  = { var_start, 0 };
			C_Int32 dimcnt[2] = { var_count, 0 };
			if (ndim == 2)
			{
				GDS_Array_GetDim(N, dimcnt, 2);
				dimcnt[0] = var_count;
			}
			PyObject *Val = GDS_Py_Array_Read(N, dimst, dimcnt, ss, svCustom);
			rv_ans = Py_BuildValue("{s:N,s:N}", "index", Index, "data", Val);
		}

	} else if (strncmp(name, "annotation/format/@", 19) == 0)
	{
		string nm(name);
		nm.erase(18, 1);  // strip '@' -> annotation/format/X
		string name2 = nm + "/@data";
		if (File.GetObj(name2.c_str(), FALSE) != NULL)
		{
			CIndex &V = VarGetStruct(File, nm).Index;
			rv_ans = V.GetLen_Sel(Sel.pVariant);
		}

	} else if (strncmp(name, "annotation/format/", 18) == 0)
	{
		// ===========================================================
		// annotation/format
		GDS_PATH_PREFIX_CHECK(name);
		string name1 = string(name) + "/data";
		PdAbstractArray N = File.GetObj(name1.c_str(), TRUE);

		CIndex &V = VarGetStruct(File, name1).Index;
		int var_start, var_count;
		vector<C_BOOL> var_sel;
		PyObject *Index = V.GetLen_Sel(Sel.pVariant, var_start, var_count, var_sel);

		C_BOOL *ss[2] = { &var_sel[0], Sel.pSample };
		C_Int32 dimst[2]  = { var_start, 0 };
		C_Int32 dimcnt[2];
		GDS_Array_GetDim(N, dimcnt, 2);
		dimcnt[0] = var_count;
		PyObject *Val = GDS_Py_Array_Read(N, dimst, dimcnt, ss, svCustom);

		rv_ans = Py_BuildValue("{s:N,s:N}", "index", Index, "data", Val);

	} else if (strncmp(name, "sample.annotation/", 18) == 0)
	{
		// ===========================================================
		// sample.annotation
		GDS_PATH_PREFIX_CHECK(name);
		PdAbstractArray N = File.GetObj(name, TRUE);
		int ndim = GDS_Array_DimCnt(N);
		if ((ndim!=1) && (ndim!=2))
			throw ErrSeqArray(ERR_DIM, name);
		C_Int32 dim[2];
		GDS_Array_GetDim(N, dim, 2);
		if (dim[0] != File.SampleNum())
			throw ErrSeqArray(ERR_DIM, name);

		C_BOOL *ss[2] = { Sel.pSample, NULL };
		if (ndim == 2)
			ss[1] = NeedArrayTRUEs(dim[1]);
		rv_ans = GDS_Py_Array_Read(N, NULL, NULL, ss, svCustom);

	} else if (strcmp(name, "$num_allele")==0)
	{
		// the number of distinct alleles per variant
		ssize_t nVariant = File.VariantSelNum();
		rv_ans = numpy_new_int32(nVariant);
		int *p = (int*)numpy_getptr(rv_ans);
		if (nVariant > 0)
		{
			CApply_Variant_NumAllele NodeVar(File);
			for (ssize_t i=0; i < nVariant; i++)
			{
				p[i] = NodeVar.GetNumAllele();
				NodeVar.Next();
			}
		}

	} else if (strcmp(name, "$ref")==0)
	{
		// the reference allele per variant
		PdAbstractArray N = File.GetObj("allele", TRUE);
		if ((GDS_Array_DimCnt(N) != 1) ||
				(GDS_Array_GetTotalCount(N) != File.VariantNum()))
			throw ErrSeqArray(ERR_DIM, name);
		size_t n = File.VariantSelNum();
		vector<string> buffer(n);
		C_BOOL *ss = Sel.pVariant;
		GDS_Array_ReadDataEx(N, NULL, NULL, &ss, &buffer[0], svStrUTF8);
		rv_ans = numpy_new_string(n);
		PyObject **pi = (PyObject**)numpy_getptr(rv_ans);
		for (size_t i=0; i < n; i++)
		{
			const char *p = buffer[i].c_str();
			size_t m = 0;
			for (const char *s=p; *s!=',' && *s!=0; s++) m++;
			numpy_setval(rv_ans, pi, PYSTR_SET2(p, m));
			pi ++;
		}

	} else if (strcmp(name, "$alt")==0)
	{
		// the alternative allele(s) per variant
		PdAbstractArray N = File.GetObj("allele", TRUE);
		if ((GDS_Array_DimCnt(N) != 1) ||
				(GDS_Array_GetTotalCount(N) != File.VariantNum()))
			throw ErrSeqArray(ERR_DIM, name);
		size_t n = File.VariantSelNum();
		vector<string> buffer(n);
		C_BOOL *ss = Sel.pVariant;
		GDS_Array_ReadDataEx(N, NULL, NULL, &ss, &buffer[0], svStrUTF8);
		rv_ans = numpy_new_string(n);
		PyObject **pi = (PyObject**)numpy_getptr(rv_ans);
		for (size_t i=0; i < n; i++)
		{
			const char *p = buffer[i].c_str();
			for (; *p!=',' && *p!=0; p++);
			if (*p == ',') p++;
			numpy_setval(rv_ans, pi, PYSTR_SET(p));
			pi ++;
		}

	} else if (strcmp(name, "$chrom_pos")==0)
	{
		// "chromosome:position" per variant
		int n = File.VariantSelNum();
		rv_ans = numpy_new_string(n);
		if (n > 0)
		{
			CChromIndex &Chrom = File.Chromosome();
			const int *pos = &File.Position()[0];
			C_BOOL *s = Sel.pVariant + Sel.varStart;
			PyObject **p = (PyObject**)numpy_getptr(rv_ans);
			size_t i = Sel.varStart;
			char buf[1024];
			for (; n > 0; i++)
			{
				if (*s++)
				{
					snprintf(buf, sizeof(buf), "%s:%d", Chrom[i].c_str(), pos[i]);
					numpy_setval(rv_ans, p++, PYSTR_SET(buf));
					n--;
				}
			}
		}

	} else if (strcmp(name, "$chrom_pos_allele")==0)
	{
		// "chromosome:position_ref_alt" per variant
		PdAbstractArray N = File.GetObj("allele", TRUE);
		CChromIndex &Chrom = File.Chromosome();
		const int *pos = &File.Position()[0];
		size_t n = File.VariantSelNum();
		vector<string> allele(n);
		C_BOOL *ss = Sel.pVariant;
		GDS_Array_ReadDataEx(N, NULL, NULL, &ss, &allele[0], svStrUTF8);
		rv_ans = numpy_new_string(n);
		PyObject **p = (PyObject**)numpy_getptr(rv_ans);
		C_BOOL *s = Sel.pVariant + Sel.varStart;
		size_t i = Sel.varStart, k = 0;
		char buf[8192];
		while (k < n)
		{
			if (*s++)
			{
				for (size_t j=0; j < allele[k].size(); j++)
					if (allele[k][j] == ',') allele[k][j] = '_';
				snprintf(buf, sizeof(buf), "%s:%d_%s",
					Chrom[i].c_str(), pos[i], allele[k].c_str());
				numpy_setval(rv_ans, p++, PYSTR_SET(buf));
				k++;
			}
			i++;
		}

	} else if (strcmp(name, "$sample_index")==0)
	{
		// 1-based indices of selected samples
		ssize_t num = File.SampleSelNum();
		rv_ans = numpy_new_int32(num);
		int *p = (int*)numpy_getptr(rv_ans), i = 0;
		const C_BOOL *s = Sel.pSample;
		while (num > 0)
			if (s[i++]) { *p++ = i; num--; }

	} else if (strcmp(name, "$variant_index")==0)
	{
		// 1-based indices of selected variants
		ssize_t num = File.VariantSelNum();
		rv_ans = numpy_new_int32(num);
		int *p = (int*)numpy_getptr(rv_ans), i = Sel.varStart;
		const C_BOOL *s = Sel.pVariant;
		while (num > 0)
			if (s[i++]) { *p++ = i; num--; }

	} else {
		throw ErrSeqArray(
			"'%s' is not a standard variable name; valid names include:\n"
			"    sample.id, variant.id, position, chromosome, allele, genotype,\n"
			"    phase, annotation/id, annotation/qual, annotation/filter,\n"
			"    annotation/info/VARIABLE, annotation/format/VARIABLE,\n"
			"    sample.annotation/VARIABLE, and $dosage, $dosage_alt[2],\n"
			"    $dosage_sp[2], $num_allele, $ref, $alt, $chrom_pos[_allele],\n"
			"    $sample_index, $variant_index.", name);
	}

	return rv_ans;
}


/// Get data from a working space
COREARRAY_DLL_EXPORT PyObject* SEQ_GetData(PyObject *self, PyObject *args)
{
	int file_id;
	const char *name;
	if (!PyArg_ParseTuple(args, "is", &file_id, &name))
		return NULL;

	COREARRAY_TRY
		CFileInfo &File = GetFileInfo(file_id);
		return VarGetData(File, name);
	COREARRAY_CATCH_NONE
}


/// Apply functions over variants in block
COREARRAY_DLL_EXPORT PyObject* SEQ_BApply_Variant(PyObject *self, PyObject *args)
{
	int file_id;
	PyObject *name;
	PyObject *func;
	PyObject *obj;
	const char *as_is;
	int bsize;
	int verbose;
	if (!PyArg_ParseTuple(args, "iOOOsi" BSTR, &file_id, &name, &func,
			&obj, &as_is, &bsize, &verbose))
		return NULL;

	if (!PyCallable_Check(func))
	{
		PyErr_SetString(PyExc_TypeError, "'fun' must be callable.");
		return NULL;
	}
	if (bsize < 1)
	{
		PyErr_SetString(PyExc_ValueError, "'bsize' must be >= 1.");
		return NULL;
	}

	COREARRAY_TRY

		vector<string> name_list;
		numpy_to_string(name, name_list);
		if (name_list.empty())
			throw ErrSeqArray("'name' should be specified.");

		PyObject *rv_ans = NULL;

		CFileInfo &File = GetFileInfo(file_id);
		File.VarMap().clear();
		TSelection &Selection = File.Selection();
		Selection.GetStructVariant();

		// the number of selected variants
		int nVariant = File.VariantSelNum();
		if (nVariant <= 0)
			throw ErrSeqArray("There is no selected variant.");

		// the number of data blocks
		int NumBlock = nVariant / bsize;
		if (nVariant % bsize) NumBlock ++;

		// as_is
		if (strcmp(as_is, "list")==0 || strcmp(as_is, "unlist")==0)
			rv_ans = PyList_New(NumBlock);
		else if (strcmp(as_is, "none") != 0)
			throw ErrSeqArray("'asis' should be 'none', 'list' or 'unlist'.");

		// function arguments
		int num_var = name_list.size();
		int st_var = 0;
		if (obj != Py_None) { num_var++; st_var = 1; }

		// local selection (initialize sample filter, clear variant filter)
		TSelection &Sel = File.Push_Selection(true, false);

		C_BOOL *pBase, *pSel, *pEnd;
		pBase = pSel = Selection.pVariant;
		pEnd = pBase + File.VariantNum();

		// progress object
		CProgressStdOut progress(NumBlock, verbose!=0);

		// for-loop
		for (int idx=0; idx < NumBlock; idx++)
		{
			// assign sub-selection
			Sel.ClearSelectVariant();
			pSel = VEC_BOOL_FIND_TRUE(pSel, pEnd);
			Sel.varStart = pSel - pBase;
			C_BOOL *pNewSel = Sel.pVariant;
			int bs = bsize;
			for (; bs > 0; bs--)
			{
				while ((pSel < pEnd) && (*pSel == FALSE)) pSel ++;
				if (pSel < pEnd)
				{
					pNewSel[pSel - pBase] = TRUE;
					pSel ++;
				} else
					break;
			}
			Sel.varTrueNum = bsize - bs;
			Sel.varEnd = pSel - pBase;

			// build the argument tuple and load data
			PyObject *fargs = PyTuple_New(num_var);
			if (obj != Py_None)
			{
				Py_INCREF(obj);
				PyTuple_SetItem(fargs, 0, obj);
			}
			for (int i=st_var; i < num_var; i++)
			{
				PyObject *v = VarGetData(File, name_list[i-st_var].c_str());
				PyTuple_SetItem(fargs, i, v);  // steals v
			}

			// call Python function
			PyObject *val = PyObject_CallObject(func, fargs);
			Py_DECREF(fargs);
			if (val == NULL)
			{
				File.Pop_Selection();
				if (rv_ans) Py_DECREF(rv_ans);
				return NULL;
			}

			// store data
			if (rv_ans && val!=Py_None)
				PyList_SetItem(rv_ans, idx, val);  // steals val
			else
				Py_DECREF(val);

			progress.Forward();
		}

		File.Pop_Selection();

		if (rv_ans) return rv_ans;

	COREARRAY_CATCH_NONE
}

} // extern "C"
