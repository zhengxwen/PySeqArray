// ===========================================================
//
// PySeqArray.cpp: the C/C++ codes for the PySeqArray package
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

// Module entry point + filter (working-space) functions, updated to the modern
// TSelection API: pSample/pVariant are raw pointers (not methods), the
// selection stack uses Push_Selection/Pop_Selection, and after mutating a raw
// selection the cached structures must be cleared (ClearStructVariant /
// ClearStructSample) so VariantSelNum()/genotype reading recompute correctly.

#include "Index.h"

#include <set>
#include <algorithm>

#include "ReadByVariant.h"
#include <ctype.h>


#define PY_EXPORT    static


extern "C"
{

using namespace CoreArray;
using namespace PySeqArray;


// ===========================================================
// Open / close a GDS file
// ===========================================================

/// initialize a SeqArray file
PY_EXPORT PyObject* SEQ_File_Init(PyObject *self, PyObject *args)
{
	int file_id;
	if (!PyArg_ParseTuple(args, "i", &file_id))
		return NULL;
	COREARRAY_TRY
		CFileInfo &file = GetFileInfo(file_id);
		file.Selection();  // force to initialize selection
	COREARRAY_CATCH_NONE
}

/// finalize a SeqArray file
PY_EXPORT PyObject* SEQ_File_Done(PyObject *self, PyObject *args)
{
	int file_id;
	if (!PyArg_ParseTuple(args, "i", &file_id))
		return NULL;
	COREARRAY_TRY
		map<int, CFileInfo>::iterator p = GDSFile_ID_Info.find(file_id);
		if (p != GDSFile_ID_Info.end())
			GDSFile_ID_Info.erase(p);
	COREARRAY_CATCH_NONE
}



// ===========================================================
// Set a working space
// ===========================================================

/// push the current filter to the stack
PY_EXPORT PyObject* SEQ_FilterPush(PyObject *self, PyObject *args)
{
	int file_id;
	int new_flag;
	if (!PyArg_ParseTuple(args, "i" BSTR, &file_id, &new_flag)) return NULL;

	COREARRAY_TRY
		map<int, CFileInfo>::iterator it = GDSFile_ID_Info.find(file_id);
		if (it == GDSFile_ID_Info.end())
			throw ErrSeqArray("The GDS file is closed or invalid.");
		CFileInfo &File = it->second;
		File.Selection();  // make sure it is initialized
		if (new_flag)
		{
			// push a fresh all-selected filter
			TSelection &s = File.Push_Selection(false, false);
			memset(s.pSample, TRUE, File.SampleNum());
			memset(s.pVariant, TRUE, File.VariantNum());
			s.ClearStructVariant();
			s.ClearStructSample();
		} else {
			// push a copy of the current filter
			File.Push_Selection(true, true);
		}
	COREARRAY_CATCH_NONE
}


/// pop up the previous filter from the stack
PY_EXPORT PyObject* SEQ_FilterPop(PyObject *self, PyObject *args)
{
	int file_id;
	if (!PyArg_ParseTuple(args, "i", &file_id)) return NULL;

	COREARRAY_TRY
		map<int, CFileInfo>::iterator it = GDSFile_ID_Info.find(file_id);
		if (it == GDSFile_ID_Info.end())
			throw ErrSeqArray("The GDS file is closed or invalid.");
		it->second.Pop_Selection();
	COREARRAY_CATCH_NONE
}


/// set a working space with selected sample id
PY_EXPORT PyObject* SEQ_SetSpaceSample(PyObject *self, PyObject *args)
{
	int file_id;
	PyObject *samp_id;
	int intersect, verbose;
	if (!PyArg_ParseTuple(args, "iO" BSTR BSTR, &file_id, &samp_id, &intersect, &verbose))
		return NULL;

	COREARRAY_TRY

		CFileInfo &File = GetFileInfo(file_id);
		TSelection &Sel = File.Selection();
		C_BOOL *pArray = Sel.pSample;
		int Count = File.SampleNum();
		PdAbstractArray varSamp = File.GetObj("sample.id", TRUE);

		if (samp_id == Py_None)
		{
			memset(pArray, TRUE, Count);
		} else if (numpy_is_array_or_list(samp_id))
		{
			if (numpy_is_array_int(samp_id))
			{
				set<int> set_id;
				{
					vector<int> ary;
					numpy_to_int32(samp_id, ary);
					set_id.insert(ary.begin(), ary.end());
				}
				vector<int> sample_id(Count);
				C_Int32 _st=0, _cnt=Count;
				GDS_Array_ReadData(varSamp, &_st, &_cnt, &sample_id[0], svInt32);
				if (!intersect)
				{
					for (int i=0; i < Count; i++)
						*pArray++ = (set_id.find(sample_id[i]) != set_id.end());
				} else {
					for (int i=0; i < Count; i++, pArray++)
						if (*pArray)
							*pArray = (set_id.find(sample_id[i]) != set_id.end());
				}
			} else {
				set<string> set_id;
				{
					vector<string> ary;
					numpy_to_string(samp_id, ary);
					set_id.insert(ary.begin(), ary.end());
				}
				vector<string> sample_id(Count);
				C_Int32 _st=0, _cnt=Count;
				GDS_Array_ReadData(varSamp, &_st, &_cnt, &sample_id[0], svStrUTF8);
				if (!intersect)
				{
					for (int i=0; i < Count; i++)
						*pArray++ = (set_id.find(sample_id[i]) != set_id.end());
				} else {
					for (int i=0; i < Count; i++, pArray++)
						if (*pArray)
							*pArray = (set_id.find(sample_id[i]) != set_id.end());
				}
			}
		} else
			throw ErrSeqArray("Invalid type of 'sample.id'.");

		// the sample selection changed
		Sel.ClearStructSample();

		if (verbose)
			printf("# of selected samples: %s\n", PrettyInt(File.SampleSelNum()));

	COREARRAY_CATCH_NONE
}


/// set a working space with selected sample id (bool vector or index)
PY_EXPORT PyObject* SEQ_SetSpaceSample2(PyObject *self, PyObject *args)
{
	int file_id;
	PyObject *samp_sel;
	int intersect, verbose;
	if (!PyArg_ParseTuple(args, "iO" BSTR BSTR, &file_id, &samp_sel, &intersect, &verbose))
		return NULL;

	COREARRAY_TRY

		CFileInfo &File = GetFileInfo(file_id);
		TSelection &Sel = File.Selection();
		C_BOOL *pArray = Sel.pSample;
		int Count = File.SampleNum();

		if (numpy_is_bool(samp_sel))
		{
			if (!intersect)
			{
				if (numpy_size(samp_sel) != (size_t)Count)
					throw ErrSeqArray("Invalid length of 'sample'.");
				memcpy(pArray, numpy_getptr(samp_sel), Count);
			} else {
				if (numpy_size(samp_sel) != (size_t)File.SampleSelNum())
					throw ErrSeqArray("Invalid length of 'sample' (should be equal to the number of selected samples).");
				C_BOOL *base = (C_BOOL*)numpy_getptr(samp_sel);
				for (int i=0; i < Count; i++)
					if (pArray[i]) pArray[i] = ((*base++) != 0);
			}
		} else if (numpy_is_int(samp_sel))
		{
			vector<int> idx;
			numpy_to_int32(samp_sel, idx);
			size_t N = idx.size();
			if (!intersect)
			{
				for (size_t i=0; i < N; i++)
					if ((idx[i] < 0) || (idx[i] >= Count))
						throw ErrSeqArray("Out of range 'sample'.");
				memset((void*)pArray, 0, Count);
				for (size_t i=0; i < N; i++) pArray[idx[i]] = TRUE;
			} else {
				int Cnt = File.SampleSelNum();
				for (size_t i=0; i < N; i++)
					if ((idx[i] < 0) || (idx[i] >= Cnt))
						throw ErrSeqArray("Out of range 'sample'.");
				vector<int> Idx;
				Idx.reserve(Cnt);
				for (int i=0; i < Count; i++)
					if (pArray[i]) Idx.push_back(i);
				memset((void*)pArray, 0, Count);
				for (size_t i=0; i < N; i++) pArray[Idx[idx[i]]] = TRUE;
			}
		} else if (samp_sel == Py_None)
		{
			memset(pArray, TRUE, Count);
		} else
			throw ErrSeqArray("Invalid type of 'sample'.");

		Sel.ClearStructSample();

		if (verbose)
			printf("# of selected samples: %s\n", PrettyInt(File.SampleSelNum()));

	COREARRAY_CATCH_NONE
}


/// set a working space with selected variant id
PY_EXPORT PyObject* SEQ_SetSpaceVariant(PyObject *self, PyObject *args)
{
	int file_id;
	PyObject *variant_id;
	int intersect, verbose;
	if (!PyArg_ParseTuple(args, "iO" BSTR BSTR, &file_id, &variant_id, &intersect, &verbose))
		return NULL;

	COREARRAY_TRY

		CFileInfo &File = GetFileInfo(file_id);
		TSelection &Sel = File.Selection();
		C_BOOL *pArray = Sel.pVariant;
		int Count = File.VariantNum();
		PdAbstractArray varVariant = File.GetObj("variant.id", TRUE);

		if (variant_id == Py_None)
		{
			memset(pArray, TRUE, Count);
		} else if (numpy_is_array_or_list(variant_id))
		{
			if (numpy_is_array_int(variant_id))
			{
				set<int> set_id;
				{
					vector<int> ary;
					numpy_to_int32(variant_id, ary);
					set_id.insert(ary.begin(), ary.end());
				}
				vector<int> var_id(Count);
				C_Int32 _st=0, _cnt=Count;
				GDS_Array_ReadData(varVariant, &_st, &_cnt, &var_id[0], svInt32);
				if (!intersect)
				{
					for (int i=0; i < Count; i++)
						*pArray++ = (set_id.find(var_id[i]) != set_id.end());
				} else {
					for (int i=0; i < Count; i++, pArray++)
						if (*pArray)
							*pArray = (set_id.find(var_id[i]) != set_id.end());
				}
			} else {
				set<string> set_id;
				{
					vector<string> ary;
					numpy_to_string(variant_id, ary);
					set_id.insert(ary.begin(), ary.end());
				}
				vector<string> var_id(Count);
				C_Int32 _st=0, _cnt=Count;
				GDS_Array_ReadData(varVariant, &_st, &_cnt, &var_id[0], svStrUTF8);
				if (!intersect)
				{
					for (int i=0; i < Count; i++)
						*pArray++ = (set_id.find(var_id[i]) != set_id.end());
				} else {
					for (int i=0; i < Count; i++, pArray++)
						if (*pArray)
							*pArray = (set_id.find(var_id[i]) != set_id.end());
				}
			}
		} else
			throw ErrSeqArray("Invalid type of 'variant.id'.");

		// the variant selection changed
		Sel.ClearStructVariant();

		if (verbose)
			printf("# of selected variants: %s\n", PrettyInt(File.VariantSelNum()));

	COREARRAY_CATCH_NONE
}


/// set a working space with selected variant (bool vector or index)
PY_EXPORT PyObject* SEQ_SetSpaceVariant2(PyObject *self, PyObject *args)
{
	int file_id;
	PyObject *var_sel;
	int intersect, verbose;
	if (!PyArg_ParseTuple(args, "iO" BSTR BSTR, &file_id, &var_sel, &intersect, &verbose))
		return NULL;

	COREARRAY_TRY

		CFileInfo &File = GetFileInfo(file_id);
		TSelection &Sel = File.Selection();
		C_BOOL *pArray = Sel.pVariant;
		int Count = File.VariantNum();

		if (numpy_is_bool(var_sel))
		{
			if (!intersect)
			{
				if (numpy_size(var_sel) != (size_t)Count)
					throw ErrSeqArray("Invalid length of 'variant.sel'.");
				memcpy(pArray, numpy_getptr(var_sel), Count);
			} else {
				if (numpy_size(var_sel) != (size_t)File.VariantSelNum())
					throw ErrSeqArray("Invalid length of 'variant' (should be equal to the number of selected variants).");
				C_BOOL *base = (C_BOOL*)numpy_getptr(var_sel);
				for (int i=0; i < Count; i++)
					if (pArray[i]) pArray[i] = ((*base++) != 0);
			}
		} else if (numpy_is_int(var_sel))
		{
			vector<int> idx;
			numpy_to_int32(var_sel, idx);
			size_t N = idx.size();
			if (!intersect)
			{
				for (size_t i=0; i < N; i++)
					if ((idx[i] < 0) || (idx[i] >= Count))
						throw ErrSeqArray("Out of range 'variant'.");
				memset((void*)pArray, 0, Count);
				for (size_t i=0; i < N; i++) pArray[idx[i]] = TRUE;
			} else {
				int Cnt = File.VariantSelNum();
				for (size_t i=0; i < N; i++)
					if ((idx[i] < 0) || (idx[i] >= Cnt))
						throw ErrSeqArray("Out of range 'variant'.");
				vector<int> Idx;
				Idx.reserve(Cnt);
				for (int i=0; i < Count; i++)
					if (pArray[i]) Idx.push_back(i);
				memset((void*)pArray, 0, Count);
				for (size_t i=0; i < N; i++) pArray[Idx[idx[i]]] = TRUE;
			}
		} else if (var_sel == Py_None)
		{
			memset(pArray, TRUE, Count);
		} else
			throw ErrSeqArray("Invalid type of 'variant'.");

		Sel.ClearStructVariant();

		if (verbose)
			printf("# of selected variants: %s\n", PrettyInt(File.VariantSelNum()));

	COREARRAY_CATCH_NONE
}


// ================================================================

/// get a logical vector with the current sample/variant selection
PY_EXPORT PyObject* SEQ_GetSpace(PyObject *self, PyObject *args)
{
	int file_id;
	int sample;
	if (!PyArg_ParseTuple(args, "i" BSTR, &file_id, &sample))
		return NULL;

	COREARRAY_TRY

		CFileInfo &File = GetFileInfo(file_id);
		TSelection &Sel = File.Selection();
		PyObject *rv_ans;
		if (sample)
		{
			size_t n = File.SampleNum();
			rv_ans = numpy_new_bool(n);
			memcpy(numpy_getptr(rv_ans), Sel.pSample, n);
		} else {
			size_t n = File.VariantNum();
			rv_ans = numpy_new_bool(n);
			memcpy(numpy_getptr(rv_ans), Sel.pVariant, n);
		}
		return rv_ans;

	COREARRAY_CATCH_NONE
}


// ===========================================================

inline static C_BOOL *CLEAR_SELECTION(size_t num, C_BOOL *p)
{
	while (num > 0)
	{
		if (*p != FALSE) { num--; *p = FALSE; }
		p ++;
	}
	return p;
}
inline static C_BOOL *SKIP_SELECTION(size_t num, C_BOOL *p)
{
	while (num > 0)
	{
		if (*p != FALSE) num--;
		p ++;
	}
	return p;
}

/// split the selected variants/samples according to multiple processes
PY_EXPORT PyObject* SEQ_SplitSelection(PyObject *self, PyObject *args)
{
	int file_id, proc_idx, proc_ncpu;
	const char *split;
	if (!PyArg_ParseTuple(args, "iiis", &file_id, &proc_idx, &proc_ncpu, &split))
		return NULL;

	COREARRAY_TRY

		CFileInfo &File = GetFileInfo(file_id);
		TSelection &s = File.Selection();

		int SelectCount;
		C_BOOL *sel;
		bool by_variant = false;
		if (strcmp(split, "by.variant") == 0)
		{
			sel = s.pVariant;
			SelectCount = GetNumOfTRUE(sel, File.VariantNum());
			by_variant = true;
		} else if (strcmp(split, "by.sample") == 0)
		{
			sel = s.pSample;
			SelectCount = GetNumOfTRUE(sel, File.SampleNum());
		} else if (strcmp(split, "none") == 0)
		{
			Py_RETURN_NONE;
		} else {
			throw ErrSeqArray("'split' should be 'by.variant', 'by.sample' or 'none'.");
		}

		// split a list
		vector<int> splt(proc_ncpu);
		double avg = (double)SelectCount / proc_ncpu;
		double start = 0;
		for (int i=0; i < proc_ncpu; i++)
		{
			start += avg;
			splt[i] = (int)(start + 0.5);
		}

		int st = 0;
		for (int i=0; i < proc_idx; i++)
		{
			sel = CLEAR_SELECTION(splt[i] - st, sel);
			st = splt[i];
		}
		int ans_n = splt[proc_idx] - st;
		sel = SKIP_SELECTION(ans_n, sel);
		st = splt[proc_idx];
		for (int i=proc_idx+1; i < proc_ncpu; i++)
		{
			sel = CLEAR_SELECTION(splt[i] - st, sel);
			st = splt[i];
		}

		// the selection changed
		if (by_variant) s.ClearStructVariant(); else s.ClearStructSample();

	COREARRAY_CATCH_NONE
}



// ===========================================================
// Module entry
// ===========================================================

extern PyObject* SEQ_GetData(PyObject *self, PyObject *args);
extern PyObject* SEQ_BApply_Variant(PyObject *self, PyObject *args);
extern PyObject* SEQ_Apply_Variant(PyObject *self, PyObject *args);


static PyMethodDef module_methods[] = {
	// file operations
	{ "file_init", (PyCFunction)SEQ_File_Init, METH_VARARGS, NULL },
	{ "file_done", (PyCFunction)SEQ_File_Done, METH_VARARGS, NULL },

	{ "flt_push", (PyCFunction)SEQ_FilterPush, METH_VARARGS, NULL },
	{ "flt_pop", (PyCFunction)SEQ_FilterPop, METH_VARARGS, NULL },
	{ "flt_split", (PyCFunction)SEQ_SplitSelection, METH_VARARGS, NULL },

	{ "set_sample", (PyCFunction)SEQ_SetSpaceSample, METH_VARARGS, NULL },
	{ "set_sample2", (PyCFunction)SEQ_SetSpaceSample2, METH_VARARGS, NULL },
	{ "set_variant", (PyCFunction)SEQ_SetSpaceVariant, METH_VARARGS, NULL },
	{ "set_variant2", (PyCFunction)SEQ_SetSpaceVariant2, METH_VARARGS, NULL },

	{ "get_filter", (PyCFunction)SEQ_GetSpace, METH_VARARGS, NULL },

	// get data
	{ "get_data", (PyCFunction)SEQ_GetData, METH_VARARGS, NULL },
	{ "apply", (PyCFunction)SEQ_BApply_Variant, METH_VARARGS, NULL },
	{ "apply_variant", (PyCFunction)SEQ_Apply_Variant, METH_VARARGS, NULL },

	// end
	{ NULL, NULL, 0, NULL }
};


// Module entry point

#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef ModStruct =
{
	PyModuleDef_HEAD_INIT,
	"PySeqArray.ccall",  // name of module
	"C functions for data manipulation",  // module documentation
	-1,
	module_methods
};

PyMODINIT_FUNC PyInit_ccall()
{
	if (!numpy_init()) return NULL;
	if (Init_GDS_Routines() < 0) return NULL;
#else
PyMODINIT_FUNC initccall()
{
	if (!numpy_init()) return;
	if (Init_GDS_Routines() < 0) return;
#endif

	PyObject *mod;
#if PY_MAJOR_VERSION >= 3
	mod = PyModule_Create(&ModStruct);
	return mod;
#else
	mod = Py_InitModule("PySeqArray.ccall", module_methods);
#endif
}

} // extern "C"
