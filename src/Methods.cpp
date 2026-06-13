// ===========================================================
//
// Methods.cpp: statistic methods for the PySeqArray package
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

// ---------------------------------------------------------------------------
// In SeqArray the per-variant statistics (allele frequency / count, missing
// rate, etc.) are C callbacks (FC_*) invoked by seqApply/seqParallel. In the
// native PySeqArray these statistics are implemented in the Python layer
// (PySeqArray/__init__.py) using vectorised NumPy over the block-apply engine,
// and are R-validated. This translation unit is therefore intentionally light;
// it is retained as a build slot and a home for future C-level accelerators
// (which can reuse the vec_* SIMD helpers in vectorization.h).
// ---------------------------------------------------------------------------

#include "Index.h"

namespace PySeqArray
{
	// (no native FC_* statistics are exported yet; see note above)
}
