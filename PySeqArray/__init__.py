# import numpy
import numpy as np
# import os
import os
# import multiprocessing
import multiprocessing as mp
import multiprocessing.pool as pl
# import pygds
import pygds
# import c library
import PySeqArray.ccall as cc
# other ...
from sys import platform
from functools import reduce


## export version number
__version__ = '0.1.0'


# ---------------------------------------------------------------------------
# small value formatters used by VCF export
# ---------------------------------------------------------------------------

def _fmt_num(x):
	"""Format a numeric scalar like R: integers without a trailing '.0'."""
	try:
		xf = float(x)
	except (TypeError, ValueError):
		return str(x)
	if np.isnan(xf):
		return '.'
	if xf == int(xf):
		return str(int(xf))
	return repr(xf)

def _fmt_cell(x):
	"""Format one INFO/FORMAT value (numeric or string); '.' for missing."""
	if x is None:
		return '.'
	if isinstance(x, (bytes, bytearray)):
		x = x.decode('utf-8', 'replace')
	if isinstance(x, str):
		return x if x != '' else '.'
	if isinstance(x, (float, np.floating)):
		return _fmt_num(x)
	if isinstance(x, (int, np.integer)):
		# NA sentinel for int32 in GDS is the minimum int value
		if int(x) == -2147483648:
			return '.'
		return str(int(x))
	return str(x)

def _mk_var_accessor(raw, per_sample):
	"""Build a per-variant (INFO) or per-(variant,sample) (FORMAT) value
	accessor over a GDS annotation field.

	``raw`` is either a flat/2-D numpy array (fixed Number) or a dict
	``{'index': counts-per-variant, 'data': flattened values}`` for
	variable-length fields (R's ``$length``/``$data`` form).
	"""
	if isinstance(raw, dict):
		idx = np.asarray(raw['index']).astype(np.int64)
		data = np.asarray(raw['data'])
		off = np.concatenate(([0], np.cumsum(idx)))
		if per_sample:
			def acc(i, j):
				n = int(idx[i])
				if n == 0:
					return '.'
				seg = np.atleast_1d(data[off[i]:off[i] + n, j])
				return ','.join(_fmt_cell(x) for x in seg)
			return acc
		def acc(i, raw_flag=False):
			n = int(idx[i])
			if n == 0:
				return 0 if raw_flag else None
			seg = np.atleast_1d(data[off[i]:off[i] + n])
			if raw_flag:
				return seg[0]
			return ','.join(_fmt_cell(x) for x in seg)
		return acc
	arr = np.asarray(raw)
	if per_sample:
		def acc(i, j):
			return _fmt_cell(arr[i, j] if arr.ndim >= 2 else arr[i])
		return acc
	def acc(i, raw_flag=False):
		v = arr[i] if arr.ndim >= 1 else arr
		return v if raw_flag else _fmt_cell(v)
	return acc



# ===========================================================================

def seqExample(filename=None):
	"""Example files

	Return a file name in the folder of example data.

	Parameters
	----------
	filename : str
		a file name in the folder of example data, or None for returning the path of example folder

	Returns
	-------
	string

	Examples
	--------
	>>> seqExample('1KG_phase1_release_v3_chr22.gds')
	"""
	import PySeqArray
	s = os.path.dirname(PySeqArray.__file__)
	if filename is None:
		return os.path.join(s, 'data')
	else:
		return os.path.join(s, 'data', filename)



# ===========================================================================

# define internal function using forking
def _proc_fork_func(x):
	i = x[0]; ncpu = x[1]
	file = x[2]; fun = x[3]; param = x[4]; split = x[5]
	cc.flt_split(file.fileid, i, ncpu, split)
	return fun(file, param)

# define a process function
def _proc_func(x):
	i = x[0]; ncpu = x[1]
	fn = x[2]; fun = x[3]; param = x[4]; sel = x[5]; split = x[6]
	import PySeqArray
	import PySeqArray.ccall as cc
	file = PySeqArray.SeqArrayFile()
	file.open(fn, allow_dup=True)
	file.FilterSet2(sel[0], sel[1], verbose=False)
	cc.flt_split(file.fileid, i, ncpu, split)
	return fun(file, param)



# ===========================================================================

class SeqArrayFile(pygds.gdsfile):
	"""
	Class for SeqArray GDS files
	"""

	def __init__(self):
		pygds.gdsfile.__init__(self)

	def __del__(self):
		cc.file_done(self.fileid)
		pygds.gdsfile.__del__(self)


	def create(self, filename, allow_dup=False):
		raise Exception('not supported!')


	def open(self, filename, readonly=True, allow_dup=False):
		"""Open an SeqArray file

		Open an existing file of SeqArray GDS for reading or writing.

		Parameters
		----------
		filename : str
			the file name of a new GDS file to be created
		readonly : bool
			if True, the file is opened read-only; otherwise, it is allowed to write data to the file
		allow_dup : bool
			if True, it is allowed to open a GDS file with read-only mode when it has been opened in the same session

		Returns
		-------
		None

		See Also
		--------
		close: close a SeqArray file
		"""
		pygds.gdsfile.open(self, filename, readonly, allow_dup)
		cc.file_init(self.fileid)
		# TODO: file checking


	def close(self):
		"""Close a SeqArray file

		Close a SeqArray GDS file.

		Returns
		-------
		None

		See Also
		--------
		open : open an existing SeqArray file
		"""
		cc.file_done(self.fileid)
		pygds.gdsfile.close(self)


	def FilterSet(self, sample_id=None, variant_id=None, intersect=False, verbose=True):
		"""Set a filter

		Set a filter to sample and/or variant with IDs.

		Parameters
		----------
		sample_id : str
			sample id to be selected
		variant_id : bool
			variant id to be selected
		intersect : bool
			if False, the candidate variants for selection are all possible variants (by default);
			if True, the candidate variants are from the selected variants defined via the previous call
		verbose : bool
			if True, show information

		Returns
		-------
		None

		See Also
		--------
		FilterReset : reset the filter
		"""
		s = not sample_id is None
		v = not variant_id is None
		if s or v:
			if s:
				cc.set_sample(self.fileid, sample_id, intersect, verbose)
			if v:
				cc.set_variant(self.fileid, variant_id, intersect, verbose)


	def FilterSet2(self, sample=None, variant=None, intersect=False, verbose=True):
		"""Set a filter

		Set a filter to sample and/or variant with a bool vector or an index vector.

		Parameters
		----------
		sample : vector, range
			a bool vector, an indexing vecot for selecting samples or a range object
		variant : vector, range
			a bool vector, an indexing vecot for selecting variants or a range object
		intersect : bool
			if False, the candidate variants for selection are all possible variants (by default);
			if True, the candidate variants are from the selected variants defined via the previous call
		verbose : bool
			if True, show information

		Returns
		-------
		None

		See Also
		--------
		FilterSet : set a filter
		FilterReset : reset the filter
		"""
		if not sample is None:
			cc.set_sample2(self.fileid, sample, intersect, verbose)
		if not variant is None:
			cc.set_variant2(self.fileid, variant, intersect, verbose)


	def FilterReset(self, sample=True, variant=True, verbose=True):
		"""Reset the filter

		Clear the existing filter on sample and/or variant.

		Parameters
		----------
		sample : str
			if True, reset the filter of sample
		variant : bool
			if True, reset the filter of variant
		verbose : bool
			if True, show information

		Returns
		-------
		None

		See Also
		--------
		FilterSet : set a filter
		"""
		if sample:
			cc.set_sample(self.fileid, None, False, verbose)
		if variant:
			cc.set_variant(self.fileid, None, False, verbose)


	def FilterPush(reset=True):
		"""Push a filter

		Push the current filter to the stack.

		Parameters
		----------
		reset : bool
			if True, reset the filter of sample and variant after pushing

		Returns
		-------
		None

		See Also
		--------
		FilterPop : recover the last filter
		"""
		cc.flt_push(self.fileid, reset)


	def FilterPop():
		"""Pop a filter

		Pop or recover the last filter in the stack.

		Returns
		-------
		None

		See Also
		--------
		FilterPush : push the current filter to the stack
		"""
		cc.flt_pop(self.fileid)


	def FilterGet(self, sample=True):
		"""Get a sample/variant filter

		Get a sample or variant filter.

		Parameters
		----------
		sample : bool
			If True, return the sample filter; otherwise, return the variant filter

		Returns
		-------
		A numpy object (a bool vector)

		See Also
		--------
		FilterSet : set a filter
		"""
		return(cc.get_filter(self.fileid, sample))


	def GetData(self, name):
		"""Get data

		Get data from a SeqArray file with a given variable name and a sample/variant filter

		Parameters
		----------
		name : str
			the variable name

		Returns
		-------
		a numpy array object

		Notes
		-----
		Variable-length INFO/FORMAT fields are returned as a dict
		``{'index':..., 'data':...}``. The sparse selectors ``$dosage_sp`` /
		``$dosage_sp2`` are returned as a :class:`scipy.sparse.csc_matrix`
		(sample x variant); if SciPy is unavailable the raw CSC dict
		``{'sparse':'csc','data','indices','indptr','shape'}`` is returned.

		See Also
		--------
		FilterSet : set a filter
		"""
		d = cc.get_data(self.fileid, name)
		if isinstance(d, dict) and d.get('sparse') == 'csc':
			try:
				from scipy.sparse import csc_matrix
				return csc_matrix((d['data'], d['indices'], d['indptr']),
					shape=d['shape'])
			except ImportError:
				return d
		return d


	def Apply(self, name, fun, param=None, asis='none', bsize=1024, verbose=False):
		"""Apply function over array margins

		Apply a user-defined function to margins of genotypes and annotations via blocking

		Parameters
		----------
		name : str, list
			the variable name, or a list of variable names
		fun : function
			the user-defined function
		param: object
			the parameter passed to the user-defined function if it is not None
		asis : str
			'none', no return; 'list', a list of the returned values from the user-defined function;
			'unlist', flatten the returned values from the user-defined function
		bsize : int
			block size
		verbose : bool
			show progress information if True

		Returns
		-------
		None, a list or a numpy array object

		See Also
		--------
		FilterSet : set a filter
		"""
		v = cc.apply(self.fileid, name, fun, param, asis, bsize, verbose)
		if asis == 'unlist':
			v = np.hstack(v)
		return(v)


	def RunParallel(self, fun, param=None, ncpu=0, split='by.variant', combine='unlist'):
		"""Apply Functions in Parallel

		Apply a user-defined function in parallel over array margins

		Parameters
		----------
		fun : function
			the user-defined function
		param : object
			the parameter passed to the user-defined function if it is not None
		ncpu : int
			the number of cores or an instance of 'multiprocessing.pool.Pool';
			0 to use the number of cores minus 1
		split : str
			'by.variant', 'by.sample', 'none': split the dataset by variant or sample according to multiple processes, or "none" for no split
		combine : str, function
			'none', no return; 'list', a list of the returned values from the user-defined function;
			'unlist', flatten the returned values from the user-defined function

		Returns
		-------
		None, a list or a numpy array object
		"""
		# check
		if not isinstance(ncpu, (int, float)):
			raise ValueError('`ncpu` should be a numeric value.')
		if not (combine is None or isinstance(combine, str) or callable(combine)):
			raise ValueError('`combine` should be None, a string or a function.')
		if ncpu <= 0:
			ncpu = max(1, mp.cpu_count() - 1)
		ncpu = int(ncpu)
		# serial fast-path
		if ncpu <= 1:
			return fun(self, param)
		# parallel: fork-based, like R's mclapply -- the child runs the
		# closure directly in inherited memory (no function pickling), each
		# child reopens its own file handle to get an independent fd, applies
		# its split, runs `fun`, and pipes back the (picklable) result.
		v = self._fork_apply(fun, param, ncpu, split)
		# combine
		if combine is None or combine == 'none':
			return None
		elif combine == 'unlist':
			return np.hstack(v)
		elif combine == 'list':
			return v
		elif callable(combine):
			return reduce(combine, v)
		else:
			raise ValueError('`combine` is invalid.')

	def _fork_apply(self, fun, param, ncpu, split):
		import os, pickle
		fn = self.filename
		sel0 = np.asarray(self.FilterGet(True))
		sel1 = np.asarray(self.FilterGet(False))
		readers = []
		pids = []
		for i in range(ncpu):
			r_fd, w_fd = os.pipe()
			pid = os.fork()
			if pid == 0:
				# ---- child ----
				os.close(r_fd)
				try:
					child = SeqArrayFile()
					child.open(fn, allow_dup=True)
					child.FilterSet2(sel0, sel1, verbose=False)
					cc.flt_split(child.fileid, i, ncpu, split)
					res = fun(child, param)
					payload = pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL)
				except BaseException as e:  # noqa
					import traceback
					payload = pickle.dumps(('__pyseq_error__',
						traceback.format_exc()))
				with os.fdopen(w_fd, 'wb') as wf:
					wf.write(payload)
				os._exit(0)
			else:
				# ---- parent ----
				os.close(w_fd)
				readers.append(r_fd)
				pids.append(pid)
		# collect results in worker order
		out = []
		for r_fd in readers:
			chunks = []
			with os.fdopen(r_fd, 'rb') as rf:
				while True:
					b = rf.read(1 << 20)
					if not b:
						break
					chunks.append(b)
			out.append(pickle.loads(b''.join(chunks)) if chunks else None)
		for pid in pids:
			os.waitpid(pid, 0)
		for res in out:
			if isinstance(res, tuple) and len(res) == 2 and \
					res[0] == '__pyseq_error__':
				raise RuntimeError('Error in parallel worker:\n' + res[1])
		return out


	####  Summary statistics  ####

	# missing genotype value (NA) for the uint8 genotype array
	_GENO_NA = 255

	def NumAllele(self):
		"""Number of alleles per variant.

		Returns
		-------
		numpy.ndarray (int32), one value per selected variant (>= 1)

		See Also
		--------
		AlleleFreq, AlleleCount
		"""
		al = np.asarray(self.GetData('allele'))
		return np.fromiter((s.count(',') + 1 for s in al), dtype=np.int32,
			count=len(al))

	def AlleleFreq(self, ref=0, ncpu=1, verbose=False):
		"""Allele frequencies per variant.

		Parameters
		----------
		ref : int, array-like or None
			the reference allele index: 0 for the first allele (default);
			an integer vector specifying a per-variant reference allele index;
			a string vector specifying a per-variant reference allele; or
			None to return the frequencies of all alleles as a list per variant.
		ncpu : int
			number of processes (reserved; computation is vectorised)
		verbose : bool
			show progress if True

		Returns
		-------
		numpy.ndarray (float64) of per-variant frequencies, or a list of
		arrays (one per variant) when ``ref`` is None.

		See Also
		--------
		AlleleCount, Missing, NumAllele
		"""
		return self._allele_stat(ref, freq=True, verbose=verbose)

	def AlleleCount(self, ref=None, ncpu=1, verbose=False):
		"""Allele counts per variant.

		Parameters
		----------
		ref : int, array-like or None
			see :meth:`AlleleFreq`; None (default) returns the count of every
			allele as a list per variant.

		Returns
		-------
		numpy.ndarray (int32) of per-variant counts, or a list of arrays
		(one per variant) when ``ref`` is None.

		See Also
		--------
		AlleleFreq
		"""
		return self._allele_stat(ref, freq=False, verbose=verbose)

	def _allele_stat(self, ref, freq, verbose=False):
		# Per-variant allele frequency / count over the genotype margin.
		# ref: None -> all alleles (list); int scalar; per-variant int vector;
		#      per-variant string vector (matched against $allele).
		na = self._GENO_NA
		nall = self.NumAllele()
		alleles = None
		ridx = None
		if ref is None:
			pass
		elif isinstance(ref, str) or (hasattr(ref, '__len__') and
				len(ref) > 0 and isinstance(np.asarray(ref).ravel()[0], str)):
			alleles = np.asarray(self.GetData('allele'))
		elif isinstance(ref, (int, np.integer)):
			ridx = None  # scalar
		else:
			ridx = np.asarray(ref, dtype=np.int64)  # per-variant index

		state = {'i': 0}
		refarr = None if isinstance(ref, str) else \
			(np.asarray(ref) if (ref is not None and hasattr(ref, '__len__')) else None)

		def fc(g):
			# g: (nVar, nSample, ploidy) uint8 block
			nv = g.shape[0]
			g2 = g.reshape(nv, -1)
			nonmiss = (g2 != na)
			ntot = nonmiss.sum(axis=1).astype(np.int64)
			i0 = state['i']; state['i'] += nv
			if ref is None:
				# all alleles per variant -> list
				out = []
				for k in range(nv):
					m = int(nall[i0 + k])
					row = g2[k][nonmiss[k]]
					cnt = np.bincount(row, minlength=m)[:m].astype(np.int64)
					if freq:
						s = cnt.sum()
						out.append((cnt / s) if s > 0 else
							np.full(m, np.nan))
					else:
						out.append(cnt.astype(np.int32))
				return out
			# single reference allele per variant
			if alleles is not None:
				# resolve string allele -> index within each variant
				aidx = np.empty(nv, dtype=np.int64)
				for k in range(nv):
					opts = alleles[i0 + k].split(',')
					try:
						aidx[k] = opts.index(refarr[i0 + k])
					except ValueError:
						aidx[k] = -1
			elif refarr is not None:
				aidx = refarr[i0:i0 + nv].astype(np.int64)
			else:
				aidx = np.zeros(nv, dtype=np.int64)
			cnt = (g2 == aidx[:, None]) & nonmiss
			cnt = cnt.sum(axis=1).astype(np.int64)
			cnt[aidx < 0] = 0
			if freq:
				with np.errstate(invalid='ignore', divide='ignore'):
					res = np.where(ntot > 0, cnt / ntot, np.nan)
				return res.astype(np.float64)
			else:
				return cnt.astype(np.int32)

		parts = self.Apply('genotype', fc, asis='list', verbose=verbose)
		if ref is None:
			out = []
			for p in parts:
				out.extend(p)
			return out
		return np.hstack(parts) if len(parts) else np.array([], dtype=
			np.float64 if freq else np.int32)

	def Missing(self, per_variant=True, verbose=False):
		"""Missing genotype rate.

		Parameters
		----------
		per_variant : bool
			if True (default) return the missing rate per variant; otherwise
			return the missing rate per sample.

		Returns
		-------
		numpy.ndarray (float64)
		"""
		na = self._GENO_NA
		if per_variant:
			def fc(g):
				nv = g.shape[0]
				g2 = g.reshape(nv, -1)
				m = (g2 == na).sum(axis=1).astype(np.float64)
				return m / g2.shape[1]
			parts = self.Apply('genotype', fc, asis='list', verbose=verbose)
			return np.hstack(parts) if len(parts) else np.array([],
				dtype=np.float64)
		else:
			# per-sample: accumulate missing counts across all variant blocks
			acc = {'miss': None, 'n': 0}
			def fc(g):
				nv, ns, ploidy = g.shape
				miss = (g == na).reshape(nv, ns, ploidy).sum(axis=(0, 2))
				if acc['miss'] is None:
					acc['miss'] = miss.astype(np.int64)
				else:
					acc['miss'] += miss
				acc['n'] += nv * ploidy
				return None
			self.Apply('genotype', fc, asis='none', verbose=verbose)
			if acc['miss'] is None:
				return np.array([], dtype=np.float64)
			return acc['miss'].astype(np.float64) / acc['n']


	####  Counts and chromosome/position filters  ####

	def NumVariant(self, selected=False):
		"""Total number of variants, or the number selected if ``selected``."""
		m = cc.get_filter(self.fileid, False)
		m = np.asarray(m)
		return int(m.sum()) if selected else int(m.size)

	def NumSample(self, selected=False):
		"""Total number of samples, or the number selected if ``selected``."""
		m = cc.get_filter(self.fileid, True)
		m = np.asarray(m)
		return int(m.sum()) if selected else int(m.size)

	def SetFilterChrom(self, include=None, frm_bp=None, to_bp=None,
			intersect=False, verbose=True):
		"""Set the variant filter by chromosome and optional position range.

		Parameters
		----------
		include : str or array-like
			one or more chromosome codes to keep; None keeps all chromosomes
		frm_bp, to_bp : int or array-like
			inclusive base-pair range (matched element-wise with ``include``
			when both are vectors); None means unbounded
		intersect : bool
			if True, restrict to the currently selected variants
		"""
		if not intersect:
			self.FilterReset(sample=False, variant=True, verbose=False)
		chrom = np.asarray(self.GetData('chromosome'))
		pos = np.asarray(self.GetData('position'))
		if include is None:
			mask = np.ones(chrom.shape, dtype=bool)
		else:
			inc = np.atleast_1d(np.asarray(include).astype(str))
			mask = np.isin(chrom.astype(str), inc)
		if frm_bp is not None:
			mask &= (pos >= int(np.atleast_1d(frm_bp)[0]))
		if to_bp is not None:
			mask &= (pos <= int(np.atleast_1d(to_bp)[0]))
		self.FilterSet2(variant=mask, intersect=intersect, verbose=verbose)

	def SetFilterPos(self, chrom, pos, intersect=False, verbose=True):
		"""Select variants matching given (chromosome, position) pairs.

		Parameters
		----------
		chrom : array-like
			chromosome code(s)
		pos : array-like
			base-pair position(s), paired element-wise with ``chrom``
		"""
		if not intersect:
			self.FilterReset(sample=False, variant=True, verbose=False)
		C = np.asarray(self.GetData('chromosome')).astype(str)
		P = np.asarray(self.GetData('position'))
		want = set(zip(np.atleast_1d(np.asarray(chrom).astype(str)).tolist(),
			[int(x) for x in np.atleast_1d(pos)]))
		mask = np.fromiter((( (C[i], int(P[i])) in want) for i in range(C.size)),
			dtype=bool, count=C.size)
		self.FilterSet2(variant=mask, intersect=intersect, verbose=verbose)


	####  Summary and digest  ####

	def Summary(self, varname=None):
		"""Summarise the dataset or a data variable (R: seqSummary).

		With ``varname`` None, return a dict describing the selection (number
		of samples/variants, ploidy, allele counts). Otherwise summarise the
		named variable.
		"""
		if varname in (None, '', 'genotype'):
			ns = self.NumSample(selected=True)
			nv = self.NumVariant(selected=True)
			nall = self.NumAllele()
			tab = {}
			if nall.size:
				u, c = np.unique(nall, return_counts=True)
				tab = dict(zip(u.tolist(), c.tolist()))
			return {
				'num_sample': ns,
				'num_variant': nv,
				'num_sample_total': self.NumSample(),
				'num_variant_total': self.NumVariant(),
				'allele_table': tab,
			}
		else:
			d = np.asarray(self.GetData(varname))
			return {'name': varname, 'shape': tuple(d.shape),
				'dtype': str(d.dtype)}

	def Digest(self, name, algo='md5'):
		"""Hash the data of a variable for the current selection.

		Native digest using :mod:`hashlib`. Unlike R's ``seqDigest`` the byte
		layout follows numpy's C-order serialisation, so digests are stable
		within PySeqArray but not byte-identical to R.
		"""
		import hashlib
		h = hashlib.new(algo)
		d = self.GetData(name)
		arr = np.asarray(d)
		if arr.dtype == object:  # string/object array
			for s in arr.ravel():
				h.update(str(s).encode('utf-8'))
				h.update(b'\x00')
		else:
			h.update(np.ascontiguousarray(arr).tobytes())
		return h.hexdigest()


	####  VCF export  ####

	def GDS2VCF(self, filename, info_fields=None, fmt_fields=None,
			verbose=False):
		"""Export the current selection to a VCF file (R: seqGDS2VCF).

		Writes plain text, or gzip-compressed when ``filename`` ends in
		``.gz``. INFO/FORMAT annotation fields present in the GDS are included.

		Parameters
		----------
		filename : str
			output path; ``.gz`` suffix selects gzip output
		info_fields, fmt_fields : list of str or None
			restrict to these annotation/info and annotation/format fields;
			None (default) auto-detects all available fields
		"""
		import gzip, datetime
		nv = self.NumVariant(selected=True)
		ns = self.NumSample(selected=True)

		# --- column data ---
		chrom = np.asarray(self.GetData('chromosome')).astype(str)
		pos = np.asarray(self.GetData('position'))
		vid = np.asarray(self.GetData('annotation/id')).astype(str)
		allele = np.asarray(self.GetData('allele')).astype(str)
		try:
			qual = np.asarray(self.GetData('annotation/qual'))
		except Exception:
			qual = None
		try:
			flt = np.asarray(self.GetData('annotation/filter')).astype(str)
		except Exception:
			flt = None
		samp = np.asarray(self.GetData('sample.id')).astype(str)

		# REF / ALT split from allele string "ref,alt1,alt2"
		ref = np.empty(nv, dtype=object)
		alt = np.empty(nv, dtype=object)
		for i in range(nv):
			parts = allele[i].split(',', 1)
			ref[i] = parts[0]
			alt[i] = parts[1] if len(parts) > 1 and parts[1] != '' else '.'

		# INFO fields (auto-detect children of annotation/info). Each becomes
		# a per-variant accessor; variable-length fields arrive as a dict
		# {'index': counts-per-variant, 'data': flattened values}.
		info_names = info_fields if info_fields is not None else \
			self._list_children('annotation/info')
		info_proc = []   # (name, meta, is_flag, accessor(i)->str|None)
		for nm in info_names:
			try:
				raw = self.GetData('annotation/info/' + nm)
			except Exception:
				continue
			meta = self._vcf_field_meta('info', nm)
			is_flag = meta.get('Type') == 'Flag'
			info_proc.append((nm, meta, is_flag,
				_mk_var_accessor(raw, per_sample=False)))

		# FORMAT fields (besides GT): per-(variant,sample) accessor.
		fmt_names = fmt_fields if fmt_fields is not None else \
			self._list_children('annotation/format')
		fmt_proc = []    # (name, meta, accessor(i,j)->str)
		for nm in fmt_names:
			try:
				raw = self.GetData('annotation/format/' + nm)
			except Exception:
				continue
			meta = self._vcf_field_meta('format', nm)
			fmt_proc.append((nm, meta,
				_mk_var_accessor(raw, per_sample=True)))

		# genotype & phase
		geno = np.asarray(self.GetData('genotype'))   # (nv, ns, ploidy)
		try:
			phase = np.asarray(self.GetData('phase'))  # (nv, ns)
		except Exception:
			phase = None

		# --- header ---
		fileformat = self._description_attr('vcf.fileformat') or 'VCFv4.0'
		today = datetime.date.today().strftime('%Y%m%d')
		H = []
		H.append('##fileformat=' + fileformat)
		H.append('##fileDate=' + today)
		H.append('##source=SeqArray_Format_v1.0')
		for nm, m, _isf, _a in info_proc:
			H.append('##INFO=<ID=%s,Number=%s,Type=%s,Description="%s">' % (
				nm, m.get('Number', '.'), m.get('Type', 'String'),
				m.get('Description', '')))
		for lv, desc in self._filter_levels():
			# R's seqGDS2VCF does not quote FILTER descriptions
			H.append('##FILTER=<ID=%s,Description=%s>' % (lv, desc))
		H.append('##FORMAT=<ID=GT,Number=1,Type=String,Description=Genotype>')
		for nm, m, _a in fmt_proc:
			H.append('##FORMAT=<ID=%s,Number=%s,Type=%s,Description="%s">' % (
				nm, m.get('Number', '.'), m.get('Type', 'String'),
				m.get('Description', '')))
		cols = ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO']
		fmt_keys = ':'.join(['GT'] + [nm for nm, _m, _a in fmt_proc])
		if ns > 0:
			cols += ['FORMAT'] + samp.tolist()
		H.append('\t'.join(cols))

		op = gzip.open(filename, 'wt', newline='\n') if \
			str(filename).endswith('.gz') else open(filename, 'wt', newline='\n')
		with op as out:
			out.write('\n'.join(H) + '\n')
			NA = self._GENO_NA
			for i in range(nv):
				q = '.' if (qual is None or np.isnan(qual[i])) else \
					_fmt_num(qual[i])
				fv = '.' if flt is None else (flt[i] if flt[i] else '.')
				# INFO
				items = []
				for nm, _m, is_flag, acc in info_proc:
					if is_flag:
						v = acc(i, raw_flag=True)
						if v:
							items.append(nm)
					else:
						s = acc(i)
						if s is not None and s != '' and s != '.':
							items.append('%s=%s' % (nm, s))
				info = ';'.join(items) if items else '.'
				row = [chrom[i], str(int(pos[i])), vid[i] if vid[i] else '.',
					ref[i], alt[i], q, fv, info]
				if ns > 0:
					row.append(fmt_keys)
					gi = geno[i]            # (ns, ploidy)
					pi = phase[i] if phase is not None else None
					for j in range(ns):
						sep = '|' if (pi is not None and pi[j]) else '/'
						cell = sep.join('.' if int(x) == NA else str(int(x))
							for x in gi[j])
						for nm, _m, acc in fmt_proc:
							cell += ':' + acc(i, j)
						row.append(cell)
				out.write('\t'.join(row) + '\n')
		if verbose:
			print('Done: %d variant(s), %d sample(s) -> %s' %
				(nv, ns, filename))
		return filename

	# --- small GDS-introspection helpers (use the pygds node API) ---
	def _list_children(self, path):
		try:
			n = self.index(path)
			return list(n.ls()) if n is not None else []
		except Exception:
			return []

	def _description_attr(self, key):
		try:
			n = self.index('description')
			a = n.getattr() if n is not None else {}
			return a.get(key)
		except Exception:
			return None

	def _filter_levels(self):
		try:
			n = self.index('annotation/filter')
			a = n.getattr()
			levels = a.get('R.levels')
			desc = a.get('Description', '')
			if levels is None:
				return [('PASS', '')]
			if isinstance(levels, str):
				levels = [levels]
			if isinstance(desc, str):
				desc = [desc] * len(levels)
			return list(zip([str(x) for x in levels],
				[str(d) for d in desc] + [''] * len(levels)))
		except Exception:
			return [('PASS', '')]

	def _vcf_field_meta(self, kind, nm):
		try:
			n = self.index('annotation/%s/%s' % (kind, nm))
			a = n.getattr()
			return {'Number': a.get('Number', '.'),
				'Type': a.get('Type', 'String'),
				'Description': a.get('Description', '')}
		except Exception:
			return {}

# ===========================================================================
# R-compatible module-level seq* functions
# ===========================================================================

def seqOpen(filename, readonly=True, allow_dup=False):
	"""Open a SeqArray GDS file (R: seqOpen). Returns a SeqArrayFile."""
	f = SeqArrayFile()
	f.open(filename, readonly=readonly, allow_dup=allow_dup)
	return f

def seqClose(gdsfile):
	"""Close a SeqArray GDS file (R: seqClose)."""
	gdsfile.close()

def seqExampleFileName(filename='1KG_phase1_release_v3_chr22.gds'):
	"""Path to a bundled example file (R: seqExampleFileName)."""
	return seqExample(filename)

def seqGetData(gdsfile, name):
	"""Get data for the current selection (R: seqGetData)."""
	return gdsfile.GetData(name)

def seqAlleleFreq(gdsfile, ref_allele=0, verbose=False):
	"""Allele frequencies per variant (R: seqAlleleFreq)."""
	return gdsfile.AlleleFreq(ref=ref_allele, verbose=verbose)

def seqAlleleCount(gdsfile, ref_allele=None, verbose=False):
	"""Allele counts per variant (R: seqAlleleCount)."""
	return gdsfile.AlleleCount(ref=ref_allele, verbose=verbose)

def seqMissing(gdsfile, per_variant=True, verbose=False):
	"""Missing genotype rate (R: seqMissing)."""
	return gdsfile.Missing(per_variant=per_variant, verbose=verbose)

def seqNumAllele(gdsfile):
	"""Number of alleles per variant (R: seqNumAllele)."""
	return gdsfile.NumAllele()

def seqApply(gdsfile, name, fun, param=None, asis='none', bsize=1024,
		verbose=False):
	"""Apply a function over blocks of variants (R: seqBlockApply)."""
	return gdsfile.Apply(name, fun, param=param, asis=asis, bsize=bsize,
		verbose=verbose)

seqBlockApply = seqApply

def seqParallel(gdsfile, fun, param=None, ncpu=0, split='by.variant',
		combine='unlist'):
	"""Apply a function in parallel over the dataset (R: seqParallel)."""
	return gdsfile.RunParallel(fun, param=param, ncpu=ncpu, split=split,
		combine=combine)

def seqSetFilter(gdsfile, sample_id=None, variant_id=None, sample=None,
		variant=None, intersect=False, verbose=True):
	"""Set sample/variant filter (R: seqSetFilter).

	Use ``sample_id``/``variant_id`` for IDs, or ``sample``/``variant`` for
	boolean/index vectors.
	"""
	if sample_id is not None or variant_id is not None:
		gdsfile.FilterSet(sample_id=sample_id, variant_id=variant_id,
			intersect=intersect, verbose=verbose)
	if sample is not None or variant is not None:
		gdsfile.FilterSet2(sample=sample, variant=variant,
			intersect=intersect, verbose=verbose)

def seqResetFilter(gdsfile, sample=True, variant=True, verbose=True):
	"""Reset sample/variant filter (R: seqResetFilter)."""
	gdsfile.FilterReset(sample=sample, variant=variant, verbose=verbose)

def seqGetFilter(gdsfile):
	"""Return (sample_mask, variant_mask) boolean arrays (R: seqGetFilter)."""
	return (gdsfile.FilterGet(True), gdsfile.FilterGet(False))

def seqSetFilterChrom(gdsfile, include=None, frm_bp=None, to_bp=None,
		intersect=False, verbose=True):
	"""Set variant filter by chromosome/position (R: seqSetFilterChrom)."""
	gdsfile.SetFilterChrom(include=include, frm_bp=frm_bp, to_bp=to_bp,
		intersect=intersect, verbose=verbose)

def seqSetFilterPos(gdsfile, chrom, pos, intersect=False, verbose=True):
	"""Select variants at given (chrom, pos) pairs (R: seqSetFilterPos)."""
	gdsfile.SetFilterPos(chrom, pos, intersect=intersect, verbose=verbose)

def seqSummary(gdsfile, varname=None):
	"""Summarise the dataset or a data variable (R: seqSummary)."""
	return gdsfile.Summary(varname)

def seqDigest(gdsfile, name, algo='md5'):
	"""Native hashlib digest of a variable (R: seqDigest, see note)."""
	return gdsfile.Digest(name, algo=algo)

def seqGDS2VCF(gdsfile, vcf_fn, info_fields=None, fmt_fields=None,
		verbose=False):
	"""Export the current selection to a VCF file (R: seqGDS2VCF)."""
	return gdsfile.GDS2VCF(vcf_fn, info_fields=info_fields,
		fmt_fields=fmt_fields, verbose=verbose)


# ===========================================================================
# VCF import (native parser)
# ===========================================================================

from PySeqArray._vcf_import import (  # noqa: E402,F401
	seqVCF2GDS, seqVCF_Header, seqVCF_SampID)


# ===========================================================================
# Streaming numeric writer (block append; no VCF text)
# ===========================================================================

from PySeqArray._writer import (  # noqa: E402,F401
	SeqVarGDSWriter, seqCreateGDS, seqAppendVariants, seqCloseGDS)


# ===========================================================================
# Format converters (PLINK BED, subset export)
# ===========================================================================

from PySeqArray._convert import (  # noqa: E402,F401
	seqGDS2BED, seqBED2GDS, seqExport, seqGDS2SNP, seqSNP2GDS, seqMerge)
