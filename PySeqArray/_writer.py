# ===========================================================================
#
# _writer.py: streaming numeric writer for SeqArray GDS (milestone M2)
#
# Build a SeqArray-format GDS by appending blocks of K variants at a time,
# from numeric numpy arrays -- no VCF text in the loop.  The genotype Bit2
# packing is done by pygds' storage='bit2'; here we just stream blocks into
# appendable, compressed nodes and finalize at close.  The on-disk node layout
# matches _vcf_import._write_gds, so the result is a valid SeqArray GDS that R
# SeqArray / PySeqArray read identically.
#
# Public API:
#     w = seqCreateGDS(path, sample_id, ...)         # or SeqVarGDSWriter(...)
#     seqAppendVariants(w, chrom, pos, allele, geno, ...)   # call N times
#     seqCloseGDS(w)
#
# Copyright (C) 2026  Xiuwen Zheng
# GPLv3
# ===========================================================================

import numpy as np
import pygds

from PySeqArray._vcf_import import _TYPE_STORAGE, _NA_INT32, _GENO_NA


def _empty_1d(storage):
	if storage == 'int32':
		return np.empty(0, np.int32)
	if storage in ('float64', 'float32'):
		return np.empty(0, np.float64 if storage == 'float64' else np.float32)
	return []  # string


def _empty_2d(storage, nsamp):
	if storage == 'int32':
		return np.empty((0, nsamp), np.int32)
	if storage in ('float64', 'float32'):
		return np.empty((0, nsamp), np.float64 if storage == 'float64'
			else np.float32)
	return np.empty((0, nsamp), dtype=object)  # string


def _coerce_1d(v, storage, k):
	"""length-k vector for a fixed (Number=1) field; None -> all missing."""
	if storage == 'int32':
		if v is None:
			return np.full(k, _NA_INT32, np.int32)
		return np.asarray(v, np.int32)
	if storage in ('float64', 'float32'):
		dt = np.float64 if storage == 'float64' else np.float32
		if v is None:
			return np.full(k, np.nan, dt)
		return np.asarray(v, dt)
	# string
	if v is None:
		return ['' for _ in range(k)]
	return [('' if x is None else str(x)) for x in v]


def _coerce_2d(v, storage, k, nsamp):
	"""(k, nsamp) matrix for a fixed (Number=1) FORMAT field."""
	if storage == 'int32':
		if v is None:
			return np.full((k, nsamp), _NA_INT32, np.int32)
		return np.asarray(v, np.int32).reshape(k, nsamp)
	if storage in ('float64', 'float32'):
		dt = np.float64 if storage == 'float64' else np.float32
		if v is None:
			return np.full((k, nsamp), np.nan, dt)
		return np.asarray(v, dt).reshape(k, nsamp)
	# string
	if v is None:
		return np.full((k, nsamp), '', dtype=object)
	return np.asarray(v, dtype=object).reshape(k, nsamp)


def _np_dtype(storage):
	if storage == 'int32':
		return np.int32
	if storage == 'float64':
		return np.float64
	if storage == 'float32':
		return np.float32
	return object


def _set_attrs(node, fld):
	node.putattr('Number', str(fld.get('Number', '.')))
	node.putattr('Type', str(fld.get('Type', 'String')))
	node.putattr('Description', str(fld.get('Description', '')))


class SeqVarGDSWriter:
	"""Streaming writer for a SeqArray-format GDS file.

	Parameters
	----------
	path : str
		output GDS path
	sample_id : sequence of str
		sample identifiers (fixes the sample dimension)
	ploidy : int
		genotype ploidy (default 2)
	compress : str
		GDS compression for every node (default 'LZMA_RA')
	info_fields : dict, optional
		name -> {'Number','Type','Description'}; INFO annotation fields.
		Fields with Number=='1' (or Flag) are fixed; others are
		variable-length (a hidden @<name> index of per-variant counts).
	format_fields : dict, optional
		name -> {'Number','Type','Description'}; per-genotype FORMAT fields.
		Only Number=='1' is supported in this milestone.
	filter_levels : sequence of str, optional
		FILTER factor levels (default ['PASS']).
	fileformat : str
		stored in description/vcf.fileformat.
	"""

	def __init__(self, path, sample_id, ploidy=2, compress='LZMA_RA',
			info_fields=None, format_fields=None, filter_levels=None,
			fileformat='VCFv4.3'):
		self.nSamp = len(sample_id)
		self.ploidy = int(ploidy)
		self.C = compress
		self.fileformat = fileformat
		self.filter_levels = list(filter_levels) if filter_levels else ['PASS']
		self._nvar = 0

		f = pygds.gdsfile()
		f.create(path)
		self.f = f
		r = f.root()
		self._r = r
		r.putattr('FileFormat', 'SEQ_ARRAY')
		r.putattr('FileVersion', 'v1.0')

		# fixed sample dimension
		r.add('sample.id', list(sample_id), storage='string', compress=self.C,
			closezip=True)

		# 1-D appendable site nodes.  NB: a compressed node must be created
		# with an empty initial write (closezip=False) -- creating it with
		# val=None leaves the compression stream uninitialised and corrupts
		# on read after append.
		self.n_vid = r.add('variant.id', np.empty(0, np.int32),
			storage='int32', compress=self.C, closezip=False)
		self.n_pos = r.add('position', np.empty(0, np.int32),
			storage='int32', compress=self.C, closezip=False)
		self.n_chrom = r.add('chromosome', [], storage='string',
			compress=self.C, closezip=False)
		self.n_allele = r.add('allele', [], storage='string',
			compress=self.C, closezip=False)

		# genotype (3-D: variant x sample x ploidy -> pre-set inner dims)
		g = r.addfolder('genotype')
		self._g = g
		g.putattr('VariableName', 'GT')
		g.putattr('Description', 'Genotype')
		self.n_geno = g.add('data', np.empty((0, self.nSamp, self.ploidy),
			np.uint8), storage='bit2', compress=self.C, closezip=False)
		self.n_gidx = g.add('@data', np.empty(0, np.uint8), storage='uint8',
			compress=self.C, visible=False, closezip=False)

		# phase (2-D: variant x sample)
		p = r.addfolder('phase')
		self._p = p
		self.n_phase = p.add('data', np.empty((0, self.nSamp), np.uint8),
			storage='bit1', compress=self.C, closezip=False)

		# annotation
		a = r.addfolder('annotation')
		self._a = a
		self.n_id = a.add('id', [], storage='string', compress=self.C,
			closezip=False)
		self.n_qual = a.add('qual', np.empty(0, np.float32),
			storage='float32', compress=self.C, closezip=False)
		self.n_filter = a.add('filter', np.empty(0, np.int32),
			storage='int32', compress=self.C, closezip=False)
		self._filt_index = {lv: i + 1 for i, lv in enumerate(self.filter_levels)}

		# INFO fields
		self.info_fields = dict(info_fields) if info_fields else {}
		info_node = a.addfolder('info')
		self._info = {}   # name -> (datanode, idxnode_or_None, kind, storage)
		for k, fld in self.info_fields.items():
			typ = fld.get('Type', 'String')
			if typ == 'Flag':
				n = info_node.add(k, np.empty(0, np.uint8), storage='bit1',
					compress=self.C, closezip=False)
				self._info[k] = (n, None, 'flag', 'bit1')
			else:
				storage = _TYPE_STORAGE.get(typ, 'string')
				n = info_node.add(k, _empty_1d(storage), storage=storage,
					compress=self.C, closezip=False)
				variable = str(fld.get('Number', '.')) != '1'
				idx = None
				if variable:
					idx = info_node.add('@' + k, np.empty(0, np.int32),
						storage='int32', compress=self.C, visible=False,
						closezip=False)
				self._info[k] = (n, idx, 'var' if variable else 'fixed',
					storage)
			_set_attrs(info_node.index(k), fld)

		# FORMAT fields.  Number=='1' -> fixed (one value per genotype);
		# anything else (R/G/A/./int>1) -> variable length (per-variant count
		# in @data, data is variant-blocked: count_v rows x nSamp per variant).
		self.format_fields = dict(format_fields) if format_fields else {}
		fmt_node = a.addfolder('format')
		self._fmt = {}   # name -> (datanode, idxnode, storage, kind)
		for k, fld in self.format_fields.items():
			storage = _TYPE_STORAGE.get(fld.get('Type', 'String'), 'string')
			kind = 'fixed' if str(fld.get('Number', '1')) == '1' else 'var'
			fdir = fmt_node.addfolder(k)
			_set_attrs(fdir, fld)
			dn = fdir.add('data', _empty_2d(storage, self.nSamp),
				storage=storage, compress=self.C, closezip=False)
			di = fdir.add('@data', np.empty(0, np.int32), storage='int32',
				compress=self.C, visible=False, closezip=False)
			self._fmt[k] = (dn, di, storage, kind)

	# -- streaming append -------------------------------------------------

	def append(self, chromosome, position, allele, genotype, phase=None,
			variant_id=None, annot_id=None, qual=None, filter=None,
			info=None, format=None):
		"""Append a block of K variants.

		genotype : int array (K, nSamp, ploidy); -1 = missing (allele >= 3 is
			escaped to missing, matching the VCF importer).
		info : dict name -> length-K vector (fixed/flag) OR (flat_values,
			counts_K) for variable-length fields.
		format : dict name -> (K, nSamp) array for Number=1 fields.
		"""
		pos = np.asarray(position, np.int32)
		k = int(pos.shape[0])
		info = info or {}
		format = format or {}

		# genotype -> Bit2 codes
		g = np.asarray(genotype)
		code = np.where((g >= 0) & (g < _GENO_NA), g, _GENO_NA).astype(np.uint8)
		self.n_geno.append(code.reshape(k, self.nSamp, self.ploidy))
		self.n_gidx.append(np.ones(k, np.uint8))

		# phase
		if phase is None:
			ph = np.zeros((k, self.nSamp), np.uint8)
		else:
			ph = np.asarray(phase, np.uint8).reshape(k, self.nSamp)
		self.n_phase.append(ph)

		# site keys
		if variant_id is None:
			variant_id = np.arange(self._nvar + 1, self._nvar + 1 + k,
				dtype=np.int32)
		self.n_vid.append(np.asarray(variant_id, np.int32))
		self.n_pos.append(pos)
		self.n_chrom.append([str(c) for c in chromosome])
		self.n_allele.append([str(x) for x in allele])

		if annot_id is None:
			self.n_id.append(['.'] * k)
		else:
			self.n_id.append(['.' if (x is None or x == '') else str(x)
				for x in annot_id])
		self.n_qual.append(np.full(k, np.nan, np.float32) if qual is None
			else np.asarray(qual, np.float32))

		if filter is None:
			fcode = np.full(k, self._filt_index.get('PASS', 1), np.int32)
		elif isinstance(filter[0], str):
			fcode = np.asarray([self._filt_index.get(x, 1) for x in filter],
				np.int32)
		else:
			fcode = np.asarray(filter, np.int32)
		self.n_filter.append(fcode)

		# INFO (append for every declared field to stay aligned with nvar)
		for name, (n, idx, kind, storage) in self._info.items():
			v = info.get(name)
			if kind == 'flag':
				n.append(np.zeros(k, np.uint8) if v is None
					else np.asarray(v, np.uint8))
			elif kind == 'fixed':
				n.append(_coerce_1d(v, storage, k))
			else:  # variable length: v = (flat_values, counts)
				if v is None:
					idx.append(np.zeros(k, np.int32))
				else:
					flat, counts = v
					if len(flat):
						n.append(_coerce_1d(flat, storage, len(flat)))
					idx.append(np.asarray(counts, np.int32))

		# FORMAT.  Fixed: value is a (k, nSamp) array.  Variable: value is
		# (data_block, counts) where data_block is (sum(counts), nSamp)
		# variant-blocked (value-major within each variant) and counts is the
		# per-variant value count.
		for name, (dn, di, storage, kind) in self._fmt.items():
			v = format.get(name)
			if kind == 'fixed':
				dn.append(_coerce_2d(v, storage, k, self.nSamp))
				di.append(np.ones(k, np.int32))
			elif v is None:
				di.append(np.zeros(k, np.int32))
			else:
				data_block, counts = v
				data_block = np.asarray(data_block)
				if data_block.shape[0]:
					dn.append(data_block.astype(_np_dtype(storage),
						copy=False))
				di.append(np.asarray(counts, np.int32))

		self._nvar += k
		return self._nvar

	# -- finalize ---------------------------------------------------------

	def close(self):
		"""Write the empty extra nodes, set attributes, finalize compression
		on every appendable node, and close the file."""
		C = self.C
		ei = self._g.add('extra.index', np.empty((0, 3), np.int32),
			storage='int32', compress=C, visible=False, closezip=True)
		ei.putattr('R.colnames', ['sample.index', 'variant.index', 'length'])
		self._g.add('extra', np.empty(0, np.int16), storage='int16',
			compress=C, visible=False, closezip=True)
		pei = self._p.add('extra.index', np.empty((0, 3), np.int32),
			storage='int32', compress=C, visible=False, closezip=True)
		pei.putattr('R.colnames', ['sample.index', 'variant.index', 'length'])
		self._p.add('extra', np.empty(0, np.uint8), storage='bit1',
			compress=C, visible=False, closezip=True)

		self.n_filter.putattr('R.class', 'factor')
		self.n_filter.putattr('R.levels', self.filter_levels
			if len(self.filter_levels) > 1 else self.filter_levels[0])

		d = self._r.addfolder('description')
		d.putattr('vcf.fileformat', self.fileformat)

		for n in (self.n_vid, self.n_pos, self.n_chrom, self.n_allele,
				self.n_geno, self.n_gidx, self.n_phase, self.n_id,
				self.n_qual, self.n_filter):
			n.readmode()
		for (n, idx, kind, storage) in self._info.values():
			n.readmode()
			if idx is not None:
				idx.readmode()
		for (dn, di, storage, kind) in self._fmt.values():
			dn.readmode()
			di.readmode()
		self.f.close()


# ---- functional wrappers (the seqCreateGDS / seqAppendVariants / ... API) --

def seqCreateGDS(path, sample_id, ploidy=2, compress='LZMA_RA',
		info_fields=None, format_fields=None, filter_levels=None,
		fileformat='VCFv4.3'):
	"""Create a streaming SeqArray GDS writer (R-flavoured name)."""
	return SeqVarGDSWriter(path, sample_id, ploidy=ploidy, compress=compress,
		info_fields=info_fields, format_fields=format_fields,
		filter_levels=filter_levels, fileformat=fileformat)


def seqAppendVariants(writer, chromosome, position, allele, genotype, **kw):
	"""Append a block of K variants to a streaming writer."""
	return writer.append(chromosome, position, allele, genotype, **kw)


def seqCloseGDS(writer):
	"""Finalize and close a streaming writer."""
	writer.close()
