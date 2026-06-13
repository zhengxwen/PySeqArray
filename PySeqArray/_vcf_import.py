# ===========================================================================
#
# _vcf_import.py: native VCF -> SeqArray GDS importer (R: seqVCF2GDS)
#
# Pure-Python parser that builds a SeqArray-format GDS file using the pygds
# write API. No R / SEXP layer. The produced file is a valid SeqArray GDS
# readable by both PySeqArray and R's SeqArray.
#
# Copyright (C) 2017-2026  Xiuwen Zheng
# GPLv3
# ===========================================================================

import gzip
import re
import numpy as np
import pygds


# ---- VCF type mapping -----------------------------------------------------

_TYPE_STORAGE = {
	'Integer': 'int32',
	'Float': 'float64',
	'Flag': 'bit1',
	'String': 'string',
	'Character': 'string',
}

_NA_INT32 = -2147483648
_GENO_NA = 3  # 2-bit missing/escape code


def _open_text(fn):
	if str(fn).endswith('.gz'):
		return gzip.open(fn, 'rt')
	return open(fn, 'rt')


def _parse_meta_line(line):
	# parse a ##KEY=<ID=...,Number=...,Type=...,Description="...">
	m = re.match(r'##(\w+)=<(.+)>\s*$', line)
	if not m:
		return None, None
	kind = m.group(1)
	body = m.group(2)
	# split on commas not inside quotes
	fields = {}
	for part in re.findall(r'(\w+)=("[^"]*"|[^,]*)', body):
		k, v = part
		if v.startswith('"') and v.endswith('"'):
			v = v[1:-1]
		fields[k] = v
	return kind, fields


class _Field:
	__slots__ = ('id', 'number', 'type', 'desc', 'storage', 'fixed')

	def __init__(self, d):
		self.id = d.get('ID')
		self.number = d.get('Number', '.')
		self.type = d.get('Type', 'String')
		self.desc = d.get('Description', '')
		self.storage = _TYPE_STORAGE.get(self.type, 'string')
		# fixed single value per record?
		self.fixed = (self.number == '1')


def parse_vcf_header(lines):
	"""Parse VCF '##'/'#CHROM' header lines.

	Returns (fileformat, info_fields, format_fields, filter_levels,
	filter_desc, samples).
	"""
	fileformat = 'VCFv4.0'
	info, fmt = {}, {}
	filt_levels, filt_desc = [], {}
	samples = []
	for line in lines:
		line = line.rstrip('\n')
		if line.startswith('##'):
			if line.startswith('##fileformat='):
				fileformat = line.split('=', 1)[1]
				continue
			kind, fields = _parse_meta_line(line)
			if kind == 'INFO':
				info[fields['ID']] = _Field(fields)
			elif kind == 'FORMAT':
				fmt[fields['ID']] = _Field(fields)
			elif kind == 'FILTER':
				filt_levels.append(fields['ID'])
				filt_desc[fields['ID']] = fields.get('Description', '')
		elif line.startswith('#CHROM'):
			cols = line.split('\t')
			if len(cols) > 9:
				samples = cols[9:]
	if 'PASS' not in filt_levels:
		filt_levels = ['PASS'] + filt_levels
		filt_desc.setdefault('PASS', 'All filters passed')
	return fileformat, info, fmt, filt_levels, filt_desc, samples


def seqVCF_Header(vcf_fn):
	"""Parse the header of a VCF file (R: seqVCF_Header).

	Returns a dict with keys ``fileformat``, ``info`` (name -> field dict),
	``format``, ``filter`` (list of levels) and ``samples``.
	"""
	header_lines = []
	with _open_text(vcf_fn) as fh:
		for line in fh:
			if line.startswith('#'):
				header_lines.append(line)
				if line.startswith('#CHROM'):
					break
			else:
				break
	ff, info, fmt, flt, fdesc, samples = parse_vcf_header(header_lines)

	def _fdict(d):
		return {k: {'Number': v.number, 'Type': v.type,
			'Description': v.desc} for k, v in d.items()}

	return {'fileformat': ff, 'info': _fdict(info), 'format': _fdict(fmt),
		'filter': flt, 'samples': list(samples)}


def seqVCF_SampID(vcf_fn):
	"""Return the sample IDs from a VCF header (R: seqVCF_SampID)."""
	return seqVCF_Header(vcf_fn)['samples']


def _enc_allele(gt_allele):
	# map a VCF allele token ('0','1','.', etc.) to the 2-bit code
	if gt_allele == '.' or gt_allele == '':
		return _GENO_NA
	v = int(gt_allele)
	return v if v < _GENO_NA else _GENO_NA  # >=3 -> escape (extra not built)


def seqVCF2GDS(vcf_fn, out_fn, ploidy=2, compress='LZMA_RA', verbose=True):
	"""Import a VCF file into a SeqArray GDS file (R: seqVCF2GDS).

	Parameters
	----------
	vcf_fn : str
		input VCF path (plain or .gz)
	out_fn : str
		output GDS path
	ploidy : int
		expected ploidy of the genotypes (default 2)
	compress : str
		GDS compression method
	verbose : bool
		print progress information
	"""
	# ---- first pass: header ----
	header_lines = []
	with _open_text(vcf_fn) as fh:
		for line in fh:
			if line.startswith('#'):
				header_lines.append(line)
				if line.startswith('#CHROM'):
					break
			else:
				break
	(fileformat, info_fields, fmt_fields, filt_levels, filt_desc,
		samples) = parse_vcf_header(header_lines)
	nSamp = len(samples)

	# accumulators
	chrom_l, pos_l, id_l, ref_alt_l, qual_l, filt_l = [], [], [], [], [], []
	geno_blocks = []     # list of (nSamp, ploidy) uint8 per variant
	phase_blocks = []    # list of (nSamp,) uint8 per variant
	# INFO columns: name -> list (fixed) or (list_values, list_counts)
	info_acc = {k: {'vals': [], 'cnts': []} for k in info_fields}
	fmt_extra = [k for k in fmt_fields if k != 'GT']
	fmt_acc = {k: {'vals': [], 'cnts': []} for k in fmt_extra}

	# ---- second pass: body ----
	nVar = 0
	with _open_text(vcf_fn) as fh:
		for line in fh:
			if line.startswith('#'):
				continue
			line = line.rstrip('\n')
			if not line:
				continue
			f = line.split('\t')
			chrom_l.append(f[0])
			pos_l.append(int(f[1]))
			id_l.append('' if f[2] == '.' else f[2])
			ref = f[3]
			alt = f[4]
			ref_alt_l.append(ref if alt == '.' else ref + ',' + alt)
			qual_l.append(np.nan if f[5] == '.' else float(f[5]))
			filt_l.append(f[6])

			# INFO
			info_present = {}
			if f[7] != '.' and f[7] != '':
				for item in f[7].split(';'):
					if '=' in item:
						k, v = item.split('=', 1)
						info_present[k] = v
					else:
						info_present[item] = True
			for k, fld in info_fields.items():
				if k in info_present:
					v = info_present[k]
					if fld.type == 'Flag':
						info_acc[k]['vals'].append(1)
						info_acc[k]['cnts'].append(1)
					else:
						parts = ([] if v is True else str(v).split(','))
						info_acc[k]['vals'].extend(parts)
						info_acc[k]['cnts'].append(len(parts))
				else:
					if fld.type == 'Flag':
						info_acc[k]['vals'].append(0)
						info_acc[k]['cnts'].append(1)
					else:
						info_acc[k]['cnts'].append(0)

			# FORMAT / genotype
			gt = np.full((nSamp, ploidy), _GENO_NA, dtype=np.uint8)
			ph = np.zeros(nSamp, dtype=np.uint8)
			if nSamp > 0:
				keys = f[8].split(':') if len(f) > 8 else []
				gt_pos = keys.index('GT') if 'GT' in keys else -1
				# init per-format accumulation for this variant
				perfmt = {k: [] for k in fmt_extra}
				for j in range(nSamp):
					cell = f[9 + j] if (9 + j) < len(f) else ''
					sub = cell.split(':') if cell else []
					if gt_pos >= 0 and gt_pos < len(sub):
						g = sub[gt_pos]
						sep = '|' if '|' in g else '/'
						ph[j] = 1 if sep == '|' else 0
						alle = g.replace('|', '/').split('/')
						for a in range(min(ploidy, len(alle))):
							gt[j, a] = _enc_allele(alle[a])
					for k in fmt_extra:
						kp = keys.index(k) if k in keys else -1
						val = sub[kp] if (0 <= kp < len(sub)) else '.'
						perfmt[k].append(val)
				for k in fmt_extra:
					vals = perfmt[k]
					fmt_acc[k]['vals'].append(vals)  # one row per variant
			geno_blocks.append(gt)
			phase_blocks.append(ph)
			nVar += 1
			if verbose and (nVar % 10000 == 0):
				print('  ... %d variants parsed' % nVar)

	if verbose:
		print('Parsed %d variant(s), %d sample(s).' % (nVar, nSamp))

	# ---- build GDS ----
	_write_gds(out_fn, fileformat, samples, chrom_l, pos_l, id_l, ref_alt_l,
		qual_l, filt_l, filt_levels, filt_desc, geno_blocks, phase_blocks,
		ploidy, info_fields, info_acc, fmt_fields, fmt_extra, fmt_acc,
		compress, verbose)
	return out_fn


def _write_gds(out_fn, fileformat, samples, chrom_l, pos_l, id_l, ref_alt_l,
		qual_l, filt_l, filt_levels, filt_desc, geno_blocks, phase_blocks,
		ploidy, info_fields, info_acc, fmt_fields, fmt_extra, fmt_acc,
		compress, verbose):
	nVar = len(chrom_l)
	nSamp = len(samples)
	C = compress
	f = pygds.gdsfile()
	f.create(out_fn)
	r = f.root()
	r.putattr('FileFormat', 'SEQ_ARRAY')
	r.putattr('FileVersion', 'v1.0')

	r.add('sample.id', list(samples) if nSamp else [], storage='string',
		compress=C, closezip=True)
	r.add('variant.id', np.arange(1, nVar + 1, dtype=np.int32),
		storage='int32', compress=C, closezip=True)
	r.add('position', np.asarray(pos_l, dtype=np.int32), storage='int32',
		compress=C, closezip=True)
	r.add('chromosome', list(chrom_l), storage='string', compress=C,
		closezip=True)
	r.add('allele', list(ref_alt_l), storage='string', compress=C,
		closezip=True)

	# genotype
	g = r.addfolder('genotype')
	g.putattr('VariableName', 'GT')
	g.putattr('Description', 'Genotype')
	geno = (np.stack(geno_blocks, axis=0) if nVar else
		np.empty((0, nSamp, ploidy), dtype=np.uint8))
	g.add('data', geno, storage='bit2', compress=C, closezip=True)
	g.add('@data', np.ones(nVar, dtype=np.uint8), storage='uint8',
		compress=C, visible=False, closezip=True)
	ei = g.add('extra.index', np.empty((0, 3), dtype=np.int32),
		storage='int32', compress=C, visible=False, closezip=True)
	ei.putattr('R.colnames', ['sample.index', 'variant.index', 'length'])
	g.add('extra', np.empty(0, dtype=np.int16), storage='int16',
		compress=C, visible=False, closezip=True)

	# phase
	p = r.addfolder('phase')
	phase = (np.stack(phase_blocks, axis=0) if nVar else
		np.empty((0, nSamp), dtype=np.uint8))
	p.add('data', phase, storage='bit1', compress=C, closezip=True)
	pei = p.add('extra.index', np.empty((0, 3), dtype=np.int32),
		storage='int32', compress=C, visible=False, closezip=True)
	pei.putattr('R.colnames', ['sample.index', 'variant.index', 'length'])
	p.add('extra', np.empty(0, dtype=np.uint8), storage='bit1',
		compress=C, visible=False, closezip=True)

	# annotation
	a = r.addfolder('annotation')
	a.add('id', [s if s else '.' for s in id_l], storage='string',
		compress=C, closezip=True)
	a.add('qual', np.asarray(qual_l, dtype=np.float32), storage='float32',
		compress=C, closezip=True)
	# filter as factor
	lvl_index = {lv: i + 1 for i, lv in enumerate(filt_levels)}
	fcode = np.asarray([lvl_index.get(x if x != '.' else 'PASS', 1)
		for x in filt_l], dtype=np.int32)
	flt = a.add('filter', fcode, storage='int32', compress=C, closezip=True)
	flt.putattr('R.class', 'factor')
	flt.putattr('R.levels', filt_levels if len(filt_levels) > 1
		else filt_levels[0])
	flt.putattr('Description', [filt_desc.get(lv, '') for lv in filt_levels])

	# INFO
	info_node = a.addfolder('info')
	for k, fld in info_fields.items():
		acc = info_acc[k]
		cnts = np.asarray(acc['cnts'], dtype=np.int32)
		if fld.type == 'Flag':
			vals = np.asarray(acc['vals'], dtype=np.uint8)
			info_node.add(k, vals, storage='bit1', compress=C, closezip=True,
				visible=True)
			_set_field_attr(info_node.index(k), fld)
			continue
		data = _coerce(acc['vals'], fld.storage)
		nd = info_node.add(k, data, storage=fld.storage, compress=C,
			closezip=True)
		_set_field_attr(nd, fld)
		# variable-length index node @<k> when not strictly Number==1
		if not (fld.number == '1' and np.all(cnts == 1)):
			idxn = info_node.add('@' + k, cnts, storage='int32', compress=C,
				visible=False, closezip=True)

	# FORMAT
	fmt_node = a.addfolder('format')
	for k in fmt_extra:
		fld = fmt_fields[k]
		rows = fmt_acc[k]['vals']  # list per variant of [per-sample str]
		flat = []
		cnts = []
		for row in rows:
			# number of records per variant (here 1 set of nSamp values)
			cnts.append(1)
			flat.append(row)
		mat = _coerce_matrix(flat, fld.storage, nSamp)
		fdir = fmt_node.addfolder(k)
		_set_field_attr(fdir, fld)
		fdir.add('data', mat, storage=fld.storage, compress=C, closezip=True)
		fdir.add('@data', np.ones(nVar, dtype=np.int32), storage='int32',
			compress=C, visible=False, closezip=True)

	# description
	d = r.addfolder('description')
	d.putattr('vcf.fileformat', fileformat)

	f.close()
	if verbose:
		print('Done -> %s' % out_fn)


def _set_field_attr(node, fld):
	node.putattr('Number', fld.number)
	node.putattr('Type', fld.type)
	node.putattr('Description', fld.desc)


def _coerce(vals, storage):
	if storage == 'int32':
		out = np.empty(len(vals), dtype=np.int32)
		for i, v in enumerate(vals):
			out[i] = _NA_INT32 if v in ('.', '') else int(v)
		return out
	if storage == 'float64':
		out = np.empty(len(vals), dtype=np.float64)
		for i, v in enumerate(vals):
			out[i] = np.nan if v in ('.', '') else float(v)
		return out
	return [('' if v == '.' else str(v)) for v in vals]


def _coerce_matrix(rows, storage, nSamp):
	# rows: list (per variant) of list (per sample) of string values
	nVar = len(rows)
	if storage == 'int32':
		m = np.full((nVar, nSamp), _NA_INT32, dtype=np.int32)
		for i, row in enumerate(rows):
			for j, v in enumerate(row):
				if v not in ('.', ''):
					try:
						m[i, j] = int(v)
					except ValueError:
						m[i, j] = _NA_INT32
		return m
	if storage == 'float64':
		m = np.full((nVar, nSamp), np.nan, dtype=np.float64)
		for i, row in enumerate(rows):
			for j, v in enumerate(row):
				if v not in ('.', ''):
					try:
						m[i, j] = float(v)
					except ValueError:
						pass
		return m
	# string
	m = np.empty((nVar, nSamp), dtype=object)
	for i, row in enumerate(rows):
		for j, v in enumerate(row):
			m[i, j] = '' if v == '.' else str(v)
	return m
