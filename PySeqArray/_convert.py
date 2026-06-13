# ===========================================================================
#
# _convert.py: native format converters for PySeqArray
#   - seqGDS2BED / seqBED2GDS  (PLINK binary)
#   - seqExport                (subset -> new SeqArray GDS)
#
# Pure Python, no R / SEXP layer. Validated against R SeqArray / PLINK.
#
# Copyright (C) 2017-2026  Xiuwen Zheng -- GPLv3
# ===========================================================================

import numpy as np
import pygds

_GENO_NA = 255
_NA_INT32 = -2147483648


# ---------------------------------------------------------------------------
# PLINK BED export
# ---------------------------------------------------------------------------

def seqGDS2BED(gdsfile, bed_prefix, verbose=True):
	"""Export the current selection to PLINK .bed/.bim/.fam (R: seqGDS2BED).

	Only biallelic variants are written (PLINK is biallelic). A1 is the
	alternate allele, A2 the reference allele, matching R's seqGDS2BED.
	"""
	chrom = np.asarray(gdsfile.GetData('chromosome')).astype(str)
	pos = np.asarray(gdsfile.GetData('position'))
	vid = np.asarray(gdsfile.GetData('annotation/id')).astype(str)
	allele = np.asarray(gdsfile.GetData('allele')).astype(str)
	samp = np.asarray(gdsfile.GetData('sample.id')).astype(str)
	geno = np.asarray(gdsfile.GetData('genotype'))   # (nVar, nSamp, ploidy)
	nVar, nSamp = geno.shape[0], geno.shape[1]

	# .fam
	with open(bed_prefix + '.fam', 'w', newline='\n') as fh:
		for s in samp:
			fh.write('0\t%s\t0\t0\t0\t-9\n' % s)

	# .bim  (A1 = alt, A2 = ref)
	refs = np.empty(nVar, dtype=object)
	alts = np.empty(nVar, dtype=object)
	with open(bed_prefix + '.bim', 'w', newline='\n') as fh:
		for i in range(nVar):
			parts = allele[i].split(',')
			ref = parts[0]
			alt = parts[1] if len(parts) > 1 else '.'
			refs[i] = ref
			alts[i] = alt
			fh.write('%s\t%s\t0\t%d\t%s\t%s\n' % (
				chrom[i], vid[i] if vid[i] else '.', int(pos[i]), alt, ref))

	# .bed  (SNP-major). ref-dosage d: d==0->00, d==1->10, d==2->11, NA->01
	miss = (geno == _GENO_NA).any(axis=2)
	dose = (geno == 0).sum(axis=2).astype(np.uint8)   # ref-allele dosage
	code = np.where(miss, 1,
		np.where(dose == 0, 0, np.where(dose == 1, 2, 3))).astype(np.uint8)
	bytes_per = (nSamp + 3) // 4
	with open(bed_prefix + '.bed', 'wb') as fh:
		fh.write(bytes([0x6c, 0x1b, 0x01]))
		buf = bytearray(bytes_per)
		for i in range(nVar):
			row = code[i]
			for k in range(bytes_per):
				b = 0
				for s in range(4):
					j = k * 4 + s
					if j < nSamp:
						b |= int(row[j]) << (2 * s)
				buf[k] = b
			fh.write(bytes(buf))
	if verbose:
		print('seqGDS2BED: %d variants x %d samples -> %s.{bed,bim,fam}' %
			(nVar, nSamp, bed_prefix))
	return bed_prefix


# ---------------------------------------------------------------------------
# PLINK BED import
# ---------------------------------------------------------------------------

def seqBED2GDS(bed_prefix, out_fn, compress='LZMA_RA', verbose=True):
	"""Import PLINK .bed/.bim/.fam into a SeqArray GDS (R: seqBED2GDS)."""
	# .fam
	fam = [ln.split() for ln in open(bed_prefix + '.fam')]
	samp = [c[1] for c in fam]
	nSamp = len(samp)
	# .bim
	bim = [ln.split() for ln in open(bed_prefix + '.bim')]
	nVar = len(bim)
	chrom = [c[0] for c in bim]
	vid = [c[1] for c in bim]
	pos = np.asarray([int(c[3]) for c in bim], dtype=np.int32)
	a1 = [c[4] for c in bim]   # alt
	a2 = [c[5] for c in bim]   # ref
	allele = ['%s,%s' % (a2[i], a1[i]) for i in range(nVar)]  # "ref,alt"

	# .bed -> genotype (nVar, nSamp, 2)
	bytes_per = (nSamp + 3) // 4
	geno = np.empty((nVar, nSamp, 2), dtype=np.uint8)
	with open(bed_prefix + '.bed', 'rb') as fh:
		magic = fh.read(3)
		if magic[:2] != bytes([0x6c, 0x1b]):
			raise ValueError('not a PLINK .bed file')
		if magic[2] != 1:
			raise ValueError('only SNP-major .bed is supported')
		# PLINK 2-bit codes (A1=alt=allele1, A2=ref=allele0):
		#   00 hom A1A1 -> (alt,alt)=(1,1);  10 het -> (0,1);
		#   11 hom A2A2 -> (ref,ref)=(0,0);  01 missing
		decode = {0: (1, 1), 2: (0, 1), 3: (0, 0), 1: (_GENO_NA, _GENO_NA)}
		for i in range(nVar):
			raw = fh.read(bytes_per)
			for j in range(nSamp):
				b = raw[j >> 2]
				c = (b >> (2 * (j & 3))) & 3
				geno[i, j] = decode[c]

	_write_min_seqgds(out_fn, samp, chrom, pos, vid, allele, geno,
		compress, verbose)
	if verbose:
		print('seqBED2GDS: %d variants x %d samples -> %s' %
			(nVar, nSamp, out_fn))
	return out_fn


def _write_min_seqgds(out_fn, samp, chrom, pos, vid, allele, geno,
		compress, verbose):
	"""Write a minimal but valid SeqArray GDS (no INFO/FORMAT)."""
	from PySeqArray._vcf_import import _write_gds
	nVar = len(chrom)
	nSamp = len(samp)
	qual = [np.nan] * nVar
	filt = ['PASS'] * nVar
	geno2 = np.where(geno == _GENO_NA, 3, geno).astype(np.uint8)
	geno_blocks = [geno2[i] for i in range(nVar)]
	phase_blocks = [np.zeros(nSamp, dtype=np.uint8) for _ in range(nVar)]
	_write_gds(out_fn, 'VCFv4.1', samp, list(chrom), list(pos),
		[v if v else '.' for v in vid], list(allele), qual, filt,
		['PASS'], {'PASS': ''}, geno_blocks, phase_blocks, 2,
		{}, {}, {}, [], {}, compress, False)


# ---------------------------------------------------------------------------
# SNPRelate SNP-GDS conversion
# ---------------------------------------------------------------------------

def seqGDS2SNP(gdsfile, out_fn, compress='LZMA_RA', verbose=True):
	"""Convert the current selection to a SNPRelate SNP-GDS (R: seqGDS2SNP).

	genotype is stored as the reference-allele dosage (0/1/2, 3=missing),
	[nSample x nSNP], matching SNPRelate / R's seqGDS2SNP.
	"""
	chrom = np.asarray(gdsfile.GetData('chromosome')).astype(str)
	pos = np.asarray(gdsfile.GetData('position')).astype(np.int32)
	rsid = np.asarray(gdsfile.GetData('annotation/id')).astype(str)
	allele = np.asarray(gdsfile.GetData('allele')).astype(str)
	samp = np.asarray(gdsfile.GetData('sample.id')).astype(str)
	geno = np.asarray(gdsfile.GetData('genotype'))   # (nVar, nSamp, ploidy)
	nVar, nSamp = geno.shape[0], geno.shape[1]

	miss = (geno == _GENO_NA).any(axis=2)
	# (nVar, nSamp) ref-allele dosage; pygds reverses dims on write so this
	# is presented by gdsfmt as the sample-major [nSample x nSNP] matrix that
	# SNPRelate / R's seqGDS2SNP expect.
	dose = np.where(miss, 3, (geno == 0).sum(axis=2)).astype(np.uint8)
	snp_allele = ['/'.join(a.split(',')[:2]) for a in allele]

	f = pygds.gdsfile()
	f.create(out_fn)
	r = f.root()
	C = compress
	r.add('sample.id', list(samp), storage='string', compress=C, closezip=True)
	r.add('snp.id', np.arange(1, nVar + 1, dtype=np.int32), storage='int32',
		compress=C, closezip=True)
	r.add('snp.rs.id', list(rsid), storage='string', compress=C, closezip=True)
	r.add('snp.position', pos, storage='int32', compress=C, closezip=True)
	r.add('snp.chromosome', list(chrom), storage='string', compress=C,
		closezip=True)
	r.add('snp.allele', snp_allele, storage='string', compress=C,
		closezip=True)
	gn = r.add('genotype', dose, storage='bit2', compress=C, closezip=True)
	gn.putattr('sample.order', None)
	f.close()
	if verbose:
		print('seqGDS2SNP: %d variants x %d samples -> %s' %
			(nVar, nSamp, out_fn))
	return out_fn


def seqSNP2GDS(snp_fn, out_fn, compress='LZMA_RA', verbose=True):
	"""Convert a SNPRelate SNP-GDS to a SeqArray GDS (R: seqSNP2GDS)."""
	s = pygds.gdsfile()
	s.open(snp_fn, allow_dup=True)
	r = s.root()
	samp = list(np.asarray(r.index('sample.id').read()).astype(str))
	chrom = list(np.asarray(r.index('snp.chromosome').read()).astype(str))
	pos = np.asarray(r.index('snp.position').read(), dtype=np.int32)
	try:
		rsid = list(np.asarray(r.index('snp.rs.id').read()).astype(str))
	except Exception:
		rsid = ['.'] * len(chrom)
	sa = np.asarray(r.index('snp.allele').read()).astype(str)
	dose = np.asarray(r.index('genotype').read())   # (nSamp, nSNP) ref-dosage
	gattr = r.index('genotype').getattr()
	s.close()

	# orient to (nSNP, nSamp): SNPRelate stores sample-major when the
	# 'sample.order' attribute is present.
	if 'snp.order' in gattr:
		dose = dose.T
	if dose.shape[0] == len(samp) and dose.shape[1] == len(chrom):
		dose = dose.T   # -> (nSNP, nSamp)
	nVar, nSamp = dose.shape
	allele = [a.replace('/', ',') for a in sa]

	# ref-dosage -> diploid genotype (allele0=ref)
	geno = np.empty((nVar, nSamp, 2), dtype=np.uint8)
	d = dose
	geno[..., 0] = np.where(d == 3, _GENO_NA, np.where(d >= 1, 0, 1))
	geno[..., 1] = np.where(d == 3, _GENO_NA, np.where(d == 2, 0, 1))

	_write_min_seqgds(out_fn, samp, chrom, pos, rsid, allele, geno,
		compress, verbose)
	if verbose:
		print('seqSNP2GDS: %d variants x %d samples -> %s' %
			(nVar, nSamp, out_fn))
	return out_fn


# ---------------------------------------------------------------------------
# Merge multiple SeqArray GDS files (variant concatenation, shared samples)
# ---------------------------------------------------------------------------

def seqMerge(gds_fns, out_fn, compress='LZMA_RA', verbose=True):
	"""Merge SeqArray GDS files by concatenating variants (R: seqMerge).

	All input files must contain the same samples (in the same order). The
	common case of combining per-chromosome files. Annotation INFO/FORMAT
	fields common to all inputs are carried over.
	"""
	import PySeqArray as ps
	if isinstance(gds_fns, str):
		gds_fns = [gds_fns]
	files = [ps.seqOpen(fn, allow_dup=True) for fn in gds_fns]
	try:
		samp = np.asarray(files[0].GetData('sample.id')).astype(str)
		for f in files[1:]:
			s = np.asarray(f.GetData('sample.id')).astype(str)
			if not np.array_equal(s, samp):
				raise ValueError('seqMerge: all files must share the same '
					'samples in the same order.')
		# common INFO / FORMAT fields
		info_common = set(files[0]._list_children('annotation/info'))
		fmt_common = set(files[0]._list_children('annotation/format'))
		for f in files[1:]:
			info_common &= set(f._list_children('annotation/info'))
			fmt_common &= set(f._list_children('annotation/format'))
		info_common = [c for c in files[0]._list_children('annotation/info')
			if c in info_common]
		fmt_common = [c for c in files[0]._list_children('annotation/format')
			if c in fmt_common]

		# concatenate per-variant columns
		def cat(name, astype=None):
			parts = [np.asarray(f.GetData(name)) for f in files]
			if astype is not None:
				parts = [p.astype(astype) for p in parts]
			return np.concatenate(parts)

		chrom = list(cat('chromosome').astype(str))
		pos = list(cat('position'))
		vid = list(cat('annotation/id').astype(str))
		allele = list(cat('allele').astype(str))
		try:
			qual = list(cat('annotation/qual'))
		except Exception:
			qual = [np.nan] * len(chrom)
		try:
			filt = list(cat('annotation/filter').astype(str))
		except Exception:
			filt = ['PASS'] * len(chrom)
		nSamp = len(samp)
		geno_blocks, phase_blocks = [], []
		for f in files:
			g = np.asarray(f.GetData('genotype'))
			g = np.where(g == _GENO_NA, 3, g).astype(np.uint8)
			geno_blocks.extend(g[i] for i in range(g.shape[0]))
			try:
				ph = np.asarray(f.GetData('phase'))
				phase_blocks.extend(ph[i] for i in range(ph.shape[0]))
			except Exception:
				phase_blocks.extend(np.zeros(nSamp, dtype=np.uint8)
					for _ in range(g.shape[0]))
		nVar = len(geno_blocks)

		from PySeqArray._vcf_import import _write_gds, _Field
		# build INFO / FORMAT accumulators across files
		info_fields_d, info_acc = {}, {}
		for nm in info_common:
			meta = files[0]._vcf_field_meta('info', nm)
			fld = _Field({'ID': nm, 'Number': meta.get('Number', '.'),
				'Type': meta.get('Type', 'String'),
				'Description': meta.get('Description', '')})
			info_fields_d[nm] = fld
			acc = {'vals': [], 'cnts': []}
			for f in files:
				nv = f.NumVariant(selected=True)
				a = _accumulate_info(f.GetData('annotation/info/' + nm), nv, fld)
				acc['vals'].extend(a['vals'])
				acc['cnts'].extend(a['cnts'])
			info_acc[nm] = acc
		fmt_fields_d, fmt_acc, fmt_extra = {}, {}, []
		for nm in fmt_common:
			meta = files[0]._vcf_field_meta('format', nm)
			fld = _Field({'ID': nm, 'Number': meta.get('Number', '.'),
				'Type': meta.get('Type', 'String'),
				'Description': meta.get('Description', '')})
			fmt_fields_d[nm] = fld
			fmt_extra.append(nm)
			rows = []
			for f in files:
				nv = f.NumVariant(selected=True)
				rows.extend(_accumulate_fmt(
					f.GetData('annotation/format/' + nm), nv, nSamp))
			fmt_acc[nm] = {'vals': rows}

		flevels = files[0]._filter_levels()
		ff = files[0]._description_attr('vcf.fileformat') or 'VCFv4.1'
		_write_gds(out_fn, ff, list(samp), chrom, pos,
			[v if v else '.' for v in vid], allele, qual, filt,
			[lv for lv, _d in flevels], {lv: d for lv, d in flevels},
			geno_blocks, phase_blocks, 2, info_fields_d, info_acc,
			fmt_fields_d, fmt_extra, fmt_acc, compress, False)
	finally:
		for f in files:
			f.close()
	if verbose:
		print('seqMerge: %d files -> %d variants x %d samples -> %s' %
			(len(gds_fns), nVar, nSamp, out_fn))
	return out_fn


# ---------------------------------------------------------------------------
# Subset export to a new SeqArray GDS
# ---------------------------------------------------------------------------

def seqExport(gdsfile, out_fn, info_fields=None, fmt_fields=None,
		compress='LZMA_RA', verbose=True):
	"""Export the current selection to a new SeqArray GDS (R: seqExport)."""
	from PySeqArray._vcf_import import _write_gds, _Field

	chrom = list(np.asarray(gdsfile.GetData('chromosome')).astype(str))
	pos = list(np.asarray(gdsfile.GetData('position')))
	vid = list(np.asarray(gdsfile.GetData('annotation/id')).astype(str))
	allele = list(np.asarray(gdsfile.GetData('allele')).astype(str))
	samp = list(np.asarray(gdsfile.GetData('sample.id')).astype(str))
	try:
		qual = list(np.asarray(gdsfile.GetData('annotation/qual')))
	except Exception:
		qual = [np.nan] * len(chrom)
	try:
		filt = list(np.asarray(gdsfile.GetData('annotation/filter')).astype(str))
	except Exception:
		filt = ['PASS'] * len(chrom)
	geno = np.asarray(gdsfile.GetData('genotype'))
	nVar, nSamp, ploidy = geno.shape
	geno2 = np.where(geno == _GENO_NA, 3, geno).astype(np.uint8)
	geno_blocks = [geno2[i] for i in range(nVar)]
	try:
		phase = np.asarray(gdsfile.GetData('phase'))
		phase_blocks = [phase[i] for i in range(nVar)]
	except Exception:
		phase_blocks = [np.zeros(nSamp, dtype=np.uint8) for _ in range(nVar)]

	flevels = gdsfile._filter_levels()
	filt_levels = [lv for lv, _d in flevels]
	filt_desc = {lv: d for lv, d in flevels}

	# INFO / FORMAT: carry over the available fields
	info_names = info_fields if info_fields is not None else \
		gdsfile._list_children('annotation/info')
	info_fields_d, info_acc = {}, {}
	for nm in info_names:
		try:
			raw = gdsfile.GetData('annotation/info/' + nm)
		except Exception:
			continue
		meta = gdsfile._vcf_field_meta('info', nm)
		fld = _Field({'ID': nm, 'Number': meta.get('Number', '.'),
			'Type': meta.get('Type', 'String'),
			'Description': meta.get('Description', '')})
		info_fields_d[nm] = fld
		info_acc[nm] = _accumulate_info(raw, nVar, fld)

	fmt_names = fmt_fields if fmt_fields is not None else \
		gdsfile._list_children('annotation/format')
	fmt_fields_d, fmt_acc, fmt_extra = {}, {}, []
	for nm in fmt_names:
		try:
			raw = gdsfile.GetData('annotation/format/' + nm)
		except Exception:
			continue
		meta = gdsfile._vcf_field_meta('format', nm)
		fld = _Field({'ID': nm, 'Number': meta.get('Number', '.'),
			'Type': meta.get('Type', 'String'),
			'Description': meta.get('Description', '')})
		fmt_fields_d[nm] = fld
		fmt_extra.append(nm)
		fmt_acc[nm] = {'vals': _accumulate_fmt(raw, nVar, nSamp)}

	ff = gdsfile._description_attr('vcf.fileformat') or 'VCFv4.1'
	_write_gds(out_fn, ff, samp, chrom, pos,
		[v if v else '.' for v in vid], allele, qual, filt,
		filt_levels, filt_desc, geno_blocks, phase_blocks, ploidy,
		info_fields_d, info_acc, fmt_fields_d, fmt_extra, fmt_acc,
		compress, False)
	if verbose:
		print('seqExport: %d variants x %d samples -> %s' %
			(nVar, nSamp, out_fn))
	return out_fn


def _accumulate_info(raw, nVar, fld):
	# rebuild the {'vals','cnts'} accumulator form used by _write_gds
	from PySeqArray import _fmt_cell
	vals, cnts = [], []
	if fld.type == 'Flag':
		arr = np.asarray(raw)
		for i in range(nVar):
			vals.append(int(arr[i]) if arr.ndim else 0)
			cnts.append(1)
		return {'vals': vals, 'cnts': cnts}
	if isinstance(raw, dict):
		idx = np.asarray(raw['index']).astype(np.int64)
		data = np.asarray(raw['data'])
		off = np.concatenate(([0], np.cumsum(idx)))
		for i in range(nVar):
			n = int(idx[i])
			seg = data[off[i]:off[i] + n]
			vals.extend(_fmt_cell(x) for x in np.atleast_1d(seg)) if n else None
			cnts.append(n)
	else:
		arr = np.asarray(raw)
		for i in range(nVar):
			vals.append(_fmt_cell(arr[i]))
			cnts.append(1)
	return {'vals': vals, 'cnts': cnts}


def _accumulate_fmt(raw, nVar, nSamp):
	from PySeqArray import _fmt_cell
	rows = []
	if isinstance(raw, dict):
		idx = np.asarray(raw['index']).astype(np.int64)
		data = np.asarray(raw['data'])
		off = np.concatenate(([0], np.cumsum(idx)))
		for i in range(nVar):
			n = int(idx[i])
			if n >= 1:
				rows.append([_fmt_cell(x) for x in np.atleast_1d(data[off[i], :])])
			else:
				rows.append(['.'] * nSamp)
	else:
		arr = np.asarray(raw)
		for i in range(nVar):
			rows.append([_fmt_cell(arr[i, j]) for j in range(nSamp)])
	return rows
