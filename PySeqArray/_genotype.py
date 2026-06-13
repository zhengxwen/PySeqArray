"""Genotype decoding (page-aware).

SeqArray stores genotypes as 2-bit values.  A variant with more than 3 alleles
needs more than 2 bits, so it is split across several 2-bit *pages*.
``genotype/@data`` gives the number of pages per variant; the slowest dimension
of ``genotype/data`` indexes pages (not variants), so the page columns for a
variant are a contiguous run of length ``@data[v]``.

An allele value is reconstructed base-4, little-endian across pages::

    value = sum_i page_i * 4**i

and a call is missing when every page is 3, i.e. ``value == 4**k - 1``.

Results mirror R ``seqGetData("genotype")``: a numpy **masked** array of shape
``(ploidy, n_sel_sample, n_sel_variant)`` with masked = missing.
"""

import numpy as np

from ._types import _offsets

# block size for streaming decode in summary kernels
_GENO_BS = 1024


def _geno_npages(f):
    """Number of 2-bit pages per variant (all-ones when ``@data`` is absent)."""
    idx = f._node("genotype/@data", silent=True)
    if idx is None:
        return np.ones(f.nvar, dtype=np.int64)
    return np.asarray(idx.read(), dtype=np.int64)


def _geno_page_cols(f, svar, npages):
    """Boolean mask over the page axis selecting the pages of ``svar``.

    Returns ``(col_mask, page_counts_for_svar)`` where ``col_mask`` has length
    equal to the total number of pages in the file.
    """
    off = _offsets(npages)
    total = int(npages.sum())
    col_mask = np.zeros(total, dtype=bool)
    for v in svar:
        col_mask[off[v]:off[v] + npages[v]] = True
    return col_mask, npages[svar]


def _decode_block(raw, npages_sel):
    """Decode a (n_sel_pages, ns, p) numpy block into (p, ns, nv) value+mask.

    ``raw`` holds the selected pages in ascending order; ``npages_sel`` is the
    per-selected-variant page count.  Returns ``(values, mask)`` int / bool
    arrays of shape ``(p, ns, nv)``.
    """
    raw = np.asarray(raw)
    n_pages, ns, p = raw.shape
    nv = len(npages_sel)
    # widen to int when any variant is multi-page (>= 4 alleles)
    dtype = np.int8 if (nv == 0 or npages_sel.max() <= 1) else np.int64
    values = np.empty((p, ns, nv), dtype=dtype)
    mask = np.empty((p, ns, nv), dtype=bool)
    coloff = 0
    for j in range(nv):
        k = int(npages_sel[j])
        if k == 1:
            page = raw[coloff]               # (ns, p)
            val = page.T                     # (p, ns)
            miss = (page == 3).T
        else:
            sentinel = (1 << (2 * k)) - 1
            acc = np.zeros((ns, p), dtype=np.int64)
            for pg in range(k):
                acc += raw[coloff + pg].astype(np.int64) << (2 * pg)
            val = acc.T
            miss = (acc == sentinel).T
        values[:, :, j] = val
        mask[:, :, j] = miss
        coloff += k
    return values, mask


def _read_genotype(f):
    """Read+decode genotypes for the current selection.

    Returns a ``numpy.ma.MaskedArray`` of shape
    ``(ploidy, n_sel_sample, n_sel_variant)``.
    """
    p = f.ploidy
    svar = f.sel_variant_idx()
    ssamp = f.sel_sample_idx()
    ns, nv = len(ssamp), len(svar)
    npages = _geno_npages(f)
    if nv == 0:
        empty = np.empty((p, ns, 0), dtype=np.int8)
        return np.ma.MaskedArray(empty, mask=np.zeros_like(empty, dtype=bool))
    col_mask, npages_sel = _geno_page_cols(f, svar, npages)
    smask = f.sample_sel

    # sample-major fast path: when far fewer samples than variants are selected,
    # read the transposed ``~data`` twin if present (identical results).
    tnode = f._node("genotype/~data", silent=True)
    if tnode is not None and ns < nv:
        # ~data numpy dims are [sample, page, ploidy]; sel in that order.
        raw = np.asarray(tnode.readex([smask, col_mask, None]))  # (ns, npg, p)
        raw = np.transpose(raw, (1, 0, 2))                        # -> (npg, ns, p)
    else:
        node = f._node("genotype/data")
        # data numpy dims are [page, sample, ploidy].
        raw = np.asarray(node.readex([col_mask, smask, None]))    # (npg, ns, p)

    values, mask = _decode_block(raw, npages_sel)
    return np.ma.MaskedArray(values, mask=mask)


def _foreach_variant_geno(f, f_each):
    """Stream decoded genotypes one variant at a time.

    Calls ``f_each(values, mask, j)`` where ``values``/``mask`` are
    ``(ploidy, n_sel_sample)`` arrays for selected-variant position ``j``.
    Used by the summary statistics (M2).
    """
    node = f._node("genotype/data")
    svar = f.sel_variant_idx()
    smask = f.sample_sel
    npages = _geno_npages(f)
    off = _offsets(npages)
    total = int(npages.sum())
    nv = len(svar)
    j = 0
    while j < nv:
        blk = svar[j:min(j + _GENO_BS, nv)]
        col_mask = np.zeros(total, dtype=bool)
        for v in blk:
            col_mask[off[v]:off[v] + npages[v]] = True
        raw = np.asarray(node.readex([col_mask, smask, None]))  # (npg, ns, p)
        vals, mask = _decode_block(raw, npages[blk])
        for b in range(len(blk)):
            f_each(vals[:, :, b], mask[:, :, b], j + b)
        j += len(blk)
