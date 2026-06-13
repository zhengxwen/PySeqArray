"""PySeqArray
=========

Python port of the R/Bioconductor **SeqArray** package: data management and
access for whole-genome sequencing variant calls stored in *SeqArray GDS* files.

Unlike a from-scratch reimplementation, PySeqArray **reuses SeqArray's own C++
engine** (``SeqArray/src/*.cpp``) compiled into the :mod:`PySeqArray.cclib`
extension, with the R ``.Call`` glue replaced by a CPython C-API layer — exactly
the way :mod:`pygds` reuses the CoreArray C++ engine of gdsfmt.  The GDS data
itself is opened with pygds, whose GDS C-API is shared with the engine through a
PyCapsule.  See ``DESIGN.md``.

Public API mirrors SeqArray::

    seqOpen, seqClose, seqSummary, seqExampleFileName,
    seqSetFilter, seqResetFilter, seqGetFilter,
    seqGetData
"""

import os
import numpy as np
import pygds

from . import cclib
from ._vcf import seqVCF2GDS, seqGDS2VCF
from ._write import seqAddValue, seqDelete, seqRecompress

__version__ = "0.2.0"

# bind the pygds GDS C-API capsule (idempotent; also done at import of cclib)
cclib.init_gds()

_NA_INT = np.int32(-2147483648)


class SeqVarGDSClass:
    """An open SeqArray GDS file (pygds handle + SeqArray engine file id)."""

    __slots__ = ("gds", "fileid", "filename")

    def __init__(self, gds):
        self.gds = gds
        self.fileid = gds.fileid
        self.filename = gds.filename

    def _dims(self):
        # (nSamp, nVar, ploidy, nSampSel, nVarSel) via the native CFileInfo path
        return cclib.n_dims(self.fileid)

    def nsamp(self):
        return self._dims()[0]

    def nvar(self):
        return self._dims()[1]

    def ploidy(self):
        return self._dims()[2]

    def __repr__(self):
        d = self._dims()
        return (f"SeqVarGDSClass: {self.filename} "
                f"(selected {d[3]}/{d[0]} samples, {d[4]}/{d[1]} variants)")


def seqOpen(filename, readonly=True, allow_dup=False):
    """Open a SeqArray GDS file and initialise its selection to all."""
    gds = pygds.gdsfile()
    gds.open(str(filename), readonly=readonly, allow_dup=allow_dup)
    f = SeqVarGDSClass(gds)
    cclib.file_init(f.fileid)
    return f


def seqClose(f):
    """Close the underlying GDS file."""
    f.gds.close()


class TVarData:
    """Result of :func:`seqGetData` for a variable-length info/format field.

    ``length`` is the per-selected-variant entry count; ``data`` is the
    concatenated values across the selected variants.
    """
    __slots__ = ("length", "data")

    def __init__(self, length, data):
        self.length = np.asarray(length, dtype=np.int32)
        self.data = data

    def __repr__(self):
        return f"TVarData({len(self.length)} variants)"


_VARIANT_FIELDS = ("variant.id", "position", "chromosome", "allele",
                   "annotation/id", "annotation/qual", "annotation/filter")


def _exist(f, path):
    return f.gds.root().exist(path)


def _offsets(cnt):
    o = np.zeros(len(cnt) + 1, dtype=np.int64)
    if len(cnt):
        o[1:] = np.cumsum(cnt)
    return o


def _read_info(f, path, svar):
    node = f.gds.index(path)
    field = path.rsplit("/", 1)[1]
    idxpath = path[:len(path) - len(field)] + "@" + field
    if not _exist(f, idxpath):
        vmask = np.zeros(f.nvar(), dtype=bool); vmask[svar] = True
        return np.asarray(node.readex([vmask]))           # fixed: one per variant
    cnt = np.asarray(f.gds.index(idxpath).read()).astype(np.int64)
    full = np.asarray(node.read())
    off = _offsets(cnt)
    lens = cnt[svar]
    if np.all(lens == 1):
        return full[off[svar]]
    pieces = [full[off[v]:off[v + 1]] for v in svar]
    return TVarData(lens, np.concatenate(pieces) if pieces else full[:0])


def _read_format(f, path, svar, smask):
    field = path[len("annotation/format/"):]
    if field.endswith("/data"):
        field = field[:-len("/data")]
    base = "annotation/format/" + field
    node = f.gds.index(base + "/data")
    idxpath = base + "/@data"
    cnt = (np.ones(f.nvar(), dtype=np.int64) if not _exist(f, idxpath)
           else np.asarray(f.gds.index(idxpath).read()).astype(np.int64))
    off = _offsets(cnt)
    cols = (np.concatenate([np.arange(off[v], off[v + 1]) for v in svar])
            if len(svar) else np.array([], dtype=np.int64))
    total_cols = int(node.description()["dim"][0])       # numpy [cols, sample]
    colmask = np.zeros(total_cols, dtype=bool); colmask[cols] = True
    mat = np.asarray(node.readex([colmask, smask]))       # (ncols, nSampSel)
    if np.all(cnt[svar] == 1):
        return mat                                        # (variant, sample)
    return TVarData(cnt[svar], mat)


def seqGetData(f, name, use_raw=False):
    """Read field ``name`` for the current selection — fully SEXP-free.

    Genotype-family fields drive the engine's CApply_Variant_* readers straight
    into numpy (``cclib.genotype``/``cclib.dosage``); else a plain pygds read honoured
    against the current selection masks.  Returns a numpy array (multi-dimensional
    fields in numpy C-order, reversed vs R's column-major dims), a list, a
    :class:`TVarData` (variable-length info/format), or ``None``.  Missing integer
    entries use R's ``NA_INTEGER`` (-2147483648); see :func:`na_mask`.
    """
    fid = f.fileid
    if not use_raw:
        if name == "genotype":
            return cclib.genotype(fid)
        if name == "$dosage":
            return cclib.dosage(fid, 0)
        if name == "$dosage_alt":
            return cclib.dosage(fid, 1)

    vmask = np.asarray(cclib.get_variant_sel(fid)).astype(bool)
    smask = np.asarray(cclib.get_sample_sel(fid)).astype(bool)
    svar = np.nonzero(vmask)[0]
    gds = f.gds

    # ---- sample-level ------------------------------------------------------
    if name == "sample.id":
        return np.asarray(gds.index("sample.id").readex([smask]))
    if name == "$sample_index":
        return np.nonzero(smask)[0]
    if name == "$variant_index":
        return svar

    # ---- fixed variant-level (pygds applies factor levels for filter) ------
    if name in _VARIANT_FIELDS:
        return np.asarray(gds.index(name).readex([vmask]))

    # ---- derived -----------------------------------------------------------
    if name in ("$num_allele", "$ref", "$alt", "$chrom_pos"):
        if name == "$chrom_pos":
            ch = np.asarray(gds.index("chromosome").readex([vmask]))
            ps = np.asarray(gds.index("position").readex([vmask]))
            return np.array([f"{c}:{p}" for c, p in zip(ch, ps)], dtype=object)
        al = np.asarray(gds.index("allele").readex([vmask]))
        if name == "$num_allele":
            return np.array([str(a).count(",") + 1 for a in al], dtype=np.int32)
        if name == "$ref":
            return np.array([str(a).split(",", 1)[0] for a in al], dtype=object)
        return np.array([(str(a).split(",", 1) + [""])[1] for a in al], dtype=object)

    # ---- phase (Bit1, numpy [variant, sample]) -----------------------------
    if name == "phase":
        if not _exist(f, "phase/data"):
            return None
        return np.asarray(gds.index("phase/data").readex([vmask, smask]))

    # ---- annotations -------------------------------------------------------
    if name.startswith("sample.annotation/"):
        return np.asarray(gds.index(name).readex([smask]))
    if name.startswith("annotation/info/"):
        return _read_info(f, name, svar)
    if name.startswith("annotation/format/"):
        return _read_format(f, name, svar, smask)

    raise ValueError(f"seqGetData: unsupported field name '{name}'")


def seqApply(fun, f, name, margin="by.variant", as_is="list", bsize=1, **kwargs):
    """Apply ``fun`` over the selected variants (or samples), ``bsize`` per call.

    SEXP-free: each block sets a sub-selection and reads via the native
    :func:`seqGetData`.  ``name`` is a field name or a list of names (``fun`` then
    receives a tuple).  ``margin`` is ``"by.variant"`` (default) or ``"by.sample"``;
    ``as_is`` is ``"list"`` (collect), ``"unlist"`` (concatenate) or ``"none"``.
    """
    by_variant = margin == "by.variant"
    if not by_variant and margin != "by.sample":
        raise ValueError('margin must be "by.variant" or "by.sample"')
    s0, v0 = seqGetFilter(f)                       # save selection
    units = np.nonzero(v0 if by_variant else s0)[0]
    results = []
    try:
        i = 0
        while i < len(units):
            blk = units[i:i + bsize]
            m = np.zeros(len(v0) if by_variant else len(s0), dtype=bool)
            m[blk] = True
            if by_variant:
                seqSetFilter(f, variant_sel=m, verbose=False)
            else:
                seqSetFilter(f, sample_sel=m, verbose=False)
            data = (seqGetData(f, name) if isinstance(name, str)
                    else tuple(seqGetData(f, n) for n in name))
            r = fun(*data, **kwargs) if isinstance(data, tuple) else fun(data, **kwargs)
            if as_is != "none":
                results.append(r)
            i += bsize
    finally:
        seqSetFilter(f, sample_sel=s0, variant_sel=v0, verbose=False)
    if as_is == "none":
        return None
    if as_is == "list":
        return results
    if as_is == "unlist":
        if not results:
            return np.array([])
        return np.concatenate([np.atleast_1d(np.asarray(r)).ravel() for r in results])
    raise ValueError(f"unsupported as_is={as_is!r}")


def seqBlockApply(fun, f, name, margin="by.variant", as_is="list", bsize=1024,
                  **kwargs):
    """Like :func:`seqApply` but a block of ``bsize`` units per call (default 1024)."""
    return seqApply(fun, f, name, margin=margin, as_is=as_is, bsize=bsize, **kwargs)


def _parallel_worker(arg):
    filename, units, sel0, name, margin, as_is, kwargs = arg
    by_variant = margin == "by.variant"
    f = seqOpen(filename, allow_dup=True)
    try:
        s0, v0 = sel0
        # restore the outer filter, then restrict to this worker's unit chunk
        seqSetFilter(f, sample_sel=s0, variant_sel=v0, verbose=False)
        full = v0 if by_variant else s0
        m = np.zeros(len(full), dtype=bool)
        m[units] = True
        m &= full
        if by_variant:
            seqSetFilter(f, variant_sel=m, verbose=False)
        else:
            seqSetFilter(f, sample_sel=m, verbose=False)
        return seqApply(_PARALLEL_FUN, f, name, margin=margin, as_is=as_is, **kwargs)
    finally:
        seqClose(f)


_PARALLEL_FUN = None


def _set_parallel_fun(fun):
    global _PARALLEL_FUN
    _PARALLEL_FUN = fun


def seqParallel(fun, f, name, margin="by.variant", as_is="list", ncpu=2,
                **kwargs):
    """Parallel :func:`seqApply` over ``ncpu`` worker processes.

    The current selection is split into ``ncpu`` contiguous unit-chunks; each
    worker re-opens the file (``allow_dup=True``), applies ``fun`` to its chunk,
    and the results are concatenated in order.  ``fun`` must be importable by the
    workers (a module-level function — not a lambda/closure), the same constraint
    R's ``seqParallel`` places on cluster functions.  ``as_is="unlist"`` returns a
    single concatenated array; ``"list"`` a flat list.
    """
    import multiprocessing as mp

    by_variant = margin == "by.variant"
    s0, v0 = seqGetFilter(f)
    units = np.nonzero(v0 if by_variant else s0)[0]
    if ncpu <= 1 or len(units) <= 1:
        return seqApply(fun, f, name, margin=margin, as_is=as_is, **kwargs)
    chunks = [c for c in np.array_split(units, ncpu) if len(c)]
    args = [(f.filename, c.tolist(), (s0, v0), name, margin, as_is, kwargs)
            for c in chunks]
    _set_parallel_fun(fun)            # fork inherits this -> fun need not be picklable
    ctx = mp.get_context("fork")
    with ctx.Pool(len(chunks)) as pool:
        parts = pool.map(_parallel_worker, args)
    if as_is == "none":
        return None
    if as_is == "unlist":
        parts = [np.atleast_1d(np.asarray(p)) for p in parts if p is not None and len(np.atleast_1d(np.asarray(p)))]
        return np.concatenate(parts) if parts else np.array([])
    out = []
    for p in parts:
        out.extend(p if isinstance(p, list) else list(p))
    return out


def na_mask(arr):
    """Boolean mask of R-``NA_INTEGER`` missing entries in an integer array."""
    a = np.asarray(arr)
    return a == _NA_INT


def seqSetFilter(f, sample_sel=None, variant_sel=None, intersect=False,
                 verbose=True):
    """Set the sample and/or variant selection from boolean masks.

    ``sample_sel`` / ``variant_sel`` are full-length boolean arrays (or None to
    leave that dimension unchanged).  ``intersect=True`` ANDs with the current
    selection instead of replacing it.
    """
    flag = 1 if intersect else 0
    if sample_sel is not None:
        cclib.set_sample(f.fileid, np.ascontiguousarray(sample_sel, dtype=bool), flag)
    if variant_sel is not None:
        cclib.set_variant(f.fileid, np.ascontiguousarray(variant_sel, dtype=bool), flag)
    if verbose:
        _report(f)
    return f


def seqResetFilter(f, sample=True, variant=True, verbose=True):
    """Reset the selection to all samples and/or all variants."""
    cclib.reset_filter(f.fileid, 1 if sample else 0, 1 if variant else 0)
    if verbose:
        _report(f)
    return f


def seqGetFilter(f):
    """Return the current ``(sample_sel, variant_sel)`` boolean masks."""
    s = np.asarray(cclib.get_sample_sel(f.fileid)).astype(bool)
    v = np.asarray(cclib.get_variant_sel(f.fileid)).astype(bool)
    return s, v


def seqSummary(f, verbose=True):
    """Summarise the file: sample/variant counts and the current selection.

    The selection masks returned by the engine are full-length, so their lengths
    give the total sample/variant counts.
    """
    s, v = seqGetFilter(f)
    info = dict(num_samples=len(s), num_variants=len(v),
                num_selected_samples=int(s.sum()),
                num_selected_variants=int(v.sum()))
    if verbose:
        print("SeqArray GDS:", f.filename)
        print("  # of samples: ", info["num_samples"])
        print("  # of variants:", info["num_variants"])
        print("  selected:", info["num_selected_samples"], "samples,",
              info["num_selected_variants"], "variants")
    return info


def seqNumAllele(f):
    """Number of alleles per selected variant (from the ``allele`` strings)."""
    al = seqGetData(f, "allele")
    return np.array([str(a).count(",") + 1 for a in al], dtype=np.int32)


def _geno_counts(f, allele_index):
    """(count of ``allele_index``, count of non-missing alleles) per variant.

    Genotype comes from the real SeqArray engine; the reduction is numpy.
    Shape of g is (variant, sample, ploidy) with R NA_INTEGER for missing.
    """
    g = np.asarray(seqGetData(f, "genotype"))
    miss = na_mask(g)
    nv = g.shape[0]
    flat = g.reshape(nv, -1)
    fmiss = miss.reshape(nv, -1)
    valid = ~fmiss
    total = valid.sum(axis=1)
    cnt = ((flat == allele_index) & valid).sum(axis=1)
    return cnt.astype(np.int64), total.astype(np.int64)


def seqAlleleCount(f, ref_allele=0):
    """Per-variant count of allele ``ref_allele`` (default reference)."""
    cnt, _ = _geno_counts(f, ref_allele)
    return cnt


def seqAlleleFreq(f, ref_allele=0):
    """Per-variant frequency of allele ``ref_allele`` (default reference)."""
    cnt, total = _geno_counts(f, ref_allele)
    with np.errstate(divide="ignore", invalid="ignore"):
        freq = np.where(total > 0, cnt / total, np.nan)
    return freq


def seqMissing(f, per_variant=True):
    """Missing-call rate per variant (``per_variant=True``) or per sample."""
    g = np.asarray(seqGetData(f, "genotype"))
    miss = na_mask(g)                       # (variant, sample, ploidy)
    any_miss = miss.any(axis=2)             # (variant, sample): call is missing
    if per_variant:
        return any_miss.mean(axis=1)
    return any_miss.mean(axis=0)


def seqExampleFileName(kind="gds"):
    """Path to a bundled SeqArray example file ("gds", "1KG", or "vcf")."""
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "..", "SeqArray", "inst", "extdata")
    name = {"gds": "CEU_Exon.gds",
            "1KG": "1KG_phase1_release_v3_chr22.gds",
            "vcf": "CEU_Exon.vcf.gz"}.get(kind)
    if name is None:
        raise ValueError(f"unknown example kind '{kind}'")
    return os.path.normpath(os.path.join(base, name))


def _report(f):
    s, v = seqGetFilter(f)
    print(f"# of selected samples: {int(s.sum())}")
    print(f"# of selected variants: {int(v.sum())}")


__all__ = [
    "SeqVarGDSClass", "TVarData",
    "seqOpen", "seqClose", "seqGetData", "na_mask",
    "seqSetFilter", "seqResetFilter", "seqGetFilter",
    "seqApply", "seqBlockApply", "seqParallel",
    "seqSummary", "seqExampleFileName",
    "seqNumAllele", "seqAlleleCount", "seqAlleleFreq", "seqMissing",
    "seqVCF2GDS", "seqGDS2VCF", "seqAddValue", "seqDelete", "seqRecompress",
]
