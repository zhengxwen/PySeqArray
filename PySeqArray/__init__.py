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

    def __repr__(self):
        ns = int(np.asarray(cclib.get_sample_sel(self.fileid)).sum())
        nv = int(np.asarray(cclib.get_variant_sel(self.fileid)).sum())
        return f"SeqVarGDSClass: {self.filename} (selected {ns} samples, {nv} variants)"


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


def seqGetData(f, name, use_raw=False):
    """Read field ``name`` for the current selection, via the SeqArray engine.

    Returns a numpy array (multi-dimensional fields come back in numpy C-order,
    i.e. reversed relative to R's column-major dims), a list (for string / list
    fields), or ``None``.  Genotype/missing integer fields use R's ``NA_INTEGER``
    (-2147483648) sentinel; use :func:`na_mask` to locate missing entries.
    """
    return cclib.get_data(f.fileid, name, 1 if use_raw else 0)


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
    "SeqVarGDSClass",
    "seqOpen", "seqClose", "seqGetData", "na_mask",
    "seqSetFilter", "seqResetFilter", "seqGetFilter",
    "seqSummary", "seqExampleFileName",
    "seqNumAllele", "seqAlleleCount", "seqAlleleFreq", "seqMissing",
]
