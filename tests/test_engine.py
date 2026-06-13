"""B1/B4 validation: the real SeqArray C++ engine driven from Python via cclib.

Oracle values are byte-for-byte cross-checks from R SeqArray on the bundled
CEU_Exon.gds (90 samples x 1348 variants, ploidy 2):

    non-missing genotype sum = 32683
    missing alleles          = 17216
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "..", "pygds"))
sys.path.insert(0, os.path.join(HERE, ".."))

import PySeqArray as sa  # noqa: E402

GDS = os.path.join(HERE, "..", "..", "SeqArray", "inst", "extdata", "CEU_Exon.gds")


def _open():
    return sa.seqOpen(GDS)


def test_open_close_summary():
    f = _open()
    try:
        info = sa.seqSummary(f, verbose=False)
        assert info["num_samples"] == 90
        assert info["num_variants"] == 1348
    finally:
        sa.seqClose(f)


def test_genotype_oracle():
    f = _open()
    try:
        g = np.asarray(sa.seqGetData(f, "genotype"))
        assert g.shape == (1348, 90, 2)         # numpy C-order (rev of R 2x90x1348)
        nm = sa.na_mask(g)
        assert int(g[~nm].sum()) == 32683
        assert int(nm.sum()) == 17216
    finally:
        sa.seqClose(f)


def test_fields():
    f = _open()
    try:
        assert len(sa.seqGetData(f, "sample.id")) == 90
        assert len(np.asarray(sa.seqGetData(f, "position"))) == 1348
        al = sa.seqGetData(f, "allele")
        assert "," in al[0]
        ch = sa.seqGetData(f, "chromosome")
        assert len(ch) == 1348
    finally:
        sa.seqClose(f)


def test_filter():
    f = _open()
    try:
        sa.seqSetFilter(f, sample_sel=np.arange(90) < 10,
                        variant_sel=np.arange(1348) < 100, verbose=False)
        g = np.asarray(sa.seqGetData(f, "genotype"))
        assert g.shape == (100, 10, 2)
        ds = np.asarray(sa.seqGetData(f, "$dosage"))
        assert ds.shape == (100, 10)
        s, v = sa.seqGetFilter(f)
        assert s.sum() == 10 and v.sum() == 100
        sa.seqResetFilter(f, verbose=False)
        s, v = sa.seqGetFilter(f)
        assert s.sum() == 90 and v.sum() == 1348
    finally:
        sa.seqClose(f)


def test_native_seqgetdata_all_fields():
    """Fully SEXP-free seqGetData covers every field type under a filter, and the
    filtered genotype still sums to the per-block oracle."""
    f = _open()
    try:
        # AA is variable-length over the full file (dim 1328 < 1348) -> TVarData
        assert hasattr(sa.seqGetData(f, "annotation/info/AA"), "length")
        sa.seqSetFilter(f, sample_sel=np.arange(90) < 25,
                        variant_sel=np.arange(1348) < 300, verbose=False)
        assert np.asarray(sa.seqGetData(f, "genotype")).shape == (300, 25, 2)
        assert np.asarray(sa.seqGetData(f, "$dosage")).shape == (300, 25)
        assert np.asarray(sa.seqGetData(f, "phase")).shape == (300, 25)
        assert len(sa.seqGetData(f, "sample.id")) == 25
        assert sa.seqGetData(f, "annotation/filter")[0] == "PASS"   # factor -> str
        assert len(np.asarray(sa.seqGetData(f, "position"))) == 300
        assert np.asarray(sa.seqGetData(f, "annotation/info/DP")).shape == (300,)
        sa.seqResetFilter(f, verbose=False)
        # genotype over the full file still hits the oracle (native engine path)
        g = np.asarray(sa.seqGetData(f, "genotype"))
        assert int(g[~sa.na_mask(g)].sum()) == 32683
    finally:
        sa.seqClose(f)


def test_apply():
    """seqApply (SEXP-free, Python block loop) over native seqGetData."""
    f = _open()
    try:
        sa.seqSetFilter(f, variant_sel=np.arange(1348) < 50, verbose=False)
        # alt-allele count per variant
        counts = sa.seqApply(
            lambda g: int((np.asarray(g) == 1).sum()),
            f, "genotype", margin="by.variant", as_is="list")
        assert len(counts) == 50
        # equals the whole-block computation
        gb = np.asarray(sa.seqGetData(f, "genotype"))
        per_var = (gb == 1).reshape(50, -1).sum(axis=1)
        assert counts == per_var.tolist()
        sa.seqResetFilter(f, verbose=False)
    finally:
        sa.seqClose(f)


def test_summary_stats():
    f = _open()
    try:
        ac = sa.seqAlleleCount(f)
        assert ac[:3].tolist() == [110, 105, 148]
        af = sa.seqAlleleFreq(f)
        assert abs(float(np.nansum(af)) - 1144.146) < 1e-2
        mv = sa.seqMissing(f, per_variant=True)
        assert abs(float(mv.sum()) - 95.644) < 1e-2
        ms = sa.seqMissing(f, per_variant=False)
        assert abs(float(ms.sum()) - 6.386) < 1e-2
        na = sa.seqNumAllele(f)
        assert na.min() == 2 and na.max() == 3
    finally:
        sa.seqClose(f)


def _alt_count(g):
    return int((np.asarray(g) == 1).sum())


def test_parallel():
    """seqParallel (multiprocessing) equals seqApply."""
    f = _open()
    try:
        sa.seqSetFilter(f, variant_sel=np.arange(1348) < 200, verbose=False)
        seq = sa.seqApply(_alt_count, f, "genotype", as_is="list")
        par = sa.seqParallel(_alt_count, f, "genotype", as_is="list", ncpu=4)
        assert seq == par and len(seq) == 200
        sa.seqResetFilter(f, verbose=False)
    finally:
        sa.seqClose(f)


def test_vcf2gds_roundtrip():
    """seqVCF2GDS imports the example VCF; the produced GDS reads back to the
    same genotype oracle as the reference CEU_Exon.gds."""
    import tempfile
    out = os.path.join(tempfile.gettempdir(), "pyseq_vcf_rt.gds")
    if os.path.exists(out):
        os.remove(out)
    sa.seqVCF2GDS(sa.seqExampleFileName("vcf"), out, verbose=False)
    f = sa.seqOpen(out)
    try:
        assert f.nvar() == 1348 and f.nsamp() == 90
        g = np.asarray(sa.seqGetData(f, "genotype"))
        assert g.shape == (1348, 90, 2)
        nm = sa.na_mask(g)
        assert int(g[~nm].sum()) == 32683
        assert int(nm.sum()) == 17216
        assert sa.seqGetData(f, "allele")[0] == "T,C"
    finally:
        sa.seqClose(f)
        os.remove(out)


def test_gds2vcf_roundtrip():
    """seqGDS2VCF export then seqVCF2GDS re-import preserves the genotype."""
    import tempfile
    tmp = tempfile.gettempdir()
    vcf = os.path.join(tmp, "pyseq_exp.vcf.gz")
    gds = os.path.join(tmp, "pyseq_reimp.gds")
    for p in (vcf, gds):
        if os.path.exists(p):
            os.remove(p)
    f = sa.seqOpen(sa.seqExampleFileName("gds"))
    try:
        sa.seqGDS2VCF(f, vcf, verbose=False)
    finally:
        sa.seqClose(f)
    sa.seqVCF2GDS(vcf, gds, verbose=False)
    f2 = sa.seqOpen(gds)
    try:
        g = np.asarray(sa.seqGetData(f2, "genotype"))
        nm = sa.na_mask(g)
        assert int(g[~nm].sum()) == 32683
        assert int(nm.sum()) == 17216
    finally:
        sa.seqClose(f2)
        for p in (vcf, gds):
            os.remove(p)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("All engine (B1/B4) tests passed.")
