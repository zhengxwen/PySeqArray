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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("All engine (B1/B4) tests passed.")
