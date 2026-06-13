"""M1 read-path validation for PySeqArray.

Oracle values are the byte-for-byte cross-checks recorded from R SeqArray /
JSeqArray on the bundled CEU_Exon.gds (90 samples x 1348 variants, ploidy 2):

    non-missing genotype sum = 32683
    number of missing alleles = 17216
    raw bit2 sum (incl. missing=3) = 84331  ->  32683 + 3*17216
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


def test_open_summary():
    f = _open()
    try:
        s = sa.seqSummary(f, verbose=False)
        assert s["nsamp"] == 90
        assert s["nvar"] == 1348
        assert s["ploidy"] == 2
    finally:
        sa.seqClose(f)


def test_basic_fields():
    f = _open()
    try:
        assert len(sa.seqGetData(f, "sample.id")) == 90
        assert len(sa.seqGetData(f, "variant.id")) == 1348
        pos = sa.seqGetData(f, "position")
        assert len(pos) == 1348 and np.issubdtype(pos.dtype, np.integer)
        chrom = sa.seqGetData(f, "chromosome")
        assert len(chrom) == 1348
    finally:
        sa.seqClose(f)


def test_genotype_oracle():
    f = _open()
    try:
        g = sa.seqGetData(f, "genotype")
        assert g.shape == (2, 90, 1348)            # (ploidy, sample, variant)
        data = np.ma.getdata(g)
        mask = np.ma.getmaskarray(g)
        nonmiss_sum = int(data[~mask].sum())
        n_missing = int(mask.sum())
        assert nonmiss_sum == 32683, nonmiss_sum
        assert n_missing == 17216, n_missing
        # raw-sum identity
        assert nonmiss_sum + 3 * n_missing == 84331
    finally:
        sa.seqClose(f)


def test_pseudo_fields():
    f = _open()
    try:
        na = sa.seqGetData(f, "$num_allele")
        assert na.min() >= 1
        ref = sa.seqGetData(f, "$ref")
        alt = sa.seqGetData(f, "$alt")
        assert len(ref) == len(alt) == 1348
        cp = sa.seqGetData(f, "$chrom_pos")
        assert ":" in cp[0]
    finally:
        sa.seqClose(f)


def test_dosage():
    f = _open()
    try:
        d = sa.seqGetData(f, "$dosage")
        assert d.shape == (90, 1348)               # (sample, variant)
    finally:
        sa.seqClose(f)


def test_filter_and_apply():
    f = _open()
    try:
        sa.seqSetFilter(f, sample_sel=np.arange(10),
                        variant_sel=np.arange(100), verbose=False)
        g = sa.seqGetData(f, "genotype")
        assert g.shape == (2, 10, 100)
        # seqApply: alt-allele count per variant over the filtered set
        counts = sa.seqApply(
            lambda gt: int((np.ma.getdata(gt) == 1).sum()),
            f, "genotype", margin="by.variant", as_is="list")
        assert len(counts) == 100
        sa.seqResetFilter(f, verbose=False)
        assert f.n_sel_variant() == 1348
    finally:
        sa.seqClose(f)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("All M1 tests passed.")
