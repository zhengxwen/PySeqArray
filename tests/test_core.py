# Core functional tests for native PySeqArray, validated against R SeqArray.
import numpy as np
import PySeqArray as ps


def _open():
    return ps.seqOpen(ps.seqExampleFileName())


def test_read_dims():
    f = _open()
    assert f.NumSample() == 1092
    assert f.NumVariant() == 19773
    g = np.asarray(f.GetData('genotype'))
    assert g.shape == (19773, 1092, 2) and g.dtype == np.uint8
    ps.seqClose(f)


def test_allele_freq_matches_R():
    f = _open()
    af = ps.seqAlleleFreq(f, ref_allele=0)
    # R: seqAlleleFreq(f, ref.allele=0L)
    assert np.allclose(af[:4], [0.695055, 0.943223, 0.999542, 0.999542],
        atol=1e-6)
    assert abs(float(np.sum(af)) - 18361.31) < 0.01
    ps.seqClose(f)


def test_allele_count_matches_R():
    f = _open()
    ac = ps.seqAlleleCount(f, ref_allele=0)
    assert ac[:6].tolist() == [1518, 2060, 2183, 2183, 2050, 2182]
    acl = ps.seqAlleleCount(f, ref_allele=None)
    assert acl[0].tolist() == [1518, 666]   # 1518+666 == 2*1092
    ps.seqClose(f)


def test_missing_and_numallele():
    f = _open()
    assert float(np.sum(ps.seqMissing(f, per_variant=True))) == 0.0
    assert len(ps.seqMissing(f, per_variant=False)) == 1092
    na = ps.seqNumAllele(f)
    assert np.all(na == 2)
    ps.seqClose(f)


def test_filter_chrom_pos():
    f = _open()
    ps.seqSetFilterChrom(f, include="22", frm_bp=17000000, to_bp=18000000,
        verbose=False)
    assert f.NumVariant(selected=True) == 521          # R: 521
    assert abs(float(np.sum(ps.seqAlleleFreq(f, 0))) - 476.2459) < 0.01
    ps.seqResetFilter(f, verbose=False)
    pos = np.asarray(ps.seqGetData(f, "position"))[[0, 49, 99, 499, 999]]
    ps.seqSetFilterPos(f, chrom=["22"] * 5, pos=pos, verbose=False)
    assert f.NumVariant(selected=True) == 5
    ps.seqClose(f)


def test_filter_by_id_subset():
    f = _open()
    sid = np.asarray(ps.seqGetData(f, 'sample.id'))[:10]
    vid = np.asarray(ps.seqGetData(f, 'variant.id'))[:100]
    f.FilterSet(sample_id=sid, variant_id=vid, verbose=False)
    g = np.asarray(ps.seqGetData(f, 'genotype'))
    assert g.shape == (100, 10, 2)
    ps.seqClose(f)


def test_apply_block():
    f = _open()
    # per-block alt-allele count, summed -> total alt alleles
    parts = f.Apply('genotype',
        lambda g: int(np.sum(np.asarray(g) > 0)), asis='list')
    assert sum(parts) == 3083127      # == count of genotype value 1
    ps.seqClose(f)


def test_parallel_matches_serial():
    f = _open()
    serial = np.asarray(ps.seqAlleleFreq(f, 0))
    par = np.asarray(ps.seqParallel(f,
        lambda fl, p: ps.seqAlleleFreq(fl, 0), ncpu=4, combine='unlist'))
    assert np.allclose(par, serial, equal_nan=True)
    ps.seqClose(f)


def test_gds2vcf_body():
    # Export a small subset; body validated against R seqGDS2VCF.
    import tempfile, os
    f = _open()
    f.FilterSet(sample_id=np.asarray(ps.seqGetData(f, 'sample.id'))[:4],
        variant_id=np.asarray(ps.seqGetData(f, 'variant.id'))[:6],
        verbose=False)
    out = os.path.join(tempfile.gettempdir(), 'pyseq_test.vcf')
    ps.seqGDS2VCF(f, out)
    body = [ln for ln in open(out).read().splitlines()
        if not ln.startswith('##')]
    ps.seqClose(f)
    assert body[0].startswith('#CHROM\tPOS\tID\tREF\tALT')
    assert body[1].split('\t')[:8] == \
        ['22', '16051497', 'rs141578542', 'A', 'G', '100', 'PASS', '.']
    assert body[1].split('\t')[9:] == ['0|0', '1|0', '1|0', '0|1']


_MINI_VCF = """##fileformat=VCFv4.1
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##INFO=<ID=AA,Number=.,Type=String,Description="Ancestral Allele">
##INFO=<ID=H2,Number=0,Type=Flag,Description="HapMap2">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">
##FILTER=<ID=PASS,Description="All filters passed">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3
1\t100\trs1\tA\tG\t50\tPASS\tDP=30;AA=A;H2\tGT:DP\t0|0:10\t1|0:8\t1/1:12
1\t200\trs2\tC\tT\t.\tPASS\tDP=20;AA=C\tGT:DP\t0/0:5\t0|1:7\t./.:0
2\t300\trs3\tG\tA,C\t99\tPASS\tDP=40\tGT:DP\t0|1:9\t2|0:6\t1|2:11
"""


def test_vcf_import_roundtrip():
    import tempfile, os
    d = tempfile.mkdtemp()
    vcf = os.path.join(d, 'mini.vcf')
    gds = os.path.join(d, 'mini.gds')
    open(vcf, 'w').write(_MINI_VCF)
    # header parse
    h = ps.seqVCF_Header(vcf)
    assert h['fileformat'] == 'VCFv4.1'
    assert h['samples'] == ['S1', 'S2', 'S3']
    assert set(h['info']) == {'DP', 'AA', 'H2'}
    # import
    ps.seqVCF2GDS(vcf, gds, verbose=False)
    f = ps.seqOpen(gds, allow_dup=True)
    assert f.NumVariant() == 3 and f.NumSample() == 3
    assert np.asarray(ps.seqGetData(f, 'position')).tolist() == [100, 200, 300]
    assert np.asarray(ps.seqGetData(f, 'allele')).tolist() == \
        ['A,G', 'C,T', 'G,A,C']
    g = np.asarray(ps.seqGetData(f, 'genotype'))
    assert g.shape == (3, 3, 2)
    # variant 1: 0|0, 1|0, 1/1
    assert g[0].tolist() == [[0, 0], [1, 0], [1, 1]]
    # variant 2 sample 3 missing -> 255
    assert g[1, 2].tolist() == [255, 255]
    # variant 3: multi-allelic 0|1, 2|0, 1|2
    assert g[2].tolist() == [[0, 1], [2, 0], [1, 2]]
    # INFO DP fixed
    assert np.asarray(ps.seqGetData(f, 'annotation/info/DP')).tolist() == \
        [30, 20, 40]
    ps.seqClose(f)


def test_plink_bed_roundtrip():
    import tempfile, os
    d = tempfile.mkdtemp()
    vcf = os.path.join(d, 'mini.vcf')
    gds = os.path.join(d, 'mini.gds')
    open(vcf, 'w').write(_MINI_VCF)
    ps.seqVCF2GDS(vcf, gds, verbose=False)
    f = ps.seqOpen(gds, allow_dup=True)
    # keep only biallelic variants (drop the triallelic one)
    ps.seqSetFilterChrom(f, include='1', verbose=False)
    g0 = np.asarray(ps.seqGetData(f, 'genotype'))
    dose0 = np.where((g0 == 255).any(axis=2), -1, (g0 == 0).sum(axis=2))
    bp = os.path.join(d, 'plink')
    ps.seqGDS2BED(f, bp, verbose=False)
    ps.seqClose(f)
    assert os.path.exists(bp + '.bed') and os.path.exists(bp + '.bim')
    # import back
    gds2 = os.path.join(d, 'back.gds')
    ps.seqBED2GDS(bp, gds2, verbose=False)
    fb = ps.seqOpen(gds2, allow_dup=True)
    g1 = np.asarray(ps.seqGetData(fb, 'genotype'))
    dose1 = np.where((g1 == 255).any(axis=2), -1, (g1 == 0).sum(axis=2))
    ps.seqClose(fb)
    assert np.array_equal(dose0, dose1)


def test_snp_gds_roundtrip():
    import tempfile, os
    f = _open()
    f.FilterSet(variant_id=np.asarray(ps.seqGetData(f, 'variant.id'))[:20],
        sample_id=np.asarray(ps.seqGetData(f, 'sample.id'))[:10],
        verbose=False)
    g0 = np.asarray(ps.seqGetData(f, 'genotype'))
    dose0 = np.where((g0 == 255).any(axis=2), 3, (g0 == 0).sum(axis=2))
    d = tempfile.mkdtemp()
    snp = os.path.join(d, 'snp.gds')
    ps.seqGDS2SNP(f, snp, verbose=False)
    ps.seqClose(f)
    # SNP-GDS genotype is the [nSample x nSNP] ref-dosage matrix
    import pygds
    gg = pygds.gdsfile(); gg.open(snp, allow_dup=True)
    # pygds shape is the reverse of gdsfmt's; here it reads back (nSNP, nSamp)
    snp_geno = np.asarray(gg.root().index('genotype').read())
    gg.close()
    assert np.array_equal(snp_geno, dose0)
    # back to SeqArray GDS
    back = os.path.join(d, 'back.gds')
    ps.seqSNP2GDS(snp, back, verbose=False)
    fb = ps.seqOpen(back, allow_dup=True)
    g1 = np.asarray(ps.seqGetData(fb, 'genotype'))
    dose1 = np.where((g1 == 255).any(axis=2), 3, (g1 == 0).sum(axis=2))
    ps.seqClose(fb)
    assert np.array_equal(dose0, dose1)


def test_seqexport():
    import tempfile, os
    f = _open()
    f.FilterSet(variant_id=np.asarray(ps.seqGetData(f, 'variant.id'))[:30],
        sample_id=np.asarray(ps.seqGetData(f, 'sample.id'))[:15],
        verbose=False)
    af0 = ps.seqAlleleFreq(f, 0)
    out = os.path.join(tempfile.mkdtemp(), 'sub.gds')
    ps.seqExport(f, out, verbose=False)
    ps.seqClose(f)
    g = ps.seqOpen(out, allow_dup=True)
    assert g.NumVariant() == 30 and g.NumSample() == 15
    assert np.allclose(ps.seqAlleleFreq(g, 0), af0, equal_nan=True)
    ps.seqClose(g)


def test_seqmerge():
    import tempfile, os
    d = tempfile.mkdtemp()
    f = _open()
    vid = np.asarray(ps.seqGetData(f, 'variant.id'))
    af_full = ps.seqAlleleFreq(f, 0)
    f.FilterSet(variant_id=vid[:5000], verbose=False)
    ps.seqExport(f, os.path.join(d, 'p1.gds'), verbose=False)
    f.FilterReset(verbose=False)
    f.FilterSet(variant_id=vid[5000:], verbose=False)
    ps.seqExport(f, os.path.join(d, 'p2.gds'), verbose=False)
    ps.seqClose(f)
    out = os.path.join(d, 'merged.gds')
    ps.seqMerge([os.path.join(d, 'p1.gds'), os.path.join(d, 'p2.gds')], out,
        verbose=False)
    m = ps.seqOpen(out, allow_dup=True)
    assert m.NumVariant() == len(vid)
    assert np.allclose(ps.seqAlleleFreq(m, 0), af_full, equal_nan=True)
    ps.seqClose(m)


if __name__ == '__main__':
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    npass = 0
    for fn in fns:
        try:
            fn(); print("PASS", fn.__name__); npass += 1
        except Exception:
            print("FAIL", fn.__name__); traceback.print_exc()
    print(f"\n{npass}/{len(fns)} passed")
