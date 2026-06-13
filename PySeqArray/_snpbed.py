"""PLINK BED <-> SeqArray GDS and SNPRelate GDS <-> SeqArray GDS conversion.

All four converters are PySeqArray-owned pure Python over :mod:`pygds` and numpy
(the SeqArray C++ engine is not needed: these formats hold simple biallelic
allele-dosage genotypes that pack/unpack with vectorised numpy).  This mirrors
the validated JSeqArray.jl ``snpbed.jl`` implementation 1:1.

Encoding conventions
--------------------
* PLINK1 ``.bed`` (SNP-major): per SNP ``ceil(nSamp/4)`` bytes, two bits per
  sample, codes ``00=A1A1`` ``01=missing`` ``10=het`` ``11=A2A2``.  A1 is the
  alternate allele, A2 the reference allele (so allele = ``"A2,A1"``).
* SNPRelate (snpgds): ``genotype`` is the reference-allele dosage 0/1/2 with
  3 = missing, oriented ``[sample, snp]`` (marked by the ``sample.order`` attr).
"""

import numpy as np
import pygds

# genotype/data bit2 value for a missing allele (== the 2-bit max)
_GENO_MISS = 3
_NA_INT = np.int32(-2147483648)


# ---------------------------------------------------------------------------
# GDS skeleton shared by seqBED2GDS / seqSNP2GDS
# ---------------------------------------------------------------------------

def _make_seq_skeleton(out_fn, sample_id, chrom, pos, rsid, allele, graw,
                       compress):
    """Create a biallelic diploid SeqArray GDS from already-decoded fields.

    ``graw`` is a ``(nvar, nSamp, ploidy)`` uint8 array of allele indices
    (3 = missing), in pygds CoreArray-native dim order (variant dimension
    first), matching what :func:`PySeqArray.seqGetData` reads back.
    """
    nv = len(pos)
    w = pygds.gdsfile()
    w.create(str(out_fn))
    try:
        root = w.root()
        root.putattr("FileFormat", "SEQ_ARRAY")
        desc = root.addfolder("description")
        desc.putattr("vcf.fileformat", "VCFv4.2")

        root.add("sample.id", val=list(sample_id), storage="string",
                 compress=compress)
        root.add("variant.id", val=np.arange(1, nv + 1, dtype=np.int32),
                 storage="int32", compress=compress)
        root.add("position", val=np.asarray(pos, dtype=np.int32),
                 storage="int32", compress=compress)
        root.add("chromosome", val=list(chrom), storage="string",
                 compress=compress)
        root.add("allele", val=list(allele), storage="string",
                 compress=compress)

        geno = root.addfolder("genotype")
        geno.putattr("VariableName", "GT")
        geno.putattr("Description", "Genotype")
        geno.add("data", val=np.ascontiguousarray(graw, dtype=np.uint8),
                 storage="bit2", compress=compress)
        geno.add("@data", val=np.ones(nv, dtype=np.uint8), storage="uint8",
                 compress=compress, visible=False)

        annot = root.addfolder("annotation")
        annot.add("id", val=list(rsid), storage="string", compress=compress)
        annot.add("qual", val=np.full(nv, np.nan, dtype=np.float32),
                  storage="float", compress=compress)
        filt = annot.add("filter", val=np.ones(nv, dtype=np.int32),
                         storage="int32", compress=compress)
        filt.putattr("R.class", "factor")
        filt.putattr("R.levels", ["PASS"])
        filt.putattr("Description", ["All filters passed"])
        annot.addfolder("info")
        annot.addfolder("format")
        root.addfolder("sample.annotation")
        w.sync()
    finally:
        w.close()
    return str(out_fn)


# ---------------------------------------------------------------------------
# PLINK BED
# ---------------------------------------------------------------------------

def seqBED2GDS(bed_prefix, out_fn, compress="LZMA_RA", verbose=True):
    """Convert PLINK1 ``.bed``/``.bim``/``.fam`` (sharing ``bed_prefix``) into a
    SeqArray GDS file ``out_fn`` (biallelic, diploid).  A1 = alt, A2 = ref."""
    bed_prefix = str(bed_prefix)
    # .fam -> sample ids (2nd column = within-family id)
    sid = []
    with open(bed_prefix + ".fam") as fh:
        for line in fh:
            t = line.split()
            if t:
                sid.append(t[1])
    ns = len(sid)

    # .bim -> chrom, rsid, position, A1(alt), A2(ref)
    chrom, rsid, pos, a1, a2 = [], [], [], [], []
    with open(bed_prefix + ".bim") as fh:
        for line in fh:
            t = line.split()
            if not t:
                continue
            chrom.append(t[0]); rsid.append(t[1]); pos.append(int(t[3]))
            a1.append(t[4]); a2.append(t[5])
    nv = len(pos)
    allele = [f"{a2[v]},{a1[v]}" for v in range(nv)]

    # .bed -> raw genotype bytes
    with open(bed_prefix + ".bed", "rb") as fh:
        raw = fh.read()
    if len(raw) < 3 or raw[0] != 0x6c or raw[1] != 0x1b:
        raise ValueError("not a PLINK .bed file")
    if raw[2] != 0x01:
        raise ValueError("only SNP-major .bed is supported")

    nb = (ns + 3) // 4
    body = np.frombuffer(raw, dtype=np.uint8, offset=3, count=nv * nb)
    body = body.reshape(nv, nb)
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
    codes = ((body[:, :, None] >> shifts) & 0x3).reshape(nv, nb * 4)[:, :ns]

    # code -> the two haplotype allele indices (A1=alt=1, A2=ref=0; 3=missing)
    #   0:A1A1 -> (1,1)   1:miss -> (3,3)   2:het -> (0,1)   3:A2A2 -> (0,0)
    hap0 = np.array([1, 3, 0, 0], dtype=np.uint8)[codes]
    hap1 = np.array([1, 3, 1, 0], dtype=np.uint8)[codes]
    graw = np.stack([hap0, hap1], axis=2)          # (nvar, nSamp, ploidy)

    _make_seq_skeleton(out_fn, sid, chrom, pos, rsid, allele, graw, compress)
    if verbose:
        print(f"seqBED2GDS: {nv} variants x {ns} samples -> {out_fn}")
    return str(out_fn)


def seqGDS2BED(f, out_prefix, verbose=True):
    """Write PLINK1 ``out_prefix.{bed,bim,fam}`` for the current selection of a
    SeqArray GDS handle ``f`` (biallelic variants only).  A1 = alt, A2 = ref."""
    from . import seqGetData

    out_prefix = str(out_prefix)
    sid = np.asarray(seqGetData(f, "sample.id")).astype(str)
    chrom = np.asarray(seqGetData(f, "chromosome")).astype(str)
    pos = np.asarray(seqGetData(f, "position"))
    rsid = np.asarray(seqGetData(f, "annotation/id")).astype(str)
    allele = np.asarray(seqGetData(f, "allele")).astype(str)
    d = np.asarray(seqGetData(f, "$dosage_alt"))    # alt-allele count (nvar,nSamp)
    nv, ns = d.shape

    # .fam
    with open(out_prefix + ".fam", "w") as io:
        for s in sid:
            io.write(f"{s} {s} 0 0 0 -9\n")
    # .bim
    with open(out_prefix + ".bim", "w") as io:
        for v in range(nv):
            parts = allele[v].split(",", 1)
            a2 = parts[0]
            a1 = parts[1] if len(parts) > 1 else "0"
            vid = rsid[v] if rsid[v] not in ("", "nan") else f"v{v + 1}"
            io.write(f"{chrom[v]}\t{vid}\t0\t{int(pos[v])}\t{a1}\t{a2}\n")
    # .bed (SNP-major): alt count 2->00(A1A1) 1->10(het) 0->11(A2A2) miss->01
    codes = np.full((nv, ns), 3, dtype=np.uint8)    # default a==0 -> A2A2
    codes[d == 1] = 2
    codes[d == 2] = 0
    codes[d == _NA_INT] = 1                          # missing
    pad = (-ns) % 4
    if pad:
        codes = np.pad(codes, ((0, 0), (0, pad)))
    nb = (ns + 3) // 4
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
    packed = (codes.reshape(nv, nb, 4) << shifts).sum(axis=2).astype(np.uint8)
    with open(out_prefix + ".bed", "wb") as io:
        io.write(bytes([0x6c, 0x1b, 0x01]))
        io.write(packed.tobytes())
    if verbose:
        print(f"seqGDS2BED: {nv} variants x {ns} samples -> "
              f"{out_prefix}.{{bed,bim,fam}}")
    return out_prefix


# ---------------------------------------------------------------------------
# SNPRelate GDS
# ---------------------------------------------------------------------------

def seqGDS2SNP(f, out_fn, compress="LZMA_RA", verbose=True):
    """Export the current selection of a SeqArray GDS handle ``f`` to a SNPRelate
    GDS file.  ``genotype`` is the reference-allele dosage ``[sample, snp]``
    (0/1/2, 3 = missing)."""
    from . import seqGetData

    d = np.asarray(seqGetData(f, "$dosage"))         # ref dosage (nvar, nSamp)
    geno = np.where(d == _NA_INT, 3, d).astype(np.uint8)
    geno = np.ascontiguousarray(geno.T)              # SNPRelate dim [sample, snp]
    ns, nv = geno.shape

    sid = np.asarray(seqGetData(f, "sample.id")).astype(str)
    rsid = np.asarray(seqGetData(f, "annotation/id")).astype(str)
    pos = np.asarray(seqGetData(f, "position"), dtype=np.int32)
    chrom = np.asarray(seqGetData(f, "chromosome")).astype(str)
    allele = [a.replace(",", "/") for a in
              np.asarray(seqGetData(f, "allele")).astype(str)]

    w = pygds.gdsfile()
    w.create(str(out_fn))
    try:
        root = w.root()
        root.add("sample.id", val=list(sid), storage="string", compress=compress)
        root.add("snp.id", val=np.arange(1, nv + 1, dtype=np.int32),
                 storage="int32", compress=compress)
        root.add("snp.rs.id", val=list(rsid), storage="string", compress=compress)
        root.add("snp.position", val=pos, storage="int32", compress=compress)
        root.add("snp.chromosome", val=list(chrom), storage="string",
                 compress=compress)
        root.add("snp.allele", val=list(allele), storage="string",
                 compress=compress)
        gn = root.add("genotype", val=geno, storage="bit2", compress=compress)
        gn.putattr("sample.order", None)             # [sample, snp] orientation
        w.sync()
    finally:
        w.close()
    if verbose:
        print(f"seqGDS2SNP: {nv} SNPs x {ns} samples -> {out_fn}")
    return str(out_fn)


def seqSNP2GDS(snp_fn, out_fn, compress="LZMA_RA", verbose=True):
    """Convert a SNPRelate GDS file ``snp_fn`` into a SeqArray GDS file
    (biallelic, diploid)."""
    g = pygds.gdsfile()
    g.open(str(snp_fn))
    try:
        root = g.root()
        sid = np.asarray(root.index("sample.id").read()).astype(str)
        pos = np.asarray(root.index("snp.position").read(), dtype=np.int32)
        chrom = np.asarray(root.index("snp.chromosome").read()).astype(str)
        alle = np.asarray(root.index("snp.allele").read()).astype(str)
        rsid = (np.asarray(root.index("snp.rs.id").read()).astype(str)
                if root.exist("snp.rs.id") else np.array([""] * len(pos)))
        geno = np.asarray(root.index("genotype").read())   # ref dosage
    finally:
        g.close()

    ns, nv = len(sid), len(pos)
    # orient genotype to [sample, snp]
    if geno.shape != (ns, nv):
        geno = geno.T
    # ref dosage -> diploid allele indices (2->ref/ref, 1->het, 0->alt/alt, 3->miss)
    #   slot0 = [1,0,0,3][dosage]   slot1 = [1,1,0,3][dosage]
    g0 = np.array([1, 0, 0, 3], dtype=np.uint8)[geno]
    g1 = np.array([1, 1, 0, 3], dtype=np.uint8)[geno]
    graw = np.stack([g0, g1], axis=2)                # (sample, snp, ploidy)
    graw = np.ascontiguousarray(np.transpose(graw, (1, 0, 2)))  # (snp, sample, ploidy)

    allele = [a.replace("/", ",") for a in alle]
    _make_seq_skeleton(out_fn, sid, chrom, pos, rsid, allele, graw, compress)
    if verbose:
        print(f"seqSNP2GDS: {nv} variants x {ns} samples -> {out_fn}")
    return str(out_fn)
