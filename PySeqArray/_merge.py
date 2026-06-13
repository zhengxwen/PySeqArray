"""seqMerge — variant-wise merge of several SeqArray GDS files.

Basic merge: concatenate the *variants* of several SeqArray GDS files that share
an identical ``sample.id``, producing a new valid SeqArray GDS file.  The required
variables (sample.id, variant.id, position, chromosome, allele, genotype) plus
``phase`` (when present in all inputs) are merged.  Annotation info/format fields
are not merged in this basic version (a warning is emitted if any are present).

Pure Python over :mod:`pygds` (the SeqArray engine is not needed): raw genotype
``bit2`` page values are self-contained per variant, so concatenating the page
axis of ``genotype/data`` together with ``genotype/@data`` preserves the decoding
— multi-allelic variants merge correctly without re-encoding.
"""

import warnings

import numpy as np
import pygds


def _read(g, path):
    return np.asarray(g.root().index(path).read())


def seqMerge(infiles, outfile, compress="LZMA_RA", verbose=True):
    """Merge the variants of ``infiles`` (SeqArray GDS paths) into ``outfile``.

    All inputs must share an identical ``sample.id`` (variant-wise merge).
    Returns ``outfile``.
    """
    infiles = [str(p) for p in infiles]
    if not infiles:
        raise ValueError("seqMerge: need at least one input file")

    gs = [pygds.gdsfile() for _ in infiles]
    for g, fn in zip(gs, infiles):
        g.open(fn, allow_dup=True)
    try:
        sid = _read(gs[0], "sample.id")
        for g in gs[1:]:
            if not np.array_equal(_read(g, "sample.id"), sid):
                raise ValueError("seqMerge: all inputs must share the same "
                                 "sample.id (variant-wise merge only)")
        if verbose and any(g.root().exist("annotation/info") and
                           g.root().index("annotation/info").ls()
                           for g in gs):
            warnings.warn("seqMerge (basic): annotation/info and "
                          "annotation/format fields are not merged")

        position = np.concatenate([_read(g, "position") for g in gs])
        chrom = np.concatenate([_read(g, "chromosome").astype(str) for g in gs])
        allele = np.concatenate([_read(g, "allele").astype(str) for g in gs])
        ntot = len(position)
        if verbose:
            print(f"seqMerge: {len(gs)} files -> {outfile} "
                  f"({len(sid)} samples, {ntot} variants)")

        # genotype/data: numpy [page, sample, ploidy] -> concatenate page axis
        graw = np.ascontiguousarray(
            np.concatenate([_read(g, "genotype/data") for g in gs], axis=0))
        atdata = np.concatenate(
            [_read(g, "genotype/@data") for g in gs]).astype(np.uint8)

        have_phase = all(g.root().exist("phase/data") for g in gs)
        if have_phase:
            praw = np.ascontiguousarray(
                np.concatenate([_read(g, "phase/data") for g in gs], axis=0))

        w = pygds.gdsfile()
        w.create(str(outfile))
        try:
            root = w.root()
            root.putattr("FileFormat", "SEQ_ARRAY")
            desc = root.addfolder("description")
            desc.putattr("vcf.fileformat", "VCFv4.2")

            root.add("sample.id", val=list(np.asarray(sid).astype(str)),
                     storage="string", compress=compress)
            root.add("variant.id", val=np.arange(1, ntot + 1, dtype=np.int32),
                     storage="int32", compress=compress)
            root.add("position", val=position.astype(np.int32),
                     storage="int32", compress=compress)
            root.add("chromosome", val=list(chrom), storage="string",
                     compress=compress)
            root.add("allele", val=list(allele), storage="string",
                     compress=compress)

            geno = root.addfolder("genotype")
            geno.putattr("VariableName", "GT")
            geno.putattr("Description", "Genotype")
            geno.add("data", val=graw, storage="bit2", compress=compress)
            geno.add("@data", val=atdata, storage="uint8", compress=compress,
                     visible=False)

            if have_phase:
                phase = root.addfolder("phase")
                phase.add("data", val=praw, storage="bit1", compress=compress)

            root.addfolder("annotation")
            w.sync()
        finally:
            w.close()
    finally:
        for g in gs:
            g.close()
    return str(outfile)
