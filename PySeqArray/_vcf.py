"""seqVCF2GDS — import a VCF (plain or .gz) into a SeqArray GDS file.

The VCF *header* is parsed here (Python) to learn the samples, the INFO/FORMAT
field definitions and the ploidy; the GDS skeleton is created via pygds with the
exact node layout SeqArray's engine expects; then the VCF *body* is streamed into
that skeleton by the C++ engine (``cclib.vcf_parse`` -> ``SEQ_VCF_Parse``), which
reads the file through the gzFile connection layer and appends decoded records via
``GDS_Array_AppendData``.  ``SEQ_VCF_Parse`` skips the header itself.
"""

import gzip
import re

import numpy as np
import pygds

from . import cclib


def _reshape(root, path, dim):
    """Set the dimensions of a node whose data was appended as a flat stream."""
    root.index(path).setdim(dim)

# Type -> engine field-type code (FIELD_TYPE_* in ConvVCF2GDS.cpp)
_TYPE_CODE = {"integer": 1, "float": 2, "flag": 3, "character": 4, "string": 4}
_TYPE_STORAGE = {1: "int32", 2: "float", 3: "bit1", 4: "string"}
# Number special values -> negative int_num
_NUM_SPECIAL = {".": -1, "A": -2, "G": -3, "R": -4}

_META_RE = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|[^,]*)')


def _parse_meta(line):
    """Parse a ``##INFO=<...>`` / ``##FORMAT=<...>`` metadata line into a dict."""
    inner = line[line.index("<") + 1: line.rstrip().rindex(">")]
    d = {}
    for m in _META_RE.finditer(inner):
        k, v = m.group(1), m.group(2)
        if v.startswith('"'):
            v = v[1:-1]
        d[k] = v
    return d


def _open_text(path):
    return (gzip.open(path, "rt") if str(path).endswith(".gz")
            else open(path, "rt"))


def _parse_header(path):
    """Return (samples, info_defs, format_defs, ploidy)."""
    samples, info, fmt = [], [], []
    with _open_text(path) as fh:
        for line in fh:
            if line.startswith("##INFO="):
                info.append(_parse_meta(line))
            elif line.startswith("##FORMAT="):
                fmt.append(_parse_meta(line))
            elif line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
                break
            elif not line.startswith("#"):
                break
    return samples, info, fmt, 2   # ploidy defaults to 2 (diploid)


def _int_num(number):
    """Map a VCF Number field to SeqArray's int_num code."""
    if number in _NUM_SPECIAL:
        return _NUM_SPECIAL[number]
    try:
        return int(number)
    except (TypeError, ValueError):
        return -1


def _field_table(defs):
    """Build the engine's per-field column lists from parsed INFO/FORMAT defs."""
    ID, int_type, int_num, flag, Number, Type, Desc = [], [], [], [], [], [], []
    for d in defs:
        ID.append(d.get("ID", ""))
        t = _TYPE_CODE.get(d.get("Type", "string").lower(), 4)
        int_type.append(t)
        int_num.append(_int_num(d.get("Number", ".")))
        flag.append(True)
        Number.append(d.get("Number", "."))
        Type.append(d.get("Type", "String"))
        Desc.append(d.get("Description", ""))
    return dict(ID=ID, int_type=int_type, int_num=int_num,
                **{"import.flag": flag}, Number=Number, Type=Type,
                Description=Desc)


def _add(node, name, storage, valdim=None, visible=True):
    return node.add(name, storage=storage, valdim=valdim, visible=visible)


def _make_skeleton(gfile, samples, info_tab, fmt_tab, ploidy):
    """Create the SeqArray GDS node skeleton in the (empty) output file.

    NOTE on valdim: pygds uses CoreArray-native dimension order (no R-style
    reversal), so the extensible (variant) dimension comes FIRST.  R's
    ``valdim=c(ploidy, nSamp, 0)`` reverses to CoreArray ``[0, nSamp, ploidy]``.
    """
    root = gfile.root()
    nsamp = len(samples)

    root.add("sample.id", val=list(samples), storage="string")
    _add(root, "variant.id", "int32")
    _add(root, "chromosome", "string")
    _add(root, "position", "int32")
    _add(root, "allele", "string")

    geno = root.addfolder("genotype")
    geno.putattr("VariableName", "GT")
    geno.putattr("Description", "Genotype")
    _add(geno, "data", "bit2", valdim=[0, nsamp, ploidy])
    _add(geno, "@data", "uint8", visible=False)
    ei = _add(geno, "extra.index", "int32", valdim=[0, 3])
    ei.putattr("R.colnames", ["sample.index", "variant.index", "length"])
    _add(geno, "extra", "int16")

    phase = root.addfolder("phase")
    dm = [0, nsamp, ploidy - 1] if ploidy > 2 else [0, nsamp]
    _add(phase, "data", "bit1", valdim=dm)
    pi = _add(phase, "extra.index", "int32", valdim=[0, 3])
    pi.putattr("R.colnames", ["sample.index", "variant.index", "length"])
    _add(phase, "extra", "bit1")

    annot = root.addfolder("annotation")
    _add(annot, "id", "string")
    _add(annot, "qual", "float")
    filt = _add(annot, "filter", "int32")
    filt.putattr("R.class", "factor")
    filt.putattr("R.levels", ["PASS"])

    info = annot.addfolder("info")
    for i, fid in enumerate(info_tab["ID"]):
        t = info_tab["int_type"][i]
        num = info_tab["int_num"][i]
        _add(info, fid, _TYPE_STORAGE[t])
        # a length index @<F> is needed unless the field is exactly one fixed
        # value per variant: Flag (t==3) and Integer/Float with Number==1 are
        # fixed; strings (t==4) and any non-1 Number are variable-length.
        if not (t == 3 or (num == 1 and t != 4)):
            _add(info, "@" + fid, "int32", visible=False)

    fmt = annot.addfolder("format")
    for i, fid in enumerate(fmt_tab["ID"]):
        t = fmt_tab["int_type"][i]
        sub = fmt.addfolder(fid)
        sub.putattr("Number", fmt_tab["Number"][i])
        sub.putattr("Type", fmt_tab["Type"][i])
        sub.putattr("Description", fmt_tab["Description"][i])
        _add(sub, "data", _TYPE_STORAGE[t], valdim=[0, nsamp])
        _add(sub, "@data", "int32", visible=False)


def seqVCF2GDS(vcf_fn, out_fn, genotype_var_name="GT", verbose=True):
    """Convert a VCF file ``vcf_fn`` to a SeqArray GDS file ``out_fn``.

    Returns the path to the created GDS file.
    """
    samples, info_defs, fmt_defs, ploidy = _parse_header(vcf_fn)
    # the genotype FORMAT field is handled separately, not as an annotation
    fmt_defs = [d for d in fmt_defs if d.get("ID") != genotype_var_name]
    info_tab = _field_table(info_defs)
    fmt_tab = _field_table(fmt_defs)
    nsamp = len(samples)

    gfile = pygds.gdsfile()
    gfile.create(str(out_fn))
    try:
        gfile.root().putattr("FileFormat", "SEQ_ARRAY")
        _make_skeleton(gfile, samples, info_tab, fmt_tab, ploidy)
        gfile.sync()

        header = dict(ploidy=ploidy, info=info_tab, format=fmt_tab)
        param = {
            "sample.num": len(samples),
            "genotype.var.name": genotype_var_name,
            "raise.error": True,
            "start": 1.0,
            "count": 1e18,
            "infile": str(vcf_fn),
            "chr.prefix": "",
            "progfile": None,
            "use.file": True,
            "filter.levels": ["PASS"],
            "verbose": bool(verbose),
        }
        cclib.vcf_parse(str(vcf_fn), header, gfile.fileid, param)

        # The engine appends genotype/phase/format data as a flat stream (pygds's
        # valdim with a leading extensible 0 doesn't fix the trailing dims), so
        # restore the multi-dimensional shapes now that the data is written.  The
        # flat order is variant/page-major then sample then ploidy, so a direct
        # reshape to the CoreArray dims (extensible dim first) is correct.
        root = gfile.root()
        nvar = root.index("variant.id").description()["dim"][0]
        npages = int(np.asarray(root.index("genotype/@data").read()).sum()) if nvar else 0
        _reshape(root, "genotype/data", [npages, nsamp, ploidy])
        if root.exist("phase/data"):
            pdm = [nvar, nsamp, ploidy - 1] if ploidy > 2 else [nvar, nsamp]
            _reshape(root, "phase/data", pdm)
        if root.exist("annotation/format"):
            for fid in root.index("annotation/format").ls():
                node = root.index("annotation/format/" + fid + "/data")
                total = int(np.prod(node.description()["dim"]))
                if nsamp and total % nsamp == 0:
                    _reshape(root, "annotation/format/" + fid + "/data",
                             [total // nsamp, nsamp])
        gfile.sync()
        if verbose:
            print(f"seqVCF2GDS: {nvar} variants, {len(samples)} samples -> {out_fn}")
    finally:
        gfile.close()
    return str(out_fn)
