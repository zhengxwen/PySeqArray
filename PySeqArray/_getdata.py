"""seqGetData: read a named SeqArray field under the current selection."""

import numpy as np

from ._types import TVarData, _offsets
from ._genotype import _read_genotype

_VARIANT_FIXED = ("variant.id", "position", "chromosome", "allele",
                  "annotation/id", "annotation/qual", "annotation/filter")


def seqGetData(f, name):
    """Read the field ``name`` for the currently selected samples and variants.

    Supported ``name`` values:

    - Variant-level: ``"variant.id"``, ``"position"``, ``"chromosome"``,
      ``"allele"``, ``"annotation/id"``, ``"annotation/qual"``,
      ``"annotation/filter"``.
    - Sample-level: ``"sample.id"``, ``"sample.annotation/<F>"``.
    - Genotype: ``"genotype"`` -> masked array ``(ploidy, sample, variant)``.
    - Pseudo-fields: ``"$dosage"``, ``"$dosage_alt"``, ``"$num_allele"``,
      ``"$ref"``, ``"$alt"``, ``"$chrom_pos"``, ``"$variant_index"``,
      ``"$sample_index"``.
    - ``"phase"``.
    - ``"annotation/info/<F>"``, ``"annotation/format/<F>"`` (variable-length
      fields return a :class:`TVarData`).
    """
    # ---- sample-level ------------------------------------------------------
    if name == "sample.id":
        return _read_sel(f, "sample.id", f.sample_sel)
    if name == "$sample_index":
        return f.sel_sample_idx()

    # ---- variant-level fixed ----------------------------------------------
    if name in _VARIANT_FIXED:
        return _read_sel(f, name, f.variant_sel)
    if name == "$variant_index":
        return f.sel_variant_idx()

    # ---- derived from allele ----------------------------------------------
    if name == "$num_allele":
        al = _read_sel(f, "allele", f.variant_sel)
        return np.array([str(a).count(",") + 1 for a in al], dtype=np.int32)
    if name == "$ref":
        al = _read_sel(f, "allele", f.variant_sel)
        return np.array([str(a).split(",", 1)[0] for a in al], dtype=object)
    if name == "$alt":
        al = _read_sel(f, "allele", f.variant_sel)
        return np.array([(str(a).split(",", 1) + [""])[1] for a in al],
                        dtype=object)
    if name == "$chrom_pos":
        ch = _read_sel(f, "chromosome", f.variant_sel)
        ps = _read_sel(f, "position", f.variant_sel)
        return np.array([f"{c}:{p}" for c, p in zip(ch, ps)], dtype=object)

    # ---- genotype & dosage -------------------------------------------------
    if name == "genotype":
        return _read_genotype(f)
    if name == "$dosage":
        return _dosage(f, ref=True)
    if name == "$dosage_alt":
        return _dosage(f, ref=False)
    if name == "phase":
        return _read_phase(f)

    # ---- annotation --------------------------------------------------------
    if name.startswith("sample.annotation/"):
        return _read_sel(f, name, f.sample_sel)
    if name.startswith("annotation/info/"):
        return _read_info(f, name)
    if name.startswith("annotation/format/"):
        field = name[len("annotation/format/"):]
        if field.endswith("/data"):
            field = field[:-len("/data")]
        return _read_format(f, field)

    raise ValueError(f"seqGetData: unsupported field name '{name}'")


def _read_sel(f, path, mask):
    """Read a 1-D node keeping only selected entries (select-while-read)."""
    node = f._node(path)
    return np.asarray(node.readex([mask]))


def _dosage(f, ref):
    """Per (sample, variant) count of reference (``ref=True``) or alternate
    alleles; masked where any allele in that call is missing.

    Returns a masked array of shape ``(sample, variant)``.
    """
    g = _read_genotype(f)                       # (ploidy, sample, variant)
    p = g.shape[0]
    data = np.ma.getdata(g)
    miss = np.ma.getmaskarray(g)
    nref = (data == 0).sum(axis=0)              # (sample, variant)
    any_miss = miss.any(axis=0)
    out = nref if ref else (p - nref)
    return np.ma.MaskedArray(out.astype(np.int64), mask=any_miss)


def _read_phase(f):
    """Phase: Bit1, CoreArray [sample, variant] -> numpy [variant, sample]."""
    node = f._node("phase/data", silent=True)
    if node is None:
        return None
    # numpy dims [variant, sample]; sel in that order, result (nvar, nsamp).
    return np.asarray(node.readex([f.variant_sel, f.sample_sel]))


def _read_info(f, path):
    """annotation/info/<F>, possibly variable-length via sibling ``@<F>``."""
    node = f._node(path)
    field = path.rsplit("/", 1)[1]
    parent = path[:len(path) - len(field) - 1]      # "annotation/info"
    idxnode = f._node(f"{parent}/@{field}", silent=True)
    svar = f.sel_variant_idx()
    if idxnode is None:
        return np.asarray(node.readex([f.variant_sel]))
    cnt = np.asarray(idxnode.read(), dtype=np.int64)
    return _gather_varlen(node, cnt, svar)


def _read_format(f, field):
    """annotation/format/<F>: data is CoreArray [sample, total_cols].

    numpy dims are [total_cols, sample]; ``@data`` gives the column count each
    variant contributes.
    """
    base = f"annotation/format/{field}"
    node = f._node(f"{base}/data")
    idxnode = f._node(f"{base}/@data", silent=True)
    svar = f.sel_variant_idx()
    cnt = (np.ones(f.nvar, dtype=np.int64) if idxnode is None
           else np.asarray(idxnode.read(), dtype=np.int64))
    colstart = _offsets(cnt)
    cols = []
    for v in svar:
        for k in range(int(cnt[v])):
            cols.append(int(colstart[v]) + k)
    total_cols = int(node.description()["dim"][0])
    col_mask = np.zeros(total_cols, dtype=bool)
    col_mask[cols] = True
    # numpy order [cols, sample]; result (ncols, n_sel_sample).
    mat = np.asarray(node.readex([col_mask, f.sample_sel]))
    if np.all(cnt[svar] == 1):
        # (variant, sample): one column per variant -> already that shape.
        return mat
    return TVarData(cnt[svar].astype(np.int32), mat)


def _gather_varlen(node, cnt, svar):
    """Gather variable-length 1-D data for the selected variants."""
    off = _offsets(cnt)
    full = np.asarray(node.read())
    lens = cnt[svar].astype(np.int32)
    if np.all(lens == 1):
        return full[off[svar]]
    pieces = [full[off[v]:off[v] + cnt[v]] for v in svar]
    return TVarData(lens, np.concatenate(pieces) if pieces else full[:0])
