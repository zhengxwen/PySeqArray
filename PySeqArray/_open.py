"""Opening / closing / summary of SeqArray GDS files."""

import os
import numpy as np
import pygds

from ._types import TSeqGDSFile

REQUIRED_NODES = ("sample.id", "variant.id", "position", "chromosome",
                  "allele", "genotype")


def _node_len(gds, path):
    """Length of a 1-D node (its only / last dim), 0 if empty."""
    d = gds.index(path, False).description()["dim"]
    return int(d[-1]) if len(d) else 0


def seqOpen(filename, readonly=True, allow_dup=False):
    """Open a SeqArray GDS file.

    Validates that the required SeqArray variables are present, reads the
    sample/variant dimensions and ploidy, and initialises the selection to
    *all* samples and variants.

    Parameters
    ----------
    filename : str
    readonly : bool, default True
    allow_dup : bool, default False
        Allow opening the same file more than once concurrently.

    Returns
    -------
    TSeqGDSFile
    """
    gds = pygds.gdsfile()
    gds.open(str(filename), readonly=readonly, allow_dup=allow_dup)
    try:
        names = set(gds.root().ls())
        for v in REQUIRED_NODES:
            if v not in names:
                raise ValueError(
                    f"'{filename}' is not a SeqArray GDS file: "
                    f"required node '{v}' is missing.")
        nsamp = _node_len(gds, "sample.id")
        nvar = _node_len(gds, "variant.id")
        # genotype/data numpy dims are [variant, sample, ploidy]; ploidy is last.
        gdim = gds.index("genotype/data", False).description()["dim"]
        ploidy = int(gdim[-1]) if len(gdim) >= 1 else 2
        f = TSeqGDSFile(gds, str(filename), readonly, nsamp, nvar, ploidy,
                        np.ones(nsamp, dtype=bool), np.ones(nvar, dtype=bool),
                        [])
    except Exception:
        gds.close()
        raise
    return f


def seqClose(f):
    """Close the underlying GDS file."""
    f.gds.close()


def seqExampleFileName(kind="gds"):
    """Path to a bundled SeqArray example file.

    ``kind`` is ``"gds"``, ``"1KG"``, or ``"vcf"``.  Looks under the sibling
    ``SeqArray/inst/extdata`` directory if present.
    """
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "..", "SeqArray", "inst", "extdata")
    name = {"gds": "CEU_Exon.gds",
            "1KG": "1KG_phase1_release_v3_chr22.gds",
            "vcf": "CEU_Exon.vcf.gz"}.get(kind)
    if name is None:
        raise ValueError(f"unknown example kind '{kind}'")
    return os.path.normpath(os.path.join(base, name))


def _child_names(f, path):
    """Names of child nodes under a folder (excluding @- and ~-index nodes)."""
    n = f._node(path, silent=True)
    if n is None:
        return []
    return [c for c in n.ls() if not c.startswith("@") and not c.startswith("~")]


def seqSummary(f, verbose=True):
    """Summarise a SeqArray GDS file.

    Returns a dict with sample/variant/ploidy counts, the current selection
    sizes, and the available ``annotation/info`` and ``annotation/format``
    field names.
    """
    info_names = _child_names(f, "annotation/info")
    fmt_names = _child_names(f, "annotation/format")
    s = dict(nsamp=f.nsamp, nvar=f.nvar, ploidy=f.ploidy,
             num_selected_samples=f.n_sel_sample(),
             num_selected_variants=f.n_sel_variant(),
             info=info_names, format=fmt_names)
    if verbose:
        print("SeqArray GDS:", f.filename)
        print("  # of samples: ", f.nsamp)
        print("  # of variants:", f.nvar)
        print("  ploidy:       ", f.ploidy)
        print("  annotation/info:  ", ", ".join(info_names) or "<none>")
        print("  annotation/format:", ", ".join(fmt_names) or "<none>")
    return s
