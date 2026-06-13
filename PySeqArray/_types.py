"""Core types for PySeqArray.

A SeqArray GDS file is wrapped by :class:`TSeqGDSFile`, which carries the open
:mod:`pygds` file handle plus the current sample/variant *selection* (filter).

IMPORTANT axis-order note (pygds vs R/Julia):

``pygds`` returns arrays in **numpy C-order**, which is *reversed* relative to
the CoreArray/R column-major order.  For example ``genotype/data`` is stored
with CoreArray dims ``[ploidy, sample, variant]`` and printed by R as
``2 x 90 x 1348``; pygds reports its ``description()['dim']`` as
``[1348, 90, 2]`` (``[variant, sample, ploidy]``) and ``readex`` takes a list of
boolean masks in that same numpy order.  All low-level reads here use numpy
order; user-facing genotype results are transposed back to the R orientation
``(ploidy, sample, variant)`` so they cross-validate against R ``seqGetData``.
"""

import numpy as np


class TVarData:
    """Result of :func:`seqGetData` for a *variable-length* field.

    Mirrors SeqArray's ``SeqVarDataList`` and JSeqArray's ``TVarData``.

    Attributes
    ----------
    length : numpy.ndarray (int32)
        Number of entries contributed by each selected variant, in selection
        order.
    data : object
        The concatenated values across all selected variants.
    """

    __slots__ = ("length", "data")

    def __init__(self, length, data):
        self.length = np.asarray(length, dtype=np.int32)
        self.data = data

    def __repr__(self):
        n = len(self.data) if hasattr(self.data, "__len__") else type(self.data).__name__
        return f"TVarData({len(self.length)} variants, {n} values)"


class TSeqGDSFile:
    """Handle to an open SeqArray GDS file plus its current selection state.

    Users should treat the fields as read-only and mutate the selection through
    the ``seqSetFilter`` / ``seqResetFilter`` / ``seqFilter*`` API.

    Attributes
    ----------
    gds : pygds.gdsfile
        The underlying pygds file handle.
    filename : str
    readonly : bool
    nsamp, nvar, ploidy : int
        Full dimensions read from the file.
    sample_sel, variant_sel : numpy.ndarray (bool)
        Boolean masks marking the currently selected samples / variants.
    filter_stack : list[tuple[ndarray, ndarray]]
        Stack used by ``seqFilterPush`` / ``seqFilterPop``.
    """

    __slots__ = ("gds", "filename", "readonly", "nsamp", "nvar", "ploidy",
                 "sample_sel", "variant_sel", "filter_stack")

    def __init__(self, gds, filename, readonly, nsamp, nvar, ploidy,
                 sample_sel, variant_sel, filter_stack):
        self.gds = gds
        self.filename = filename
        self.readonly = readonly
        self.nsamp = nsamp
        self.nvar = nvar
        self.ploidy = ploidy
        self.sample_sel = sample_sel
        self.variant_sel = variant_sel
        self.filter_stack = filter_stack

    # ---- convenience accessors ---------------------------------------------

    def sel_sample_idx(self):
        """0-based indices of currently selected samples."""
        return np.nonzero(self.sample_sel)[0]

    def sel_variant_idx(self):
        """0-based indices of currently selected variants."""
        return np.nonzero(self.variant_sel)[0]

    def n_sel_sample(self):
        return int(self.sample_sel.sum())

    def n_sel_variant(self):
        return int(self.variant_sel.sum())

    # ---- node lookup -------------------------------------------------------

    def _node(self, path, silent=False):
        """Return the gdsnode at ``path``.

        With ``silent=True`` return ``None`` when the node is absent.  (pygds's
        own ``index(..., silent=True)`` would instead return the nearest
        existing parent, which is not what callers here want — so gate on
        ``exist`` first, exactly as the Julia ``_node`` helper does.)
        """
        if silent and not self.gds.root().exist(path):
            return None
        return self.gds.index(path, False)

    def __repr__(self):
        return (f"TSeqGDSFile: {self.filename}\n"
                f"  selection: {self.n_sel_sample()}/{self.nsamp} samples, "
                f"{self.n_sel_variant()}/{self.nvar} variants "
                f"(ploidy {self.ploidy})")


def _offsets(cnt):
    """Exclusive-prefix offsets: ``o[i]`` = number of entries before index ``i``."""
    cnt = np.asarray(cnt, dtype=np.int64)
    o = np.zeros(len(cnt), dtype=np.int64)
    if len(cnt):
        o[1:] = np.cumsum(cnt)[:-1]
    return o
