"""seqApply / seqBlockApply: iterate a function over selected variants/samples."""

import numpy as np

from ._getdata import seqGetData
from ._filter import seqFilterPush, seqFilterPop


def _fetch(f, name):
    """Fetch the requested field(s) under the current selection."""
    if isinstance(name, str):
        return seqGetData(f, name)
    return tuple(seqGetData(f, n) for n in name)


def _collect_result(results, as_is):
    if as_is == "none":
        return None
    if as_is == "list":
        return results
    if as_is == "unlist":
        if not results:
            return np.array([])
        return np.concatenate([np.atleast_1d(np.asarray(r)).ravel()
                               for r in results])
    raise ValueError(f"unsupported as_is={as_is!r} (use 'list', 'unlist', 'none')")


def seqApply(fun, f, name, margin="by.variant", as_is="list", bsize=1,
             **kwargs):
    """Apply ``fun`` over the selected variants (or samples).

    One *unit* of ``bsize`` variants is processed per call.  ``name`` is a field
    name or a list of field names; ``fun`` receives the corresponding data (a
    single value, or a tuple when ``name`` is a list).  Extra ``kwargs`` are
    forwarded to ``fun``.

    Parameters
    ----------
    margin : ``"by.variant"`` (default) or ``"by.sample"``.
    as_is : ``"list"`` (collect), ``"unlist"`` (concatenate), or ``"none"``.
    bsize : variants/samples per call.
    """
    by_variant = margin == "by.variant"
    if not by_variant and margin != "by.sample":
        raise ValueError('margin must be "by.variant" or "by.sample"')
    units = f.sel_variant_idx() if by_variant else f.sel_sample_idx()
    seqFilterPush(f)
    results = []
    try:
        i = 0
        n = len(units)
        while i < n:
            blk = units[i:i + bsize]
            if by_variant:
                m = np.zeros(f.nvar, dtype=bool)
                m[blk] = True
                f.variant_sel = m
            else:
                m = np.zeros(f.nsamp, dtype=bool)
                m[blk] = True
                f.sample_sel = m
            data = _fetch(f, name)
            r = fun(*data, **kwargs) if isinstance(data, tuple) else fun(data, **kwargs)
            if as_is != "none":
                results.append(r)
            i += bsize
    finally:
        seqFilterPop(f, verbose=False)
    return _collect_result(results, as_is)


def seqBlockApply(fun, f, name, margin="by.variant", as_is="list", bsize=1024,
                  **kwargs):
    """Like :func:`seqApply` but processes ``bsize`` units per call (default 1024)."""
    return seqApply(fun, f, name, margin=margin, as_is=as_is, bsize=bsize,
                    **kwargs)
