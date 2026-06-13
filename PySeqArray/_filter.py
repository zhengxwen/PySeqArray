"""Selection / filtering of samples and variants."""

import numpy as np


def _to_mask(sel, n, current):
    """Turn a user selector into a boolean mask of length ``n``.

    ``sel`` may be ``None`` (keep ``current``), a boolean vector of length
    ``n``, or a sequence/range of 0-based integer indices.
    """
    if sel is None:
        return current.copy()
    arr = np.asarray(sel)
    if arr.dtype == bool:
        if len(arr) != n:
            raise ValueError(f"logical selection length {len(arr)} != {n}")
        return arr.astype(bool, copy=True)
    if np.issubdtype(arr.dtype, np.integer):
        m = np.zeros(n, dtype=bool)
        if arr.size and (arr.min() < 0 or arr.max() >= n):
            raise IndexError(f"index out of range 0:{n}")
        m[arr] = True
        return m
    raise TypeError(f"unsupported selection type {arr.dtype}")


def _report_filter(f):
    print(f"# of selected samples: {f.n_sel_sample()}\n"
          f"# of selected variants: {f.n_sel_variant()}")


def seqSetFilter(f, sample_sel=None, variant_sel=None,
                 sample_id=None, variant_id=None,
                 action="set", verbose=True):
    """Set the sample and/or variant selection.

    Parameters
    ----------
    sample_sel, variant_sel : bool vector, integer indices, or None
        Selection over the *full* file (used with ``action`` ``"set"`` /
        ``"intersect"``).
    sample_id, variant_id : values, optional
        Select by matching values of ``sample.id`` / ``variant.id``.
    action : str
        ``"set"`` (replace), ``"intersect"`` (AND with current), ``"push"``
        (save current then set), or ``"pop"`` (restore last pushed).
    """
    if action == "push":
        seqFilterPush(f)
    elif action == "pop":
        return seqFilterPop(f, verbose=verbose)

    if sample_id is not None:
        ids = f._node("sample.id").read()
        want = set(np.asarray(sample_id).tolist())
        sample_sel = np.array([i for i, v in enumerate(ids) if v in want],
                              dtype=np.int64)
    if variant_id is not None:
        ids = f._node("variant.id").read()
        want = set(np.asarray(variant_id).tolist())
        variant_sel = np.array([i for i, v in enumerate(ids) if v in want],
                               dtype=np.int64)

    if sample_sel is not None:
        m = _to_mask(sample_sel, f.nsamp, f.sample_sel)
        f.sample_sel = (f.sample_sel & m) if action == "intersect" else m
    if variant_sel is not None:
        m = _to_mask(variant_sel, f.nvar, f.variant_sel)
        f.variant_sel = (f.variant_sel & m) if action == "intersect" else m
    if verbose:
        _report_filter(f)
    return f


def seqResetFilter(f, sample=True, variant=True, verbose=True):
    """Reset selection to all samples and/or all variants."""
    if sample:
        f.sample_sel = np.ones(f.nsamp, dtype=bool)
    if variant:
        f.variant_sel = np.ones(f.nvar, dtype=bool)
    if verbose:
        _report_filter(f)
    return f


def seqGetFilter(f):
    """Return copies of the current ``(sample_sel, variant_sel)`` masks."""
    return f.sample_sel.copy(), f.variant_sel.copy()


def seqFilterPush(f):
    """Save the current selection on an internal stack."""
    f.filter_stack.append((f.sample_sel.copy(), f.variant_sel.copy()))
    return f


def seqFilterPop(f, verbose=True):
    """Restore the most recently pushed selection."""
    if not f.filter_stack:
        raise RuntimeError("filter stack is empty")
    f.sample_sel, f.variant_sel = f.filter_stack.pop()
    if verbose:
        _report_filter(f)
    return f


def seqSetFilterChrom(f, include=None, verbose=True):
    """Restrict the variant selection to the given chromosome(s).

    ``include`` is a string or a sequence of strings; the result is intersected
    with the current variant selection.
    """
    if include is None:
        return f
    chr_ = np.asarray(f._node("chromosome").read()).astype(str)
    want = {include} if isinstance(include, str) else {str(x) for x in include}
    m = np.array([c in want for c in chr_], dtype=bool)
    f.variant_sel = f.variant_sel & m
    if verbose:
        _report_filter(f)
    return f


def seqSetFilterPos(f, chrom, pos, verbose=True):
    """Select variants matching the given ``chrom``/``pos`` pairs.

    ``chrom`` and ``pos`` may be scalars or equal-length sequences; the result
    is intersected with the current selection.
    """
    ch = np.asarray(f._node("chromosome").read()).astype(str)
    ps = np.asarray(f._node("position").read())
    pos_list = pos if isinstance(pos, (list, tuple, np.ndarray)) else [pos]
    if isinstance(chrom, (list, tuple, np.ndarray)):
        cc = [str(x) for x in chrom]
    else:
        cc = [str(chrom)] * len(pos_list)
    pp = list(pos_list) if isinstance(pos, (list, tuple, np.ndarray)) else [pos] * len(cc)
    want = set(zip(cc, pp))
    m = np.array([(c, int(p)) in want for c, p in zip(ch, ps)], dtype=bool)
    f.variant_sel = f.variant_sel & m
    if verbose:
        _report_filter(f)
    return f
