"""Write / maintenance operations: seqAddValue, seqDelete, seqRecompress.

These are plain GDS-tree edits performed through pygds; they do not need the
SeqArray engine.  ``seqAddValue`` / ``seqDelete`` require the file to have been
opened with ``readonly=False``.
"""

import numpy as np
import pygds


def seqAddValue(f, varname, val, compress="LZMA_RA", replace=False,
                visible=True):
    """Add ``val`` at GDS path ``varname`` (creating intermediate folders).

    The file ``f`` must have been opened with ``seqOpen(..., readonly=False)``.
    """
    node = f.gds.root()
    parts = varname.split("/")
    for p in parts[:-1]:
        node = node.index(p) if node.exist(p) else node.addfolder(p)
    leaf = parts[-1]
    if replace and node.exist(leaf):
        node.delete(leaf, force=True)
    node.add(leaf, val=val, compress=compress, visible=visible)
    f.gds.sync()
    return f


def seqDelete(f, info_var=None, format_var=None, sample_var=None):
    """Delete annotation ``info``/``format`` or ``sample.annotation`` variables.

    The file ``f`` must have been opened with ``seqOpen(..., readonly=False)``.
    """
    root = f.gds.root()

    def rm(path):
        if root.exist(path):
            root.index(path).delete(force=True)

    for v in (info_var or []):
        rm("annotation/info/" + v)
        rm("annotation/info/@" + v)
    for v in (format_var or []):
        rm("annotation/format/" + v)
    for v in (sample_var or []):
        rm("sample.annotation/" + v)
    f.gds.sync()
    return f


def _walk_arrays(node, fn):
    for nm in node.ls():
        try:
            child = node.index(nm)
            if child.description().get("type") == "Folder":
                _walk_arrays(child, fn)
            else:
                fn(child)
        except Exception:
            pass


def seqRecompress(filename, compress="LZMA_RA", verbose=True):
    """Recompress every array node of a GDS file, then defragment it."""
    g = pygds.gdsfile()
    g.open(str(filename), readonly=False)
    try:
        n = [0]

        def recompress(node):
            try:
                node.compression(compress)
                n[0] += 1
            except Exception:
                pass
        _walk_arrays(g.root(), recompress)
        g.sync()
    finally:
        g.close()
    pygds.cleanup_gds(str(filename), verbose=False)   # defragment
    if verbose:
        print(f"seqRecompress: recompressed {n[0]} nodes in {filename}")
    return str(filename)


def _make_transpose(root, folder, storage, compress, verbose):
    """(Re)build the sample-major ``~data`` twin next to ``<folder>/data``.

    ``genotype/data`` is numpy ``[page, sample, ploidy]`` -> twin ``[sample, page,
    ploidy]`` (swap the first two axes); ``phase/data`` is ``[variant, sample]``
    -> twin ``[sample, variant]`` (plain transpose).  Values are copied verbatim
    (the same bit2/bit1 codes), only the layout changes.
    """
    dpath = folder + "/data"
    if not root.exist(dpath):
        return
    raw = np.asarray(root.index(dpath).read())
    tdata = (np.transpose(raw, (1, 0, 2)) if raw.ndim == 3 else raw.T)
    tdata = np.ascontiguousarray(tdata)
    tpath = folder + "/~data"
    if root.exist(tpath):
        root.index(tpath).delete(force=True)
    root.index(folder).add("~data", val=tdata, storage=storage,
                           compress=compress)
    if verbose:
        print(f"  transpose {tpath} {tuple(tdata.shape)}")


def seqTranspose(filename, compress="LZMA_RA", verbose=True):
    """(Re)build the transposed ``~data`` twins for ``genotype`` and ``phase``.

    These sample-major copies accelerate by-sample iteration in tools (e.g. R
    SeqArray) that look for them.  Operates in place, then defragments.
    """
    g = pygds.gdsfile()
    g.open(str(filename), readonly=False)
    try:
        if verbose:
            print(f"Transpose {filename}")
        root = g.root()
        _make_transpose(root, "genotype", "bit2", compress, verbose)
        _make_transpose(root, "phase", "bit1", compress, verbose)
        g.sync()
    finally:
        g.close()
    pygds.cleanup_gds(str(filename), verbose=False)
    return str(filename)
