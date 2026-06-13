"""Write / maintenance operations: seqAddValue, seqDelete, seqRecompress.

These are plain GDS-tree edits performed through pygds; they do not need the
SeqArray engine.  ``seqAddValue`` / ``seqDelete`` require the file to have been
opened with ``readonly=False``.
"""

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
